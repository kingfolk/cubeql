//! Small labeled evaluator, not a production hybrid-search service. Literal matching scans the
//! saved source snapshot; experimental RRF combines its ranks with exact device cosine ranks.
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    time::Instant,
};

use cubecl::Device;
use cubeql::{Metric, SearchHit};
use cubeql_embeddings::{EmbeddingProvider, embed_query};
use serde::{Deserialize, Serialize};

use crate::{
    CodeError, CodeIndex, CodeSearchIndex, VectorPrecision, invalid,
    reporting::{CodeFilter, CodeHit, SnapshotInfo, elapsed_ms, materialize},
};

const RRF_CONSTANT: f64 = 60.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryCategory {
    Identifier,
    Conceptual,
    Behavioral,
    CrossFile,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelevantChunk {
    pub path: String,
    pub symbol: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryCase {
    pub id: String,
    pub category: QueryCategory,
    pub text: String,
    pub relevant: Vec<RelevantChunk>,
    #[serde(default)]
    pub filter: CodeFilter,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuerySet {
    pub version: u32,
    pub name: String,
    pub queries: Vec<QueryCase>,
}

impl QuerySet {
    /// Resolve each label to exactly one chunk. Reject stale/ambiguous/filtered-out labels
    /// instead of silently computing recall against a different corpus or denominator.
    pub fn resolve(&self, code: &CodeIndex) -> Result<Vec<BTreeSet<usize>>, CodeError> {
        if self.version != 1 || self.name.trim().is_empty() || self.queries.is_empty() {
            return Err(invalid(
                "evaluation needs version 1, a name and non-empty queries",
            ));
        }
        let mut ids = HashSet::new();
        self.queries
            .iter()
            .map(|query| {
                if query.id.trim().is_empty()
                    || !ids.insert(&query.id)
                    || query.text.trim().is_empty()
                    || query.relevant.is_empty()
                {
                    return Err(invalid(
                        "query IDs must be unique/non-empty; text and relevance labels are required",
                    ));
                }
                let mut relevant = BTreeSet::new();
                let mut paths = BTreeSet::new();
                for label in &query.relevant {
                    let rows = code.chunks().iter().enumerate()
                        .filter(|(_, c)| c.path == label.path && c.symbol == label.symbol)
                        .collect::<Vec<_>>();
                    if rows.len() != 1
                        || !query.filter.matches(rows[0].1)
                        || !relevant.insert(rows[0].0)
                    {
                        return Err(invalid(format!(
                            "{}: missing, ambiguous, duplicate or filtered-out label {}::{}",
                            query.id, label.path, label.symbol,
                        )));
                    }
                    paths.insert(&label.path);
                }
                if query.category == QueryCategory::CrossFile && paths.len() < 2 {
                    return Err(invalid(
                        "cross_file queries require relevant chunks in at least two files",
                    ));
                }
                Ok(relevant)
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RankingMetrics {
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub reciprocal_rank: f64,
    pub first_relevant_rank: Option<usize>,
}

pub fn ranking_metrics(
    rows: &[usize],
    relevant: &BTreeSet<usize>,
) -> Result<RankingMetrics, CodeError> {
    if relevant.is_empty() || rows.iter().collect::<HashSet<_>>().len() != rows.len() {
        return Err(invalid(
            "metrics need non-empty relevance and unique retrieved rows",
        ));
    }
    let recall = |k| {
        rows.iter()
            .take(k)
            .filter(|row| relevant.contains(row))
            .count() as f64
            / relevant.len() as f64
    };
    let first = rows
        .iter()
        .position(|row| relevant.contains(row))
        .map(|i| i + 1);
    Ok(RankingMetrics {
        recall_at_5: recall(5),
        recall_at_10: recall(10),
        reciprocal_rank: first.map_or(0.0, |rank| 1.0 / rank as f64),
        first_relevant_rank: first,
    })
}

/// Literal, case-sensitive substring baseline, not BM25 or ripgrep's entire feature set.
/// Exact qualified symbols rank first, unqualified symbols second, source substrings third;
/// equal scores use ascending snapshot row ID. Only saved chunks are searched.
pub fn literal_search(
    code: &CodeIndex,
    text: &str,
    filter: &CodeFilter,
    top_k: usize,
) -> Vec<SearchHit> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut hits = code
        .chunks()
        .iter()
        .enumerate()
        .filter(|(_, c)| filter.matches(c))
        .filter_map(|(row, c)| {
            let score = if c.symbol == text {
                3.0
            } else if c.symbol.rsplit("::").next() == Some(text) {
                2.0
            } else if c.source.contains(text) {
                1.0
            } else {
                return None;
            };
            Some(SearchHit { row, score })
        })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.row.cmp(&b.row)));
    hits.truncate(top_k);
    hits
}

/// Evaluation-only reciprocal rank fusion, one-based ranks, fixed k=60, equal source weights.
pub fn reciprocal_rank_fusion(lists: &[&[SearchHit]], top_k: usize) -> Vec<SearchHit> {
    let mut scores = BTreeMap::<usize, f64>::new();
    for list in lists {
        let mut seen = HashSet::new();
        for (rank, hit) in list.iter().enumerate() {
            if seen.insert(hit.row) {
                *scores.entry(hit.row).or_default() += 1.0 / (RRF_CONSTANT + rank as f64 + 1.0);
            }
        }
    }
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(top_k);
    ranked
        .into_iter()
        .map(|(row, score)| SearchHit {
            row,
            score: score as f32,
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Latency {
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub samples: usize,
}

fn latency(mut samples: Vec<f64>) -> Latency {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    let median = if n.is_multiple_of(2) {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    } else {
        samples[n / 2]
    };
    Latency {
        median_ms: median,
        min_ms: samples[0],
        max_ms: samples[n - 1],
        samples: n,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StrategyResult {
    pub metrics: RankingMetrics,
    /// Warm retrieval only: excludes embedding, loading, filter preparation/materialization.
    pub retrieval: Latency,
    pub materialization_ms: f64,
    /// At most 10 display hits; metrics/MRR use the full candidate window.
    pub top_hits: Vec<CodeHit>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QueryResult {
    pub id: String,
    pub category: QueryCategory,
    pub embedding_ms: f64,
    pub filter_ms: f64,
    /// First invocation for this case, not a fresh-process cold-start measurement.
    pub first_semantic_ms: f64,
    pub semantic: StrategyResult,
    pub literal: StrategyResult,
    pub experimental_rrf: StrategyResult,
}

#[derive(Clone, Debug, Serialize)]
pub struct QualitySummary {
    pub queries: usize,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
}

#[derive(Serialize)]
pub struct VariantReport {
    pub snapshot: SnapshotInfo,
    pub conversion_ms: f64,
    pub upload_ms: f64,
    pub queries: Vec<QueryResult>,
    pub overall: BTreeMap<String, QualitySummary>,
    pub by_category: BTreeMap<QueryCategory, BTreeMap<String, QualitySummary>>,
}

#[derive(Clone, Copy, Serialize)]
pub struct EvaluationOptions {
    pub candidate_k: usize,
    pub repeats: usize,
    pub compare_fp16: bool,
}
impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            candidate_k: 50,
            repeats: 3,
            compare_fp16: false,
        }
    }
}

#[derive(Serialize)]
pub struct EvaluationReport {
    pub format_version: u32,
    pub dataset: QuerySet,
    pub backend: String,
    pub options: EvaluationOptions,
    pub rrf_rank_constant: usize,
    pub embedding_total_ms: f64,
    pub variants: Vec<VariantReport>,
}

fn strategy(
    code: &CodeIndex,
    hits: &[SearchHit],
    relevant: &BTreeSet<usize>,
    samples: Vec<f64>,
) -> Result<StrategyResult, CodeError> {
    let metrics = ranking_metrics(&hits.iter().map(|h| h.row).collect::<Vec<_>>(), relevant)?;
    let start = Instant::now();
    let top_hits = materialize(code, &hits[..hits.len().min(10)], false)?;
    Ok(StrategyResult {
        metrics,
        retrieval: latency(samples),
        materialization_ms: elapsed_ms(start),
        top_hits,
    })
}

fn summarize(results: &[&QueryResult]) -> BTreeMap<String, QualitySummary> {
    ["semantic", "literal", "experimental_rrf"]
        .into_iter()
        .map(|name| {
            let metrics = results
                .iter()
                .map(|r| match name {
                    "semantic" => r.semantic.metrics,
                    "literal" => r.literal.metrics,
                    _ => r.experimental_rrf.metrics,
                })
                .collect::<Vec<_>>();
            let n = metrics.len() as f64;
            (
                name.into(),
                QualitySummary {
                    queries: metrics.len(),
                    recall_at_5: metrics.iter().map(|m| m.recall_at_5).sum::<f64>() / n,
                    recall_at_10: metrics.iter().map(|m| m.recall_at_10).sum::<f64>() / n,
                    mrr: metrics.iter().map(|m| m.reciprocal_rank).sum::<f64>() / n,
                },
            )
        })
        .collect()
}

/// Encode every labeled query once and reuse the exact query vectors across storage variants.
/// Variants execute sequentially; only one resident index is needed at a time. Model loading
/// and input-file reading are measured by the caller, not hidden inside retrieval latency.
pub fn evaluate(
    code: CodeIndex,
    device: &Device,
    provider: &mut dyn EmbeddingProvider,
    dataset: &QuerySet,
    options: EvaluationOptions,
) -> Result<EvaluationReport, CodeError> {
    if options.candidate_k < 10 || options.repeats == 0 {
        return Err(invalid(
            "evaluation needs candidate_k >= 10 and repeats >= 1",
        ));
    }
    let config = code
        .embedding_config()
        .ok_or_else(|| invalid("evaluation requires a provider-aware index"))?;
    config.ensure_compatible(provider.config())?;
    if options.compare_fp16 && code.precision() != VectorPrecision::Fp32 {
        return Err(invalid(
            "FP32/FP16 comparison requires an original FP32 index; widening FP16 cannot recover FP32 values",
        ));
    }
    let relevant = dataset.resolve(&code)?;
    let mut queries = Vec::new();
    let mut embedding_total_ms = 0.0;
    for case in &dataset.queries {
        let start = Instant::now();
        let query = embed_query(provider, &case.text)?;
        config.ensure_compatible(query.config())?;
        let ms = elapsed_ms(start);
        embedding_total_ms += ms;
        queries.push((query, ms));
    }
    let precisions = if options.compare_fp16 {
        vec![VectorPrecision::Fp32, VectorPrecision::Fp16]
    } else {
        vec![code.precision()]
    };
    let mut variants = Vec::new();
    for precision in precisions {
        let start = Instant::now();
        let variant = code.clone().with_precision(precision)?;
        let conversion_ms = elapsed_ms(start);
        let snapshot = SnapshotInfo::new(&variant)?;
        let start = Instant::now();
        let search = CodeSearchIndex::new(device, variant)?;
        let upload_ms = elapsed_ms(start);
        let mut results = Vec::new();
        for ((case, (query, embedding_ms)), relevant) in
            dataset.queries.iter().zip(&queries).zip(&relevant)
        {
            let predicates = case.filter.predicates();
            let start = Instant::now();
            let selected = search.prepare_filter(&predicates)?;
            let filter_ms = elapsed_ms(start);
            let start = Instant::now();
            let mut semantic = selected.query(query, options.candidate_k, Metric::Cosine)?;
            let first_semantic_ms = elapsed_ms(start);
            let mut literal = Vec::new();
            let mut hybrid = Vec::new();
            let (mut semantic_times, mut literal_times, mut hybrid_times) =
                (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..options.repeats {
                let start = Instant::now();
                semantic = selected.query(query, options.candidate_k, Metric::Cosine)?;
                semantic_times.push(elapsed_ms(start));
                let start = Instant::now();
                literal =
                    literal_search(search.code(), &case.text, &case.filter, options.candidate_k);
                literal_times.push(elapsed_ms(start));
                let start = Instant::now();
                let semantic_for_fusion =
                    selected.query(query, options.candidate_k, Metric::Cosine)?;
                let literal_for_fusion =
                    literal_search(search.code(), &case.text, &case.filter, options.candidate_k);
                hybrid = reciprocal_rank_fusion(
                    &[&semantic_for_fusion, &literal_for_fusion],
                    options.candidate_k,
                );
                hybrid_times.push(elapsed_ms(start));
            }
            results.push(QueryResult {
                id: case.id.clone(),
                category: case.category,
                embedding_ms: *embedding_ms,
                filter_ms,
                first_semantic_ms,
                semantic: strategy(search.code(), &semantic, relevant, semantic_times)?,
                literal: strategy(search.code(), &literal, relevant, literal_times)?,
                experimental_rrf: strategy(search.code(), &hybrid, relevant, hybrid_times)?,
            });
        }
        let overall = summarize(&results.iter().collect::<Vec<_>>());
        let mut groups = BTreeMap::<QueryCategory, Vec<&QueryResult>>::new();
        for result in &results {
            groups.entry(result.category).or_default().push(result);
        }
        let by_category = groups
            .into_iter()
            .map(|(category, rows)| (category, summarize(&rows)))
            .collect();
        variants.push(VariantReport {
            snapshot,
            conversion_ms,
            upload_ms,
            queries: results,
            overall,
            by_category,
        });
    }
    Ok(EvaluationReport {
        format_version: 1,
        dataset: dataset.clone(),
        backend: format!("{device:?}"),
        options,
        rrf_rank_constant: RRF_CONSTANT as usize,
        embedding_total_ms,
        variants,
    })
}
