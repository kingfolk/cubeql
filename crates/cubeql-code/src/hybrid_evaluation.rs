//! Current retrieval evaluation layered over an unchanged M6 baseline. Capturing the baseline's
//! query embeddings avoids re-encoding queries and preserves exact precision-comparison inputs.
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use cubecl::Device;
use cubeql_embeddings::{EmbeddedQuery, EmbeddingConfig, EmbeddingError, EmbeddingProvider};
use serde::Serialize;

use crate::{
    CodeError, CodeIndex, CodeSearchIndex,
    evaluation::{
        self, EvaluationOptions, Latency, QualitySummary, QueryCategory, QueryResult, QuerySet,
        RankingMetrics, ranking_metrics,
    },
    invalid,
    reporting::{SnapshotInfo, elapsed_ms},
    retrieval::{
        RetrievalHit, RetrievalOptions, RetrievedCodeHit, SearchStrategy, SnapshotTextIndex,
        materialize_hits,
    },
};

#[derive(Serialize)]
pub struct RetrievalStrategyResult {
    pub metrics: RankingMetrics,
    pub retrieval: Latency,
    pub materialization_ms: f64,
    pub top_hits: Vec<RetrievedCodeHit>,
}

#[derive(Serialize)]
pub struct HybridQueryResult {
    #[serde(flatten)]
    pub baseline: QueryResult,
    /// Current adapters' filter preparation, separate from the original baseline's timer.
    pub hybrid_filter_ms: f64,
    pub lexical: RetrievalStrategyResult,
    pub symbol: RetrievalStrategyResult,
    pub hybrid: RetrievalStrategyResult,
}

#[derive(Serialize)]
pub struct HybridVariantReport {
    pub snapshot: SnapshotInfo,
    pub baseline_conversion_ms: f64,
    pub baseline_upload_ms: f64,
    pub conversion_ms: f64,
    pub upload_ms: f64,
    pub text_index_ms: f64,
    pub queries: Vec<HybridQueryResult>,
    pub overall: BTreeMap<String, QualitySummary>,
    pub by_category: BTreeMap<QueryCategory, BTreeMap<String, QualitySummary>>,
}

#[derive(Serialize)]
pub struct HybridEvaluationReport {
    pub format_version: u32,
    pub dataset: QuerySet,
    pub backend: String,
    pub options: EvaluationOptions,
    pub rrf_rank_constant: usize,
    pub embedding_total_ms: f64,
    pub variants: Vec<HybridVariantReport>,
}

struct CaptureQueries<'a> {
    provider: &'a mut dyn EmbeddingProvider,
    vectors: Vec<Vec<f32>>,
}
impl EmbeddingProvider for CaptureQueries<'_> {
    fn config(&self) -> &EmbeddingConfig {
        self.provider.config()
    }
    fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(EmbeddingError::Invalid(
            "evaluation must not encode documents".into(),
        ))
    }
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let vector = self.provider.embed_query(text)?;
        self.vectors.push(vector.clone());
        Ok(vector)
    }
}

pub fn evaluate(
    code: CodeIndex,
    device: &Device,
    provider: &mut dyn EmbeddingProvider,
    dataset: &QuerySet,
    options: EvaluationOptions,
) -> Result<HybridEvaluationReport, CodeError> {
    let config = code
        .embedding_config()
        .cloned()
        .ok_or_else(|| invalid("evaluation requires a provider-aware index"))?;
    let mut capture = CaptureQueries {
        provider,
        vectors: Vec::new(),
    };
    // The original evaluator still owns validation, inference, baseline metrics and precision
    // guards. It runs first and frees its resident indexes before the current variants run.
    let baseline = evaluation::evaluate(code.clone(), device, &mut capture, dataset, options)?;
    let queries = capture
        .vectors
        .into_iter()
        .map(|v| EmbeddedQuery::new(config.clone(), v))
        .collect::<Result<Vec<_>, _>>()?;
    if queries.len() != dataset.queries.len() {
        return Err(invalid("baseline query capture count mismatch"));
    }
    let relevant = dataset.resolve(&code)?;
    let mut variants = Vec::new();
    for old in baseline.variants {
        let precision = old.snapshot.precision;
        let start = Instant::now();
        let variant = code.clone().with_precision(precision)?;
        let conversion_ms = elapsed_ms(start);
        let start = Instant::now();
        let resident = CodeSearchIndex::new(device, variant)?;
        let upload_ms = elapsed_ms(start);
        let start = Instant::now();
        let text_index = SnapshotTextIndex::new(resident.code());
        text_index.warm_up(SearchStrategy::Hybrid);
        let text_index_ms = elapsed_ms(start);
        let mut results = Vec::new();
        for (((case, query), relevant), old_query) in dataset
            .queries
            .iter()
            .zip(&queries)
            .zip(&relevant)
            .zip(old.queries)
        {
            let start = Instant::now();
            let prepared = text_index.prepare(&case.filter, Some(&resident))?;
            let hybrid_filter_ms = elapsed_ms(start);
            let mut strategies = Vec::new();
            for strategy in [
                SearchStrategy::Lexical,
                SearchStrategy::Symbol,
                SearchStrategy::Hybrid,
            ] {
                let opts = RetrievalOptions {
                    strategy,
                    top_k: options.candidate_k,
                    candidate_k: options.candidate_k,
                    deduplicate: true,
                };
                // Warm each current strategy explicitly; report the original baseline's first
                // semantic invocation separately. This is not a fresh-process cold start.
                prepared.query(&case.text, Some(query), opts)?;
                let mut hits = Vec::new();
                let mut samples = Vec::new();
                for _ in 0..options.repeats {
                    let start = Instant::now();
                    hits = prepared.query(&case.text, Some(query), opts)?.hits;
                    samples.push(elapsed_ms(start));
                }
                strategies.push(strategy_result(resident.code(), &hits, relevant, samples)?);
            }
            let mut strategies = strategies.into_iter();
            results.push(HybridQueryResult {
                baseline: old_query,
                hybrid_filter_ms,
                lexical: strategies.next().unwrap(),
                symbol: strategies.next().unwrap(),
                hybrid: strategies.next().unwrap(),
            });
        }
        let mut overall = old.overall;
        overall.extend(summarize(&results.iter().collect::<Vec<_>>()));
        let mut groups = BTreeMap::<QueryCategory, Vec<&HybridQueryResult>>::new();
        for result in &results {
            groups
                .entry(result.baseline.category)
                .or_default()
                .push(result);
        }
        let mut by_category = old.by_category;
        for (category, rows) in groups {
            by_category
                .entry(category)
                .or_default()
                .extend(summarize(&rows));
        }
        variants.push(HybridVariantReport {
            snapshot: old.snapshot,
            baseline_conversion_ms: old.conversion_ms,
            baseline_upload_ms: old.upload_ms,
            conversion_ms,
            upload_ms,
            text_index_ms,
            queries: results,
            overall,
            by_category,
        });
    }
    Ok(HybridEvaluationReport {
        format_version: 2,
        dataset: baseline.dataset,
        backend: baseline.backend,
        options,
        rrf_rank_constant: baseline.rrf_rank_constant,
        embedding_total_ms: baseline.embedding_total_ms,
        variants,
    })
}

fn strategy_result(
    code: &CodeIndex,
    hits: &[RetrievalHit],
    relevant: &BTreeSet<usize>,
    mut samples: Vec<f64>,
) -> Result<RetrievalStrategyResult, CodeError> {
    let metrics = ranking_metrics(&hits.iter().map(|h| h.row).collect::<Vec<_>>(), relevant)?;
    samples.sort_by(f64::total_cmp);
    let n = samples.len(); // Baseline validation already rejected repeats == 0.
    let median_ms = if n.is_multiple_of(2) {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    } else {
        samples[n / 2]
    };
    let retrieval = Latency {
        median_ms,
        min_ms: samples[0],
        max_ms: samples[n - 1],
        samples: n,
    };
    let start = Instant::now();
    let top_hits = materialize_hits(code, &hits[..hits.len().min(10)], false)?;
    Ok(RetrievalStrategyResult {
        metrics,
        retrieval,
        materialization_ms: elapsed_ms(start),
        top_hits,
    })
}

fn summarize(results: &[&HybridQueryResult]) -> BTreeMap<String, QualitySummary> {
    ["lexical", "symbol", "hybrid"]
        .into_iter()
        .map(|name| {
            let metrics = results
                .iter()
                .map(|r| match name {
                    "lexical" => r.lexical.metrics,
                    "symbol" => r.symbol.metrics,
                    _ => r.hybrid.metrics,
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
