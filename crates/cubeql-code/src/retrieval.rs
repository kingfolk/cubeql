//! Snapshot-local lexical/syntax adapters and explicit hybrid retrieval. No live filesystem
//! search is performed, so source ranks and device row IDs always refer to the same snapshot.
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::OnceLock,
    time::Instant,
};

use cubeql::{Metric, SearchHit};
use cubeql_embeddings::EmbeddedQuery;
use serde::{Deserialize, Serialize};

use crate::{
    CodeChunk, CodeError, CodeIndex, CodeSearchIndex, PreparedCodeSearch,
    evaluation::reciprocal_rank_fusion,
    invalid,
    reporting::{CodeFilter, CodeHit, elapsed_ms, materialize},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    #[default]
    Semantic,
    Lexical,
    Symbol,
    Hybrid,
}
impl SearchStrategy {
    pub fn needs_embedding(self) -> bool {
        matches!(self, Self::Semantic | Self::Hybrid)
    }
    pub fn score_kind(self) -> &'static str {
        match self {
            Self::Semantic => "cosine",
            Self::Lexical => "lexical",
            Self::Symbol => "symbol",
            Self::Hybrid => "rrf",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RetrievalOptions {
    pub strategy: SearchStrategy,
    pub top_k: usize,
    /// Per-source candidate budget, before fusion and range deduplication.
    pub candidate_k: usize,
    pub deduplicate: bool,
}
impl Default for RetrievalOptions {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::Semantic,
            top_k: 5,
            candidate_k: 50,
            deduplicate: true,
        }
    }
}
impl RetrievalOptions {
    pub fn validate(self) -> Result<(), CodeError> {
        if self.candidate_k < self.top_k {
            return Err(invalid("candidate_k must be >= top_k"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalSource {
    Semantic,
    Lexical,
    Symbol,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRank {
    pub source: RetrievalSource,
    /// One-based rank within this source's filtered candidate list.
    pub rank: usize,
    pub raw_score: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SuppressedHit {
    pub row: usize,
    pub path: String,
    pub symbol: String,
    pub start_line: u32,
    pub end_line: u32,
    pub provenance: Vec<SourceRank>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievalHit {
    pub row: usize,
    /// Cosine/lexical/symbol score for a standalone strategy, RRF score for hybrid.
    pub score: f32,
    pub provenance: Vec<SourceRank>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suppressed_overlaps: Vec<SuppressedHit>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievedCodeHit {
    #[serde(flatten)]
    pub code: CodeHit,
    pub provenance: Vec<SourceRank>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suppressed_overlaps: Vec<SuppressedHit>,
}

pub fn materialize_hits(
    code: &CodeIndex,
    hits: &[RetrievalHit],
    include_source: bool,
) -> Result<Vec<RetrievedCodeHit>, CodeError> {
    let raw = hits
        .iter()
        .map(|h| SearchHit {
            row: h.row,
            score: h.score,
        })
        .collect::<Vec<_>>();
    Ok(materialize(code, &raw, include_source)?
        .into_iter()
        .zip(hits)
        .map(|(code, hit)| RetrievedCodeHit {
            code,
            provenance: hit.provenance.clone(),
            suppressed_overlaps: hit.suppressed_overlaps.clone(),
        })
        .collect())
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RetrievalTimings {
    pub semantic_ms: f64,
    pub lexical_ms: f64,
    pub symbol_ms: f64,
    pub fusion_ms: f64,
    pub dedup_ms: f64,
}

pub struct RetrievalResult {
    pub hits: Vec<RetrievalHit>,
    pub timings: RetrievalTimings,
}

/// A host inverted index and syntax-symbol lookup over one immutable Arrow code snapshot.
/// Rebuild once after loading a new snapshot; reuse it for repeated queries. No model/device
/// is required by the lexical or symbol adapters. This is not BM25 or a live ripgrep wrapper.
pub struct SnapshotTextIndex<'a> {
    code: &'a CodeIndex,
    terms: OnceLock<HashMap<String, Vec<usize>>>,
    symbols: OnceLock<HashMap<&'a str, Vec<usize>>>,
}
impl<'a> SnapshotTextIndex<'a> {
    pub fn new(code: &'a CodeIndex) -> Self {
        Self {
            code,
            terms: OnceLock::new(),
            symbols: OnceLock::new(),
        }
    }

    /// Build only the selected host adapters, outside warm query timers if desired.
    /// Semantic-only queries never construct the text/symbol tables.
    pub fn warm_up(&self, strategy: SearchStrategy) {
        if matches!(strategy, SearchStrategy::Lexical | SearchStrategy::Hybrid) {
            self.terms();
        }
        if matches!(strategy, SearchStrategy::Symbol | SearchStrategy::Hybrid) {
            self.symbols();
        }
    }

    fn terms(&self) -> &HashMap<String, Vec<usize>> {
        self.terms.get_or_init(|| {
            let mut terms = HashMap::<String, Vec<usize>>::new();
            for (row, chunk) in self.code.chunks().iter().enumerate() {
                let text = format!("{} {} {}", chunk.path, chunk.symbol, chunk.source);
                for term in tokenize(&text) {
                    terms.entry(term).or_default().push(row);
                }
            }
            terms
        })
    }

    fn symbols(&self) -> &HashMap<&'a str, Vec<usize>> {
        self.symbols.get_or_init(|| {
            let mut symbols = HashMap::<&str, Vec<usize>>::new();
            for (row, chunk) in self.code.chunks().iter().enumerate() {
                if chunk.kind != "lines" {
                    symbols.entry(&chunk.symbol).or_default().push(row);
                    let leaf = chunk.symbol.rsplit("::").next().unwrap();
                    if leaf != chunk.symbol {
                        symbols.entry(leaf).or_default().push(row);
                    }
                }
            }
            symbols
        })
    }

    /// Bind both adapters to identical filters and the very same CodeIndex object. Merely
    /// matching dimensions or row counts would not prevent cross-snapshot metadata leaks.
    pub fn prepare<'s>(
        &'s self,
        filter: &CodeFilter,
        semantic: Option<&'s CodeSearchIndex>,
    ) -> Result<PreparedRetrieval<'s, 'a>, CodeError> {
        if semantic.is_some_and(|s| !std::ptr::eq(self.code, s.code())) {
            return Err(invalid(
                "text and semantic indexes must borrow the same code snapshot",
            ));
        }
        let predicates = filter.predicates();
        let semantic = semantic
            .map(|s| s.prepare_filter(&predicates))
            .transpose()?;
        Ok(PreparedRetrieval {
            index: self,
            allowed: self
                .code
                .chunks()
                .iter()
                .map(|c| filter.matches(c))
                .collect(),
            semantic,
        })
    }
}

pub struct PreparedRetrieval<'s, 'a> {
    index: &'s SnapshotTextIndex<'a>,
    allowed: Vec<bool>,
    pub(crate) semantic: Option<PreparedCodeSearch<'s>>,
}
impl PreparedRetrieval<'_, '_> {
    /// Query embeddings are supplied, not generated here. Pure lexical/symbol requests need
    /// neither a provider nor a resident device index. All returned locations are snapshot-local.
    pub fn query(
        &self,
        text: &str,
        query: Option<&EmbeddedQuery>,
        options: RetrievalOptions,
    ) -> Result<RetrievalResult, CodeError> {
        options.validate()?;
        if text.trim().is_empty() {
            return Err(invalid("search text must not be empty"));
        }
        let mut timings = RetrievalTimings::default();
        if options.top_k == 0 {
            return Ok(RetrievalResult {
                hits: Vec::new(),
                timings,
            });
        }
        let mut lists = Vec::<(RetrievalSource, Vec<SearchHit>)>::new();
        if options.strategy.needs_embedding() {
            let selected = self
                .semantic
                .as_ref()
                .ok_or_else(|| invalid("semantic/hybrid retrieval needs a resident index"))?;
            let query = query.ok_or_else(|| {
                invalid("semantic/hybrid retrieval needs a tagged query embedding")
            })?;
            let start = Instant::now();
            let hits = selected.query(query, options.candidate_k, Metric::Cosine)?;
            timings.semantic_ms = elapsed_ms(start);
            lists.push((RetrievalSource::Semantic, hits));
        }
        if matches!(
            options.strategy,
            SearchStrategy::Lexical | SearchStrategy::Hybrid
        ) {
            let start = Instant::now();
            lists.push((
                RetrievalSource::Lexical,
                self.lexical(text, options.candidate_k),
            ));
            timings.lexical_ms = elapsed_ms(start);
        }
        if matches!(
            options.strategy,
            SearchStrategy::Symbol | SearchStrategy::Hybrid
        ) {
            let start = Instant::now();
            lists.push((
                RetrievalSource::Symbol,
                self.symbol(text, options.candidate_k),
            ));
            timings.symbol_ms = elapsed_ms(start);
        }
        let mut evidence = BTreeMap::<usize, Vec<SourceRank>>::new();
        for (source, hits) in &lists {
            for (rank, hit) in hits.iter().enumerate() {
                evidence.entry(hit.row).or_default().push(SourceRank {
                    source: *source,
                    rank: rank + 1,
                    raw_score: hit.score,
                });
            }
        }
        let start = Instant::now();
        let raw = if options.strategy == SearchStrategy::Hybrid {
            reciprocal_rank_fusion(
                &lists.iter().map(|(_, h)| h.as_slice()).collect::<Vec<_>>(),
                usize::MAX,
            )
        } else {
            lists.pop().unwrap().1
        };
        let hits = raw
            .into_iter()
            .map(|h| RetrievalHit {
                row: h.row,
                score: h.score,
                provenance: evidence.remove(&h.row).unwrap(),
                suppressed_overlaps: Vec::new(),
            })
            .collect();
        timings.fusion_ms = elapsed_ms(start);
        let start = Instant::now();
        let mut hits = if options.deduplicate {
            deduplicate(self.index.code.chunks(), hits)
        } else {
            hits
        };
        // Deduplicate before truncating, so overlapping candidates do not waste result slots.
        hits.truncate(options.top_k);
        timings.dedup_ms = elapsed_ms(start);
        Ok(RetrievalResult { hits, timings })
    }

    fn lexical(&self, text: &str, k: usize) -> Vec<SearchHit> {
        let terms = tokenize(text);
        let total = self.index.code.chunks().len() as f64;
        let mut scores = BTreeMap::<usize, f64>::new();
        let mut denominator = 0.0;
        for term in terms {
            let rows = self.index.terms().get(&term);
            let df = rows.map_or(0, Vec::len) as f64;
            let idf = ((total + 1.0) / (df + 1.0)).ln() + 1.0;
            denominator += idf;
            if let Some(rows) = rows {
                for &row in rows.iter().filter(|&&row| self.allowed[row]) {
                    *scores.entry(row).or_default() += idf;
                }
            }
        }
        // Case-sensitive exact phrase matches remain available, including punctuation-only
        // queries. Token coverage is case-insensitive and splits snake_case/CamelCase names.
        for (row, chunk) in self
            .index
            .code
            .chunks()
            .iter()
            .enumerate()
            .filter(|(r, _)| self.allowed[*r])
        {
            if chunk.source.contains(text) {
                scores.entry(row).or_default();
            }
        }
        let hits = scores
            .into_iter()
            .map(|(row, coverage)| {
                let exact = self.index.code.chunks()[row].source.contains(text);
                SearchHit {
                    row,
                    score: (if denominator == 0.0 {
                        0.0
                    } else {
                        coverage / denominator
                    }) as f32
                        + if exact { 1.0 } else { 0.0 },
                }
            })
            .collect();
        sorted(hits, k)
    }

    fn symbol(&self, text: &str, k: usize) -> Vec<SearchHit> {
        let mut scores = BTreeMap::<usize, f32>::new();
        let text = text.trim();
        // Qualified atoms remain qualified: Other::load must not match Config::load.
        let atoms = std::iter::once(text).chain(
            text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
                .filter(|s| !s.is_empty()),
        );
        for atom in atoms {
            if let Some(rows) = self.index.symbols().get(atom) {
                for &row in rows.iter().filter(|&&row| self.allowed[row]) {
                    let chunk = &self.index.code.chunks()[row];
                    let score = if text == chunk.symbol {
                        3.0
                    } else if text == atom {
                        2.0
                    } else {
                        1.0
                    };
                    scores
                        .entry(row)
                        .and_modify(|s| *s = s.max(score))
                        .or_insert(score);
                }
            }
        }
        sorted(
            scores
                .into_iter()
                .map(|(row, score)| SearchHit { row, score })
                .collect(),
            k,
        )
    }
}

fn sorted(mut hits: Vec<SearchHit>, k: usize) -> Vec<SearchHit> {
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.row.cmp(&b.row)));
    hits.truncate(k);
    hits
}

/// Greedy, rank-preserving suppression for the same path with line-range IoU >= 0.8.
/// A small header inside a much larger function is not automatically a duplicate.
fn deduplicate(chunks: &[CodeChunk], hits: Vec<RetrievalHit>) -> Vec<RetrievalHit> {
    let mut kept = Vec::<RetrievalHit>::new();
    for hit in hits {
        let chunk = &chunks[hit.row];
        let overlap = kept.iter_mut().find(|other| {
            let other = &chunks[other.row];
            let start = chunk.start_line.max(other.start_line) as u64;
            let end = chunk.end_line.min(other.end_line) as u64;
            let intersection = if end >= start { end - start + 1 } else { 0 };
            let union = (chunk.end_line - chunk.start_line) as u64
                + 1
                + (other.end_line - other.start_line) as u64
                + 1
                - intersection;
            chunk.path == other.path && intersection * 5 >= union * 4
        });
        if let Some(other) = overlap {
            other.suppressed_overlaps.push(SuppressedHit {
                row: hit.row,
                path: chunk.path.clone(),
                symbol: chunk.symbol.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                provenance: hit.provenance,
            });
        } else {
            kept.push(hit);
        }
    }
    kept
}

/// Deterministic identifier-aware terms, no learned vocabulary, stemming or query syntax.
fn tokenize(text: &str) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    let mut insert = |s: &str| {
        let term = s.to_lowercase();
        if !term.is_empty() && !STOP_WORDS.contains(&term.as_str()) {
            terms.insert(term);
        }
    };
    for word in text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
    {
        insert(word);
        for part in word.split('_').filter(|s| !s.is_empty()) {
            let chars = part.char_indices().collect::<Vec<_>>();
            let mut start = 0;
            for i in 1..chars.len() {
                let prev = chars[i - 1].1;
                let curr = chars[i].1;
                let next_lower = chars.get(i + 1).is_some_and(|(_, c)| c.is_lowercase());
                if curr.is_uppercase()
                    && (prev.is_lowercase()
                        || prev.is_numeric()
                        || (prev.is_uppercase() && next_lower))
                {
                    insert(&part[start..chars[i].0]);
                    start = chars[i].0;
                }
            }
            insert(&part[start..]);
        }
    }
    terms
}

const STOP_WORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "find",
    "fn",
    "for",
    "from",
    "function",
    "functions",
    "how",
    "in",
    "is",
    "it",
    "let",
    "me",
    "mut",
    "of",
    "on",
    "or",
    "pub",
    "return",
    "self",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "where",
    "which",
    "with",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_split_identifiers_acronyms_and_unicode_without_duplicate_postings() {
        let terms = tokenize("HTTPParser load_config CaféValue load_config THE fn");
        for term in [
            "httpparser",
            "http",
            "parser",
            "load_config",
            "load",
            "config",
            "café",
            "value",
        ] {
            assert!(terms.contains(term), "missing {term}");
        }
        assert!(!terms.contains("the"));
        assert!(!terms.contains("fn"));
    }

    #[test]
    fn range_iou_dedup_preserves_small_headers_and_other_files() {
        let chunk = |path: &str, start_line, end_line| CodeChunk {
            path: path.into(),
            language: "rust".into(),
            symbol: "example".into(),
            kind: "function".into(),
            start_line,
            end_line,
            content_hash: String::new(),
            source: String::new(),
        };
        let chunks = [
            chunk("same.rs", 1, 10),
            chunk("same.rs", 2, 11),
            chunk("same.rs", 1, 2),
            chunk("other.rs", 1, 10),
        ];
        let hits = (0..4)
            .map(|row| RetrievalHit {
                row,
                score: 4.0 - row as f32,
                provenance: Vec::new(),
                suppressed_overlaps: Vec::new(),
            })
            .collect();
        let kept = deduplicate(&chunks, hits);
        assert_eq!(kept.iter().map(|h| h.row).collect::<Vec<_>>(), [0, 2, 3]);
        assert_eq!(kept[0].suppressed_overlaps[0].row, 1);
    }
}
