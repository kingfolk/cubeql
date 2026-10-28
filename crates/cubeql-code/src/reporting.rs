//! Application-level, serializable results. Core CubeQL remains Arrow/vector-only.
use std::{collections::BTreeSet, io::Write, time::Instant};

use cubeql::{Filter, SearchHit};
use cubeql_embeddings::EmbeddingConfig;
use serde::{Deserialize, Serialize};

use crate::{CodeChunk, CodeError, CodeIndex, VectorPrecision, invalid};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeFilter {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub path_prefix: Option<String>,
}

impl CodeFilter {
    pub fn predicates(&self) -> Vec<Filter<'_>> {
        let mut filters = Vec::new();
        if let Some(value) = &self.language {
            filters.push(Filter::StringEq {
                column: "language",
                value,
            });
        }
        if let Some(prefix) = &self.path_prefix {
            filters.push(Filter::PathPrefix {
                column: "path",
                prefix,
            });
        }
        filters
    }
    pub fn matches(&self, chunk: &CodeChunk) -> bool {
        self.language.as_ref().is_none_or(|v| *v == chunk.language)
            && self
                .path_prefix
                .as_ref()
                .is_none_or(|p| chunk.path.starts_with(p))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CodeHit {
    pub row: usize,
    pub path: String,
    pub language: String,
    pub symbol: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

pub fn materialize(
    code: &CodeIndex,
    hits: &[SearchHit],
    include_source: bool,
) -> Result<Vec<CodeHit>, CodeError> {
    hits.iter()
        .map(|hit| {
            let chunk = code
                .chunks()
                .get(hit.row)
                .ok_or_else(|| invalid("search hit is outside this snapshot"))?;
            Ok(CodeHit {
                row: hit.row,
                path: chunk.path.clone(),
                language: chunk.language.clone(),
                symbol: chunk.symbol.clone(),
                kind: chunk.kind.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                score: hit.score,
                source: include_source.then(|| chunk.source.clone()),
            })
        })
        .collect()
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SearchTimings {
    pub read_ms: f64,
    pub model_load_ms: f64,
    pub embedding_ms: f64,
    pub upload_ms: f64,
    pub filter_ms: f64,
    /// Includes device scoring, top-k, final readback and any first-use compilation.
    pub scoring_topk_readback_ms: f64,
    pub materialization_ms: f64,
    pub text_index_ms: f64,
    pub lexical_ms: f64,
    pub symbol_ms: f64,
    pub fusion_ms: f64,
    pub dedup_ms: f64,
    /// Complete retrieval call, including all selected branches, evidence and deduplication.
    pub retrieval_ms: f64,
}

/// CLI search format v2: scores are explicitly strategy-dependent and every hit is explained.
#[derive(Serialize)]
pub struct RetrievalResponse {
    pub format_version: u32,
    pub query: String,
    pub strategy: crate::retrieval::SearchStrategy,
    pub metric: &'static str,
    pub backend: String,
    pub precision: VectorPrecision,
    pub hits: Vec<crate::retrieval::RetrievedCodeHit>,
    pub timings: SearchTimings,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub format_version: u32,
    pub query: String,
    pub metric: &'static str,
    pub backend: String,
    pub precision: VectorPrecision,
    pub hits: Vec<CodeHit>,
    pub timings: SearchTimings,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotInfo {
    pub chunks: usize,
    pub files: usize,
    pub root: Option<String>,
    pub dimensions: usize,
    pub precision: VectorPrecision,
    pub fallback_lines: usize,
    pub max_file_bytes: u64,
    pub embedding_id: String,
    pub embedding: Option<EmbeddingConfig>,
    pub corpus_fingerprint: String,
    pub vector_storage_bytes: usize,
    /// Arrow's allocation accounting, not process RSS or a peak-memory measurement.
    pub arrow_array_memory_bytes: usize,
    /// UTF-8 source bytes in chunks; container headers/children are distinct chunks.
    pub source_text_bytes: usize,
    /// Size when re-encoded with the current CubeQL IPC writer, without writing a disk file.
    pub ipc_bytes: u64,
}

impl SnapshotInfo {
    pub fn new(index: &CodeIndex) -> Result<Self, CodeError> {
        let options = index.build_options()?;
        let logical = index
            .chunks()
            .iter()
            .map(|c| (&c.path, &c.symbol, &c.content_hash))
            .collect::<Vec<_>>();
        let fingerprint = blake3::hash(
            &serde_json::to_vec(&logical).map_err(cubeql_embeddings::EmbeddingError::from)?,
        )
        .to_hex()
        .to_string();
        let mut count = CountWriter(0);
        index.write_ipc(&mut count)?;
        Ok(Self {
            chunks: index.chunks().len(),
            files: index
                .chunks()
                .iter()
                .map(|c| &c.path)
                .collect::<BTreeSet<_>>()
                .len(),
            root: index.root().map(str::to_owned),
            dimensions: index.dimensions(),
            precision: index.precision(),
            fallback_lines: options.fallback_lines,
            max_file_bytes: options.max_file_bytes,
            embedding_id: index.embedding_id().into(),
            embedding: index.embedding_config().cloned(),
            corpus_fingerprint: fingerprint,
            vector_storage_bytes: index.chunks().len()
                * index.dimensions()
                * if index.precision() == VectorPrecision::Fp16 {
                    2
                } else {
                    4
                },
            arrow_array_memory_bytes: index.batch().get_array_memory_size(),
            source_text_bytes: index.chunks().iter().map(|c| c.source.len()).sum(),
            ipc_bytes: count.0,
        })
    }
}

struct CountWriter(u64);
impl Write for CountWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
