//! Versioned tool arguments and bounded, snapshot-local results. No request accepts a file path
//! to open: path_prefix is only a literal filter over already loaded metadata.
use cubeql_code::retrieval::{SearchStrategy, SourceRank};
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

pub const MAX_QUERY_BYTES: usize = 8192;
pub const MAX_TOP_K: usize = 100;
pub const MAX_CANDIDATES: usize = 1000;
pub const MAX_READ_BYTES: usize = 65536;
pub const MAX_RESPONSE_BYTES: usize = 512 * 1024;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Semantic,
    Lexical,
    Symbol,
    #[default]
    Hybrid,
}
impl From<Strategy> for SearchStrategy {
    fn from(value: Strategy) -> Self {
        match value {
            Strategy::Semantic => Self::Semantic,
            Strategy::Lexical => Self::Lexical,
            Strategy::Symbol => Self::Symbol,
            Strategy::Hybrid => Self::Hybrid,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    /// Natural language or identifiers. Maximum 8192 UTF-8 bytes; must not be blank.
    pub query: String,
    /// Defaults to hybrid. In text-only mode explicitly select lexical or symbol.
    #[serde(default)]
    pub strategy: Strategy,
    /// Final result count, from 1 through 100. Defaults to 5.
    #[serde(default = "default_top_k")]
    #[schemars(range(min = 1, max = 100))]
    pub top_k: usize,
    /// Per-source candidates before fusion/deduplication: top_k..=1000. Defaults to max(50, top_k).
    #[schemars(range(min = 1, max = 1000))]
    pub candidate_k: Option<usize>,
    /// Case-sensitive language equality filter over saved metadata.
    pub language: Option<String>,
    /// Literal repository-relative path prefix, not a filesystem path to read.
    pub path_prefix: Option<String>,
}
fn default_top_k() -> usize {
    5
}
impl SearchRequest {
    pub(crate) fn validate(&self) -> Result<usize, ToolFailure> {
        let candidates = self.candidate_k.unwrap_or(50.max(self.top_k));
        if self.query.trim().is_empty() || self.query.len() > MAX_QUERY_BYTES {
            return Err(ToolFailure::invalid(
                "query must contain 1..=8192 UTF-8 bytes and not be blank",
            ));
        }
        if !(1..=MAX_TOP_K).contains(&self.top_k)
            || candidates < self.top_k
            || candidates > MAX_CANDIDATES
        {
            return Err(ToolFailure::invalid(
                "top_k must be 1..=100; candidate_k must be top_k..=1000",
            ));
        }
        if self.language.as_ref().is_some_and(|v| v.len() > 256)
            || self.path_prefix.as_ref().is_some_and(|v| v.len() > 4096)
        {
            return Err(ToolFailure::invalid(
                "language/path_prefix exceeds its byte limit",
            ));
        }
        Ok(candidates)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadRequest {
    /// Copy snapshot_id from search_code. A row number alone is not a stable code identity.
    pub snapshot_id: String,
    /// Snapshot-local chunk identifier returned by search_code.
    pub chunk_id: usize,
    /// UTF-8 byte offset inside the saved chunk. Defaults to 0; must be a character boundary.
    #[serde(default)]
    pub offset: usize,
    /// Maximum source bytes returned: 4..=65536. Defaults to 8192. Follow next_offset to continue.
    #[serde(default = "default_read_bytes")]
    #[schemars(range(min = 4, max = 65536))]
    pub max_bytes: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckRequest {
    /// Copy snapshot_id from search_code; checks never accept a caller-provided file path.
    pub snapshot_id: String,
    /// Check the complete workspace file containing this saved chunk.
    pub chunk_id: usize,
}
impl CheckRequest {
    pub(crate) fn validate(&self) -> Result<(), ToolFailure> {
        if self.snapshot_id.len() > 128 {
            return Err(ToolFailure::invalid("snapshot_id is too long"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    /// Complete file bytes matched the saved file hash at check time, not a future edit lock.
    Unchanged,
    Modified,
    Missing,
    /// Never treat a missing workspace, missing manifest, or failed read as unchanged.
    Unverifiable,
}

#[derive(Serialize)]
pub struct CheckResponse {
    pub format_version: u32,
    pub snapshot_id: String,
    pub source: &'static str,
    #[serde(flatten)]
    pub location: Location,
    pub status: Freshness,
    /// Machine-readable explanation, e.g. file_hash_match or legacy_snapshot.
    pub reason: &'static str,
    pub expected_file_hash: Option<String>,
    pub actual_file_hash: Option<String>,
    pub checked_bytes: u64,
}
fn default_read_bytes() -> usize {
    8192
}
impl ReadRequest {
    pub(crate) fn validate(&self) -> Result<(), ToolFailure> {
        if self.snapshot_id.len() > 128 || !(4..=MAX_READ_BYTES).contains(&self.max_bytes) {
            return Err(ToolFailure::invalid(
                "snapshot_id is too long or max_bytes is outside 4..=65536",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ToolFailure {
    pub code: &'static str,
    pub message: String,
}
impl ToolFailure {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_request", message)
    }
}
impl std::fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for ToolFailure {}

#[derive(Serialize)]
pub struct Location {
    pub chunk_id: usize,
    pub path: String,
    pub language: String,
    pub symbol: String,
    pub start_line: u32,
    pub end_line: u32,
    pub content_hash: String,
}

#[derive(Serialize)]
pub struct SearchHit {
    #[serde(flatten)]
    pub location: Location,
    pub score: f32,
    pub provenance: Vec<SourceRank>,
    /// At most 240 UTF-8 bytes from the saved chunk, not a generated summary.
    pub preview: String,
    pub preview_truncated: bool,
    pub suppressed_overlap_count: usize,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub format_version: u32,
    pub snapshot_id: String,
    pub source: &'static str,
    pub strategy: Strategy,
    pub metric: &'static str,
    pub backend: String,
    pub candidate_k: usize,
    pub hits: Vec<SearchHit>,
    pub embedding_ms: f64,
    /// Includes filter preparation, selected branches, provenance, deduplication and result packing.
    pub retrieval_ms: f64,
}

#[derive(Serialize)]
pub struct ReadResponse {
    pub format_version: u32,
    pub snapshot_id: String,
    pub source: &'static str,
    #[serde(flatten)]
    pub location: Location,
    pub offset: usize,
    pub next_offset: Option<usize>,
    pub total_bytes: usize,
    pub text: String,
}

pub(crate) fn prefix_end(text: &str, limit: usize) -> usize {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    end
}
