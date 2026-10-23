//! Optional text preparation/model inference. This crate has no Arrow or CubeCL dependency;
//! the `cubeql` retrieval engine does not depend on it either.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "minilm")]
mod minilm;
#[cfg(feature = "minilm")]
pub use minilm::{MINILM_MODEL, MINILM_REVISION, MiniLm, download_minilm};

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("invalid embedding configuration or output: {0}")]
    Invalid(String),
    #[error("embedding configuration does not match this index")]
    Incompatible,
    #[error("precomputed {kind} embedding not found for input text")]
    Missing { kind: &'static str },
    #[error("embedding inference/download failed: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    None,
    L2,
}

/// Reproducibility and vector-space contract, compared in full, not just by dimensions.
/// Custom providers must change this contract whenever outputs/preprocessing change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub model: String,
    pub revision: String,
    pub dimensions: usize,
    pub normalization: Normalization,
    pub implementation: String,
    pub pooling: String,
    pub max_tokens: Option<usize>,
    pub document_prefix: String,
    pub query_prefix: String,
    /// Sorted filename -> SHA-256 entries for local reference model artifacts.
    pub artifacts: BTreeMap<String, String>,
}

impl EmbeddingConfig {
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.model.is_empty()
            || self.revision.is_empty()
            || self.implementation.is_empty()
            || self.pooling.is_empty()
            || self.dimensions == 0
            || self.dimensions > i32::MAX as usize
            || self.max_tokens == Some(0)
        {
            return Err(EmbeddingError::Invalid(
                "missing identity, dimensions or preprocessing".into(),
            ));
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, EmbeddingError> {
        self.validate()?;
        Ok(format!(
            "cubeql-embedding-v1:{}",
            blake3::hash(&serde_json::to_vec(self)?).to_hex()
        ))
    }

    pub fn ensure_compatible(&self, other: &Self) -> Result<(), EmbeddingError> {
        self.validate()?;
        other.validate()?;
        if self != other {
            return Err(EmbeddingError::Incompatible);
        }
        Ok(())
    }

    pub fn validate_vectors(
        &self,
        vectors: &[Vec<f32>],
        count: usize,
    ) -> Result<(), EmbeddingError> {
        self.validate()?;
        if vectors.len() != count {
            return Err(EmbeddingError::Invalid(format!(
                "expected {count} vectors, got {}",
                vectors.len()
            )));
        }
        for vector in vectors {
            self.validate_vector(vector)?;
        }
        Ok(())
    }

    pub fn validate_vector(&self, vector: &[f32]) -> Result<(), EmbeddingError> {
        self.validate()?;
        if vector.len() != self.dimensions || vector.iter().any(|v| !v.is_finite()) {
            return Err(EmbeddingError::Invalid(format!(
                "expected {} finite values, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        if self.normalization == Normalization::L2 {
            let norm = vector
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            // Zero vectors retain the engine's defined cosine-zero behavior.
            if norm != 0.0 && (norm - 1.0).abs() > 1e-3 {
                return Err(EmbeddingError::Invalid(
                    "declared L2 embedding is not unit length".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Document and query encoders may use different prefixes, but must share this contract.
/// Implementations return vectors in input order and must not silently drop failed inputs.
pub trait EmbeddingProvider {
    fn config(&self) -> &EmbeddingConfig;
    fn embed_documents(&mut self, documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError>;
}

/// A query vector tagged with its full vector-space contract. It can also wrap precomputed data.
pub struct EmbeddedQuery {
    config: EmbeddingConfig,
    vector: Vec<f32>,
}

impl EmbeddedQuery {
    pub fn new(config: EmbeddingConfig, vector: Vec<f32>) -> Result<Self, EmbeddingError> {
        config.validate()?;
        config.validate_vector(&vector)?;
        Ok(Self { config, vector })
    }
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }
}

pub fn embed_query(
    provider: &mut dyn EmbeddingProvider,
    text: &str,
) -> Result<EmbeddedQuery, EmbeddingError> {
    let config = provider.config().clone();
    config.validate()?;
    let vector = provider.embed_query(text)?;
    config.ensure_compatible(provider.config())?;
    EmbeddedQuery::new(config, vector)
}

/// Offline deterministic fixtures or vectors produced by another embedding system.
pub struct PrecomputedProvider {
    config: EmbeddingConfig,
    documents: HashMap<String, Vec<f32>>,
    queries: HashMap<String, Vec<f32>>,
}

impl PrecomputedProvider {
    pub fn new(
        config: EmbeddingConfig,
        documents: HashMap<String, Vec<f32>>,
        queries: HashMap<String, Vec<f32>>,
    ) -> Result<Self, EmbeddingError> {
        config.validate()?;
        for vector in documents.values().chain(queries.values()) {
            config.validate_vector(vector)?;
        }
        Ok(Self {
            config,
            documents,
            queries,
        })
    }
}

impl EmbeddingProvider for PrecomputedProvider {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    fn embed_documents(&mut self, documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        documents
            .iter()
            .map(|text| {
                self.documents
                    .get(*text)
                    .cloned()
                    .ok_or(EmbeddingError::Missing { kind: "document" })
            })
            .collect()
    }
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.queries
            .get(query)
            .cloned()
            .ok_or(EmbeddingError::Missing { kind: "query" })
    }
}

/// Lexical fixture only; not a trained model. Kept for cheap, offline pipeline tests.
pub struct HashProvider {
    config: EmbeddingConfig,
}

impl HashProvider {
    pub fn new(dimensions: usize) -> Result<Self, EmbeddingError> {
        let config = EmbeddingConfig {
            model: "cubeql-hash-tokens".into(),
            revision: "1".into(),
            dimensions,
            normalization: Normalization::L2,
            implementation: "blake3-signed-token-counts-v1".into(),
            pooling: "token-counts".into(),
            max_tokens: None,
            document_prefix: String::new(),
            query_prefix: String::new(),
            artifacts: BTreeMap::new(),
        };
        config.validate()?;
        Ok(Self { config })
    }
}

impl EmbeddingProvider for HashProvider {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    fn embed_documents(&mut self, documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        documents
            .iter()
            .map(|text| hash_embedding(text, self.config.dimensions))
            .collect()
    }
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        hash_embedding(query, self.config.dimensions)
    }
}

pub fn hash_embedding(text: &str, dimensions: usize) -> Result<Vec<f32>, EmbeddingError> {
    if dimensions == 0 || dimensions > i32::MAX as usize {
        return Err(EmbeddingError::Invalid(
            "dimensions must be in 1..=i32::MAX".into(),
        ));
    }
    let mut vector = vec![0.0_f32; dimensions];
    for token in text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
    {
        let hash = blake3::hash(token.to_lowercase().as_bytes());
        let bytes = hash.as_bytes();
        let bucket =
            (u64::from_le_bytes(bytes[..8].try_into().unwrap()) % dimensions as u64) as usize;
        vector[bucket] += if bytes[8] & 1 == 0 { 1.0 } else { -1.0 };
    }
    let norm = vector
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value = (*value as f64 / norm) as f32;
        }
    }
    Ok(vector)
}
