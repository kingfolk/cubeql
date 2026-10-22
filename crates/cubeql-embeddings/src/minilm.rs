use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::Path,
    time::Duration,
};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{EmbeddingConfig, EmbeddingError, EmbeddingProvider, Normalization};

pub const MINILM_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const MINILM_REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
const FILES: [(&str, &str, u64); 3] = [
    (
        "config.json",
        "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        16384,
    ),
    (
        "tokenizer.json",
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        2 * 1024 * 1024,
    ),
    (
        "model.safetensors",
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        100 * 1024 * 1024,
    ),
];

fn backend(error: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::Backend(error.to_string())
}

fn verified_bytes(path: &Path, digest: &str, limit: u64) -> Result<Vec<u8>, EmbeddingError> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    verify(&bytes, digest, limit)?;
    Ok(bytes)
}

fn verify(bytes: &[u8], digest: &str, limit: u64) -> Result<(), EmbeddingError> {
    if bytes.len() as u64 > limit || format!("{:x}", Sha256::digest(bytes)) != digest {
        return Err(EmbeddingError::Invalid(
            "model artifact SHA-256/size mismatch; existing files are never overwritten".into(),
        ));
    }
    Ok(())
}

/// Explicit one-time download of three pinned, SHA-256-verified public artifacts (~91 MB).
/// Existing valid files are reused; conflicting files are not replaced. Each file is published
/// only after verification, using a same-directory temporary file and no-clobber persistence.
pub fn download_minilm(directory: impl AsRef<Path>) -> Result<(), EmbeddingError> {
    let directory = directory.as_ref();
    fs::create_dir_all(directory)?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .into();
    for (name, hash, limit) in FILES {
        let target = directory.join(name);
        if target.exists() {
            verified_bytes(&target, hash, limit)?;
            continue;
        }
        let url = format!("https://huggingface.co/{MINILM_MODEL}/resolve/{MINILM_REVISION}/{name}");
        let mut response = agent.get(&url).call().map_err(backend)?;
        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .take(limit + 1)
            .read_to_end(&mut bytes)?;
        verify(&bytes, hash, limit)?;
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        temporary.write_all(&bytes)?;
        temporary.flush()?;
        temporary.persist_noclobber(&target).map_err(backend)?;
    }
    Ok(())
}

/// CPU-only, FP32 local reference encoder: uncased MiniLM, right truncation to 256 tokens,
/// attention-mask-aware mean pooling and L2 normalization. No network access during loading
/// or encoding. This is a general English sentence model, not a code-specialized model.
pub struct MiniLm {
    config: EmbeddingConfig,
    tokenizer: Tokenizer,
    model: BertModel,
}

impl MiniLm {
    pub fn from_dir(directory: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let directory = directory.as_ref();
        let mut artifacts = Vec::new();
        for (name, hash, limit) in FILES {
            artifacts.push(verified_bytes(&directory.join(name), hash, limit)?);
        }
        let bert_config: Config = serde_json::from_slice(&artifacts[0])?;
        let mut tokenizer = Tokenizer::from_bytes(&artifacts[1]).map_err(backend)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 256,
                ..Default::default()
            }))
            .map_err(backend)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        let weights = artifacts.pop().unwrap();
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &Device::Cpu)
            .map_err(backend)?;
        let model = BertModel::load(vb, &bert_config).map_err(backend)?;
        let config = EmbeddingConfig {
            model: MINILM_MODEL.into(),
            revision: MINILM_REVISION.into(),
            dimensions: 384,
            normalization: Normalization::L2,
            implementation: "candle-0.11-bert-fp32-cpu-v1".into(),
            pooling: "attention-mask-mean".into(),
            max_tokens: Some(256),
            document_prefix: String::new(),
            query_prefix: String::new(),
            artifacts: FILES
                .into_iter()
                .map(|(name, hash, _)| (name.into(), hash.into()))
                .collect::<BTreeMap<_, _>>(),
        };
        config.validate()?;
        Ok(Self {
            config,
            tokenizer,
            model,
        })
    }

    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() > 32 {
            return Err(EmbeddingError::Invalid(
                "MiniLM reference provider supports at most 32 inputs per batch".into(),
            ));
        }
        let encoded = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(backend)?;
        let device = &Device::Cpu;
        let input_ids = encoded.iter().map(|e| e.get_ids()).collect::<Vec<_>>();
        let type_ids = encoded.iter().map(|e| e.get_type_ids()).collect::<Vec<_>>();
        let masks = encoded
            .iter()
            .map(|e| e.get_attention_mask())
            .collect::<Vec<_>>();
        let input_ids = Tensor::new(input_ids, device).map_err(backend)?;
        let type_ids = Tensor::new(type_ids, device).map_err(backend)?;
        let masks = Tensor::new(masks, device).map_err(backend)?;
        let states = self
            .model
            .forward(&input_ids, &type_ids, Some(&masks))
            .map_err(backend)?;
        // Padding must be masked both in BERT attention AND in pooling. Otherwise a document's
        // embedding changes substantially depending on the lengths of other documents in its batch.
        let weights = masks
            .to_dtype(DType::F32)
            .and_then(|m| m.unsqueeze(2))
            .map_err(backend)?;
        let sums = states
            .broadcast_mul(&weights)
            .and_then(|v| v.sum(1))
            .map_err(backend)?;
        let counts = weights
            .sum(1)
            .and_then(|v| v.clamp(1e-9, f32::MAX))
            .map_err(backend)?;
        let mut vectors = sums
            .broadcast_div(&counts)
            .and_then(|v| v.to_vec2::<f32>())
            .map_err(backend)?;
        for vector in &mut vectors {
            let norm = vector
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for value in vector {
                    *value = (*value as f64 / norm) as f32;
                }
            }
        }
        self.config.validate_vectors(&vectors, texts.len())?;
        Ok(vectors)
    }
}

impl EmbeddingProvider for MiniLm {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    fn embed_documents(&mut self, documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.encode(documents)
    }
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(self.encode(&[query])?.remove(0))
    }
}
