//! Portable experiment adapter for ID-bound precomputed vectors, not a model implementation.
//! Build with `cargo build -p cubeql-code --example benchmark_replay --release`.
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::{Component, Path, PathBuf},
};

use clap::Parser;
use cubeql_code::benchmark::{self, Dataset, RunOptions};
use cubeql_embeddings::{EmbeddingConfig, EmbeddingError, EmbeddingProvider};
use serde::Deserialize;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    dataset: PathBuf,
    vectors: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 100)]
    top_k: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorFile {
    file: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u32,
    config: EmbeddingConfig,
    dataset_sha256: BTreeMap<String, String>,
    document_ids: Vec<String>,
    query_ids: Vec<String>,
    documents: VectorFile,
    queries: VectorFile,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_vectors(
    root: &Path,
    file: &VectorFile,
    rows: usize,
    config: &EmbeddingConfig,
) -> Result<Vec<Vec<f32>>> {
    let mut components = Path::new(&file.file).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err("vector filename must be a single path component".into());
    }
    let bytes = fs::read(root.join(&file.file))?;
    if sha256(&bytes) != file.sha256 {
        return Err(format!("vector checksum mismatch: {}", file.file).into());
    }
    let expected = rows
        .checked_mul(config.dimensions)
        .and_then(|n| n.checked_mul(4))
        .ok_or("vector dimensions overflow")?;
    if bytes.len() != expected {
        return Err("vector byte length does not match rows × dimensions × FP32".into());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let vectors = values
        .chunks_exact(config.dimensions)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    config.validate_vectors(&vectors, rows)?;
    Ok(vectors)
}

/// Replay is ordered, rather than keyed by text: duplicate documents retain their own IDs
/// and vectors, including small batch-dependent numeric differences from model inference.
struct Replay {
    config: EmbeddingConfig,
    manifest_sha256: String,
    documents: VecDeque<(String, Vec<f32>)>,
    queries: VecDeque<(String, Vec<f32>)>,
}

fn take(
    rows: &mut VecDeque<(String, Vec<f32>)>,
    text: &str,
) -> std::result::Result<Vec<f32>, EmbeddingError> {
    let (expected, vector) = rows
        .pop_front()
        .ok_or_else(|| EmbeddingError::Invalid("replay exhausted".into()))?;
    if expected != text {
        return Err(EmbeddingError::Invalid("replay text/order mismatch".into()));
    }
    Ok(vector)
}

impl EmbeddingProvider for Replay {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    fn embed_documents(
        &mut self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingError> {
        texts
            .iter()
            .map(|text| take(&mut self.documents, text))
            .collect()
    }
    fn embed_query(&mut self, text: &str) -> std::result::Result<Vec<f32>, EmbeddingError> {
        take(&mut self.queries, text)
    }
}

fn prepare(dataset_path: &Path, manifest_path: &Path) -> Result<(Dataset, Replay)> {
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.format_version != 1 {
        return Err("unsupported vector manifest version".into());
    }
    manifest.config.validate()?;
    let mut read_hashes = BTreeMap::new();
    let expected_names = BTreeSet::from(["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]);
    if manifest
        .dataset_sha256
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_names
    {
        return Err("dataset manifest must bind corpus, queries and test qrels".into());
    }
    for (name, expected) in &manifest.dataset_sha256 {
        let bytes = fs::read(dataset_path.join(name))?;
        if sha256(&bytes) != *expected {
            return Err(format!("dataset checksum mismatch: {name}").into());
        }
        read_hashes.insert(name.clone(), blake3::hash(&bytes).to_hex().to_string());
    }
    let dataset = Dataset::load(dataset_path, "test")?;
    // Bind SHA-256 checks to the exact bytes parsed by the library's streaming loader.
    if dataset.summary().input_blake3 != read_hashes {
        return Err("dataset changed while being loaded".into());
    }
    if !dataset
        .documents()
        .iter()
        .map(|d| &d.id)
        .eq(&manifest.document_ids)
        || !dataset
            .queries()
            .iter()
            .map(|q| &q.id)
            .eq(&manifest.query_ids)
    {
        return Err("vector IDs/order disagree with dataset".into());
    }
    let root = manifest_path
        .parent()
        .ok_or("vector manifest has no parent")?;
    let documents = read_vectors(
        root,
        &manifest.documents,
        dataset.documents().len(),
        &manifest.config,
    )?;
    let queries = read_vectors(
        root,
        &manifest.queries,
        dataset.queries().len(),
        &manifest.config,
    )?;
    let replay = Replay {
        config: manifest.config,
        manifest_sha256: sha256(&manifest_bytes),
        documents: dataset
            .documents()
            .iter()
            .map(|d| d.embedding_text())
            .zip(documents)
            .collect(),
        queries: dataset
            .queries()
            .iter()
            .map(|q| q.text.clone())
            .zip(queries)
            .collect(),
    };
    Ok((dataset, replay))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let options = RunOptions {
        top_k: args.top_k,
        batch_size: 8,
        run_tag: "cubeql-precomputed".into(),
    };
    options.validate()?;
    let (dataset, mut replay) = prepare(&args.dataset, &args.vectors)?;
    fs::create_dir(&args.output)?;
    let outcome = (|| -> Result<()> {
        benchmark::write_json(&args.output.join("dataset.json"), dataset.summary())?;
        benchmark::write_json(&args.output.join("qrels.json"), dataset.qrels())?;
        benchmark::write_json(
            &args.output.join("replay.json"),
            &serde_json::json!({
                "vectors_manifest_sha256": replay.manifest_sha256,
                "embedding_timers": "precomputed-vector-lookup; model inference is recorded separately",
                "executable_sha256": sha256(&fs::read(std::env::current_exe()?)?)
            }),
        )?;
        let run = benchmark::run(&dataset, &cubecl::Device::default(), &mut replay, options)?;
        if !replay.documents.is_empty() || !replay.queries.is_empty() {
            return Err("unused replay vectors".into());
        }
        run.save(&args.output)?;
        Ok(())
    })();
    if let Err(error) = outcome {
        benchmark::write_json(
            &args.output.join("failure.json"),
            &serde_json::json!({"error": error.to_string()}),
        )?;
        return Err(error);
    }
    println!("completed {} queries", dataset.queries().len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubeql_embeddings::{HashProvider, Normalization};

    #[test]
    fn vector_bytes_hash_shape_and_numeric_contract_are_checked() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = [1_f32, 0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        fs::write(dir.path().join("v.f32"), &bytes).unwrap();
        let mut config = HashProvider::new(2).unwrap().config().clone();
        config.normalization = Normalization::None;
        let mut file = VectorFile {
            file: "v.f32".into(),
            sha256: sha256(&bytes),
        };
        assert_eq!(
            read_vectors(dir.path(), &file, 1, &config).unwrap(),
            [vec![1., 0.]]
        );
        assert!(read_vectors(dir.path(), &file, 2, &config).is_err());
        file.sha256 = "wrong".into();
        assert!(read_vectors(dir.path(), &file, 1, &config).is_err());
        file.file = "../v.f32".into();
        assert!(read_vectors(dir.path(), &file, 1, &config).is_err());
    }

    #[test]
    fn duplicate_texts_retain_distinct_ordered_vectors() {
        let mut rows =
            VecDeque::from([("same".into(), vec![1., 0.]), ("same".into(), vec![0., 1.])]);
        assert_eq!(take(&mut rows, "same").unwrap(), [1., 0.]);
        assert_eq!(take(&mut rows, "same").unwrap(), [0., 1.]);
        assert!(take(&mut rows, "same").is_err());
        let mut rows = VecDeque::from([("expected".into(), vec![1.])]);
        assert!(take(&mut rows, "other").is_err());
    }
}
