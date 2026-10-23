use std::{collections::HashMap, fs, io::Cursor, process::Command};

use cubeql::{IndexOptions, Metric, write_ipc_batch};
use cubeql_code::{
    BuildOptions, CodeIndex, CodeSearchIndex, build_repository, build_repository_with_provider,
    hash_embedding,
};
use cubeql_embeddings::{
    EmbeddedQuery, EmbeddingConfig, EmbeddingError, EmbeddingProvider, HashProvider, Normalization,
    PrecomputedProvider,
};

struct Spy {
    config: EmbeddingConfig,
    calls: Vec<Vec<String>>,
    bad: Option<&'static str>,
}

impl Spy {
    fn new() -> Self {
        let mut config = HashProvider::new(2).unwrap().config().clone();
        config.model = "test-fixture".into();
        config.normalization = Normalization::None;
        Self {
            config,
            calls: Vec::new(),
            bad: None,
        }
    }
}

impl EmbeddingProvider for Spy {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.calls
            .push(texts.iter().map(|s| s.to_string()).collect());
        match self.bad {
            Some("count") => Ok(Vec::new()),
            Some("dimensions") => Ok(vec![vec![1.0]; texts.len()]),
            Some("nan") => Ok(vec![vec![f32::NAN, 0.0]; texts.len()]),
            Some("mutate") => {
                self.config.revision = "changed".into();
                Ok(vec![vec![1.0, 0.0]; texts.len()])
            }
            Some("error") => Err(EmbeddingError::Backend("offline failure".into())),
            _ => Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, 1.0])
                .collect()),
        }
    }
    fn embed_query(&mut self, _: &str) -> Result<Vec<f32>, EmbeddingError> {
        panic!("incompatible query must fail before inference")
    }
}

#[test]
fn batching_only_encodes_new_or_changed_chunks_in_source_order() {
    let dir = tempfile::tempdir().unwrap();
    let source = "fn a() {}\nfn bb() {}\nfn ccc() {}\nfn dddd() {}\nfn eeeee() {}\n";
    fs::write(dir.path().join("lib.rs"), source).unwrap();
    let mut provider = Spy::new();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let first =
        build_repository_with_provider(dir.path(), &options, None, &mut provider, 2).unwrap();
    assert_eq!(
        provider.calls.iter().map(Vec::len).collect::<Vec<_>>(),
        [2, 2, 1]
    );
    assert_eq!(provider.calls[0], ["fn a() {}", "fn bb() {}"]);
    provider.calls.clear();
    let unchanged =
        build_repository_with_provider(dir.path(), &options, Some(&first.index), &mut provider, 4)
            .unwrap();
    assert_eq!(unchanged.report.reused, 5);
    assert!(provider.calls.is_empty());
    fs::write(
        dir.path().join("lib.rs"),
        source.replace("fn bb() {}", "fn bb() { changed(); }"),
    )
    .unwrap();
    let updated =
        build_repository_with_provider(dir.path(), &options, Some(&first.index), &mut provider, 2)
            .unwrap();
    assert_eq!(
        (
            updated.report.embedded,
            updated.report.reused,
            updated.report.changed
        ),
        (1, 4, 1)
    );
    assert_eq!(
        provider.calls,
        vec![vec!["fn bb() { changed(); }".to_string()]]
    );
    let mut bytes = Vec::new();
    updated.index.write_ipc(&mut bytes).unwrap();
    let loaded = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    assert_eq!(loaded.embedding_config(), Some(provider.config()));
    assert_eq!(loaded.batch(), updated.index.batch());
}

#[test]
fn invalid_outputs_and_configurations_never_build_a_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("lib.rs"), "fn example() {}\n").unwrap();
    for bad in ["count", "dimensions", "nan", "mutate", "error"] {
        let mut provider = Spy::new();
        provider.bad = Some(bad);
        let options = BuildOptions::for_embedding(provider.config()).unwrap();
        assert!(
            build_repository_with_provider(dir.path(), &options, None, &mut provider, 2).is_err()
        );
    }
    let mut provider = Spy::new();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    assert!(build_repository_with_provider(dir.path(), &options, None, &mut provider, 0).is_err());
    assert!(provider.calls.is_empty());
    assert!(
        build_repository_with_provider(
            dir.path(),
            &BuildOptions::default(),
            None,
            &mut provider,
            2
        )
        .is_err()
    );
    let mut normalized = Spy::new();
    normalized.config.normalization = Normalization::L2;
    let options = BuildOptions::for_embedding(normalized.config()).unwrap();
    assert!(
        build_repository_with_provider(dir.path(), &options, None, &mut normalized, 2).is_err()
    );
}

#[test]
fn precomputed_queries_rank_code_and_reject_other_vector_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let read = "fn load_config() { read_from_disk(); }";
    let save = "fn save_config() { write_to_disk(); }";
    fs::write(dir.path().join("lib.rs"), format!("{read}\n{save}\n")).unwrap();
    let mut config = HashProvider::new(2).unwrap().config().clone();
    config.model = "external-fixture".into();
    config.implementation = "fixture-map-v1".into();
    let mut provider = PrecomputedProvider::new(
        config.clone(),
        HashMap::from([(read.into(), vec![1.0, 0.0]), (save.into(), vec![0.0, 1.0])]),
        HashMap::from([("find the configuration reader".into(), vec![1.0, 0.0])]),
    )
    .unwrap();
    let options = BuildOptions::for_embedding(&config).unwrap();
    let built =
        build_repository_with_provider(dir.path(), &options, None, &mut provider, 2).unwrap();
    let mut bytes = Vec::new();
    built.index.write_ipc(&mut bytes).unwrap();
    let code = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code).unwrap();
    for metric in [Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2] {
        let hits = search
            .query_text(&mut provider, "find the configuration reader", 2, metric)
            .unwrap();
        assert_eq!(search.code().chunks()[hits[0].row].symbol, "load_config");
        assert_eq!(hits.len(), 2);
    }
    let mut bad = config.clone();
    bad.revision = "wrong-model-version".into();
    let query = EmbeddedQuery::new(bad, vec![1.0, 0.0]).unwrap();
    assert!(search.query(&query, 1, Metric::Cosine).is_err());
    let mut wrong = Spy::new(); // same dimensions, deliberately panics if inference is called.
    assert!(
        search
            .query_text(&mut wrong, "text", 1, Metric::Cosine)
            .is_err()
    );
    assert!(wrong.calls.is_empty());
    let mut other = config;
    other.query_prefix.push_str("different");
    assert!(
        search
            .query(
                &EmbeddedQuery::new(other, vec![1.0, 0.0]).unwrap(),
                1,
                Metric::Cosine
            )
            .is_err()
    );
}

#[test]
fn incompatible_previous_legacy_indexes_and_tampered_metadata_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("lib.rs"), "fn example() {}\n").unwrap();
    let mut provider = Spy::new();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let built =
        build_repository_with_provider(dir.path(), &options, None, &mut provider, 2).unwrap();
    provider.calls.clear();
    provider.config.revision = "new".into();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    assert!(
        build_repository_with_provider(dir.path(), &options, Some(&built.index), &mut provider, 2)
            .is_err()
    );
    assert!(provider.calls.is_empty());
    let old = build_repository(dir.path(), &BuildOptions::default(), None, |t| {
        hash_embedding(t, 128)
    })
    .unwrap();
    assert!(
        build_repository_with_provider(dir.path(), &options, Some(&old.index), &mut provider, 2)
            .is_err()
    );
    assert!(CodeSearchIndex::new(&cubecl::Device::default(), old.index).is_err());
    for json in [
        "{}".to_string(),
        serde_json::to_string(&provider.config).unwrap(),
    ] {
        let mut batch = built.index.batch().clone();
        batch
            .schema_metadata_mut()
            .insert("cubeql.code.embedding_config", json);
        let mut bytes = Vec::new();
        write_ipc_batch(&batch, "embedding", IndexOptions::default(), &mut bytes).unwrap();
        assert!(CodeIndex::read_ipc(Cursor::new(bytes)).is_err());
    }
    // Even a self-consistent fingerprint cannot make non-unit stored vectors into L2 vectors.
    let mut forged = built.index.embedding_config().unwrap().clone();
    forged.normalization = Normalization::L2;
    let mut batch = built.index.batch().clone();
    batch
        .schema_metadata_mut()
        .insert("cubeql.code.embedding_id", forged.fingerprint().unwrap());
    batch.schema_metadata_mut().insert(
        "cubeql.code.embedding_config",
        serde_json::to_string(&forged).unwrap(),
    );
    let mut bytes = Vec::new();
    write_ipc_batch(&batch, "embedding", IndexOptions::default(), &mut bytes).unwrap();
    assert!(CodeIndex::read_ipc(Cursor::new(bytes)).is_err());
}

#[test]
fn cli_hash_query_and_missing_model_errors_are_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let source = "fn example() {}";
    fs::write(dir.path().join("lib.rs"), source).unwrap();
    let path = dir.path().join("index.arrow");
    let bin = env!("CARGO_BIN_EXE_cubeql-code");
    let run = Command::new(bin)
        .arg("index")
        .arg(dir.path())
        .arg("--output")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let run = Command::new(bin)
        .arg("query")
        .arg(&path)
        .arg(source)
        .arg("--top-k")
        .arg("1")
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("lib.rs:1-1 example score=1.000000"));
    let run = Command::new(bin)
        .arg("index")
        .arg(dir.path())
        .arg("--output")
        .arg(dir.path().join("other.arrow"))
        .arg("--embedding")
        .arg("minilm")
        .output()
        .unwrap();
    assert!(!run.status.success());
    assert!(!dir.path().join("other.arrow").exists());
}

#[cfg(feature = "local-embeddings")]
#[test]
#[ignore = "requires explicitly downloaded MiniLM files; set CUBEQL_MINILM_DIR"]
fn real_local_model_builds_persists_and_finds_code_from_natural_language() {
    let directory = std::env::var("CUBEQL_MINILM_DIR").expect("set CUBEQL_MINILM_DIR");
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("lib.rs"),
        include_str!("fixtures/semantic-demo/config.rs"),
    )
    .unwrap();
    let mut model = cubeql_embeddings::MiniLm::from_dir(directory).unwrap();
    let options = BuildOptions::for_embedding(model.config()).unwrap();
    let built = build_repository_with_provider(dir.path(), &options, None, &mut model, 2).unwrap();
    let mut bytes = Vec::new();
    built.index.write_ipc(&mut bytes).unwrap();
    let loaded = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    let search = CodeSearchIndex::new(&cubecl::Device::default(), loaded).unwrap();
    for (query, expected) in [
        ("Where do we read configuration from a file?", "load_config"),
        ("Compute the product of two numbers", "multiply"),
    ] {
        let hits = search
            .query_text(&mut model, query, 3, Metric::Cosine)
            .unwrap();
        assert_eq!(search.code().chunks()[hits[0].row].symbol, expected);
    }
}
