use std::{collections::HashMap, fs, path::PathBuf, process::Command};

use arrow_array::StringArray;
use cubeql_code::benchmark::{Dataset, RunOptions, run};
use cubeql_embeddings::{EmbeddingProvider, HashProvider, Normalization, PrecomputedProvider};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/external-benchmark")
}

fn copy_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("qrels")).unwrap();
    for name in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"] {
        fs::copy(fixture().join(name), dir.path().join(name)).unwrap();
    }
    dir
}

fn provider() -> PrecomputedProvider {
    let mut config = HashProvider::new(2).unwrap().config().clone();
    config.model = "external-benchmark-precomputed-fixture".into();
    config.normalization = Normalization::None;
    PrecomputedProvider::new(
        config,
        HashMap::from([
            ("Cache\nfn cache_value() { store(); }".into(), vec![0., 1.]),
            ("fn load_config() { read_file(); }".into(), vec![1., 0.]),
            ("fn send_request() { connect(); }".into(), vec![0.6, 0.8]),
        ]),
        HashMap::from([
            ("read settings from disk".into(), vec![1., 0.]),
            ("store a value".into(), vec![0., 1.]),
            ("settings without a judgment".into(), vec![1., 0.]),
        ]),
    )
    .unwrap()
}

#[test]
fn imports_opaque_ids_optional_titles_metadata_and_all_judgment_grades() {
    let dataset = Dataset::load(fixture(), "test").unwrap();
    assert_eq!(dataset.documents()[1].id, "../doc-config");
    assert_eq!(
        dataset.documents()[0].embedding_text(),
        "Cache\nfn cache_value() { store(); }"
    );
    assert_eq!(
        dataset.documents()[1].embedding_text(),
        "fn load_config() { read_file(); }"
    );
    assert_eq!(dataset.summary().documents, 3);
    assert_eq!(dataset.summary().queries, 3);
    assert_eq!(dataset.summary().judgments, 3);
    assert_eq!(dataset.summary().unjudged_queries, 1);
    assert_eq!(dataset.qrels()["q-config"]["doc-cache"], 0);
    assert_eq!(dataset.qrels()["q-config"]["../doc-config"], 2);
    for name in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"] {
        let expected = blake3::hash(&fs::read(fixture().join(name)).unwrap())
            .to_hex()
            .to_string();
        assert_eq!(dataset.summary().input_blake3[name], expected);
    }
}

#[test]
fn exact_cpu_retrieval_exports_native_ids_trec_json_and_arrow_without_qrels_leakage() {
    let dataset = Dataset::load(fixture(), "test").unwrap();
    let run = run(
        &dataset,
        &cubecl::Device::default(),
        &mut provider(),
        RunOptions {
            top_k: 3,
            batch_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let first = &run.report.queries[0];
    assert_eq!(first.query_id, "q-config");
    assert_eq!(
        first
            .hits
            .iter()
            .map(|h| h.document_id.as_str())
            .collect::<Vec<_>>(),
        ["../doc-config", "doc-http", "doc-cache"]
    );
    for (hit, expected) in first.hits.iter().zip([1.0, 0.6, 0.0]) {
        assert!((hit.score - expected).abs() < 1e-6);
    }
    assert_eq!(run.report.queries[1].hits[0].document_id, "doc-cache");
    assert_eq!(run.report.queries[2].query_id, "q-unjudged");
    assert_eq!(run.report.queries[2].hits.len(), 3);
    let output = tempfile::tempdir().unwrap();
    run.save(output.path()).unwrap();
    let results: serde_json::Value =
        serde_json::from_slice(&fs::read(output.path().join("results.json")).unwrap()).unwrap();
    assert_eq!(results["q-config"]["../doc-config"], 1.0);
    assert_eq!(results["q-unjudged"].as_object().unwrap().len(), 3);
    let trec = fs::read_to_string(output.path().join("run.trec")).unwrap();
    assert_eq!(trec.lines().count(), 9);
    assert_eq!(
        trec.lines().next().unwrap(),
        "q-config Q0 ../doc-config 1 1 cubeql-semantic"
    );
    let (batch, vector_column, _) =
        cubeql::read_ipc_batch(fs::File::open(output.path().join("index.arrow")).unwrap()).unwrap();
    assert_eq!(vector_column, "embedding");
    let ids = batch
        .column_by_name("document_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(ids.value(1), "../doc-config");
    assert_eq!(batch.num_rows(), 3);
    assert!(batch.column_by_name("qrels").is_none());
    let metadata = format!("{:?}", batch.schema().metadata());
    assert!(!metadata.contains("q-config"));
    assert!(!metadata.contains("qrels/test.tsv"));
    let before = fs::read(output.path().join("results.json")).unwrap();
    assert!(run.save(output.path()).is_err());
    assert_eq!(
        fs::read(output.path().join("results.json")).unwrap(),
        before
    );
}

#[test]
fn changing_judgments_cannot_change_rankings_or_embedded_index() {
    let dir = copy_fixture();
    let before = Dataset::load(dir.path(), "test").unwrap();
    fs::write(
        dir.path().join("qrels/test.tsv"),
        "query-id\tcorpus-id\tscore\nq-unjudged\tdoc-cache\t-1\nq-config\tdoc-http\t99\n",
    )
    .unwrap();
    let after = Dataset::load(dir.path(), "test").unwrap();
    assert_eq!(after.qrels()["q-unjudged"]["doc-cache"], -1);
    let device = cubecl::Device::default();
    let opts = RunOptions {
        top_k: 3,
        ..Default::default()
    };
    let a = run(&before, &device, &mut provider(), opts.clone()).unwrap();
    let b = run(&after, &device, &mut provider(), opts).unwrap();
    assert_eq!(a.batch(), b.batch());
    for (a, b) in a.report.queries.iter().zip(&b.report.queries) {
        assert_eq!(a.query_id, b.query_id);
        assert_eq!(
            serde_json::to_value(&a.hits).unwrap(),
            serde_json::to_value(&b.hits).unwrap()
        );
    }
    assert_ne!(before.summary().input_blake3, after.summary().input_blake3);
}

#[test]
fn invalid_references_duplicates_and_bad_records_are_rejected_with_locations() {
    for (file, contents, expected) in [
        (
            "qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq-config\tunknown\t0\n",
            "unknown",
        ),
        (
            "qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nunknown\tdoc-cache\t1\n",
            "unknown",
        ),
        (
            "qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq-config\tdoc-cache\t0\nq-config\tdoc-cache\t0\n",
            "duplicate",
        ),
        (
            "qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq-config\tdoc-cache\tNaN\n",
            "integer",
        ),
        ("qrels/test.tsv", "qid\tdocid\tgrade\n", "header"),
        (
            "corpus.jsonl",
            "{\"_id\":\"a\",\"text\":\"a\"}\n{\"_id\":\"a\",\"text\":\"b\"}\n",
            "duplicate document",
        ),
        (
            "corpus.jsonl",
            "{\"_id\":\"bad id\",\"text\":\"a\"}\n",
            "document ID",
        ),
        (
            "queries.jsonl",
            "{\"_id\":\"q\",\"text\":\"a\"}\n{\"_id\":\"q\",\"text\":\"b\"}\n",
            "duplicate query",
        ),
        (
            "queries.jsonl",
            "{\"_id\":\"q\",\"text\":\" \"}\n",
            "text must not be empty",
        ),
        ("queries.jsonl", "not json\n", "queries.jsonl:1:"),
    ] {
        let dir = copy_fixture();
        fs::write(dir.path().join(file), contents).unwrap();
        let error = Dataset::load(dir.path(), "test").err().unwrap().to_string();
        assert!(error.contains(expected), "{error}");
    }
    assert!(Dataset::load(fixture(), "../test").is_err());
}

#[test]
fn header_only_and_zero_only_qrels_do_not_drop_queries_or_claim_no_answer() {
    let dir = copy_fixture();
    for (text, judged) in [
        ("query-id\tcorpus-id\tscore\n", 0),
        (
            "query-id\tcorpus-id\tscore\r\nq-config\tdoc-cache\t0\r\n",
            1,
        ),
    ] {
        fs::write(dir.path().join("qrels/test.tsv"), text).unwrap();
        let dataset = Dataset::load(dir.path(), "test").unwrap();
        assert_eq!(dataset.queries().len(), 3);
        assert_eq!(dataset.summary().judged_queries, judged);
        assert_eq!(dataset.summary().unjudged_queries, 3 - judged);
    }
    fs::write(dir.path().join("qrels/test.tsv"), "").unwrap();
    assert!(Dataset::load(dir.path(), "test").is_err());
}

#[test]
fn malformed_provider_vectors_and_invalid_options_fail_without_query_inference() {
    struct Bad {
        config: cubeql_embeddings::EmbeddingConfig,
    }
    impl EmbeddingProvider for Bad {
        fn config(&self) -> &cubeql_embeddings::EmbeddingConfig {
            &self.config
        }
        fn embed_documents(
            &mut self,
            texts: &[&str],
        ) -> Result<Vec<Vec<f32>>, cubeql_embeddings::EmbeddingError> {
            Ok(vec![vec![1.0]; texts.len()])
        }
        fn embed_query(&mut self, _: &str) -> Result<Vec<f32>, cubeql_embeddings::EmbeddingError> {
            panic!("invalid options/document vectors must fail before query inference")
        }
    }
    let mut bad = Bad {
        config: provider().config().clone(),
    };
    let data = Dataset::load(fixture(), "test").unwrap();
    let device = cubecl::Device::default();
    assert!(
        run(
            &data,
            &device,
            &mut bad,
            RunOptions {
                top_k: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(run(&data, &device, &mut bad, RunOptions::default()).is_err());
    assert!(
        RunOptions {
            run_tag: "tag with spaces".into(),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn cli_validation_and_export_preserve_artifacts_and_record_failures() {
    let binary = env!("CARGO_BIN_EXE_cubeql-code");
    let validation = Command::new(binary)
        .args(["benchmark", "validate"])
        .arg(fixture())
        .output()
        .unwrap();
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&validation.stdout).unwrap();
    assert_eq!(summary["unjudged_queries"], 1);
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("run");
    let execute = || {
        Command::new(binary)
            .args(["benchmark", "run"])
            .arg(fixture())
            .arg("--output")
            .arg(&output)
            .args(["--dimensions", "8", "--top-k", "3"])
            .output()
            .unwrap()
    };
    let result = execute();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let old = fs::read(output.join("report.json")).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&old).unwrap();
    assert_eq!(report["embedding_is_hash_fixture"], true);
    assert_eq!(report["queries"].as_array().unwrap().len(), 3);
    let qrels: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("qrels.json")).unwrap()).unwrap();
    assert_eq!(qrels["q-config"]["../doc-config"], 2);
    assert_eq!(qrels["q-config"]["doc-cache"], 0);
    assert!(!execute().status.success());
    assert_eq!(old, fs::read(output.join("report.json")).unwrap());
    assert!(!output.join("failure.json").exists());
    let failed = dir.path().join("failed");
    let result = Command::new(binary)
        .args(["benchmark", "run"])
        .arg(fixture())
        .arg("--output")
        .arg(&failed)
        .args([
            "--embedding",
            "minilm",
            "--model-dir",
            "/missing-benchmark-model",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(failed.join("dataset.json").exists());
    assert!(failed.join("request.json").exists());
    assert!(failed.join("failure.json").exists());
    assert!(!failed.join("report.json").exists());
}
