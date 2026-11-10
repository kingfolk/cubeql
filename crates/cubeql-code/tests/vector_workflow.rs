use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use cubeql_code::{BuildOptions, CodeIndex, prepare_repository};
use cubeql_embeddings::{
    EmbeddingConfig, EmbeddingInput, EmbeddingInputs, EmbeddingProvider, HashProvider,
    Normalization,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
        .args(args)
        .output()
        .expect("cubeql-code should start")
}

fn success(args: &[&str]) -> Output {
    let output = cli(args);
    assert!(
        output.status.success(),
        "command failed: cubeql-code {}\nstdout:\n{}\nstderr:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    output
}

fn failed(args: &[&str]) -> Output {
    let output = cli(args);
    assert!(
        !output.status.success(),
        "command unexpectedly succeeded: cubeql-code {}\nstdout:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
    );
    output
}

fn fixture_config() -> EmbeddingConfig {
    let mut config = HashProvider::new(2).unwrap().config().clone();
    config.model = "offline-external-fixture".into();
    config.revision = "fixture-1".into();
    config.normalization = Normalization::L2;
    config.document_prefix = "document: ".into();
    config.query_prefix = "query: ".into();
    config
}

fn write_bundle(
    root: &Path,
    config: &EmbeddingConfig,
    inputs: &EmbeddingInputs,
    documents: &[[f32; 2]],
    queries: &[[f32; 2]],
) -> PathBuf {
    assert_eq!(documents.len(), inputs.documents.len());
    assert_eq!(queries.len(), inputs.queries.len());
    fs::create_dir_all(root).unwrap();

    let input_bytes = serde_json::to_vec(inputs).unwrap();
    let document_bytes = matrix_bytes(documents);
    let query_bytes = matrix_bytes(queries);
    fs::write(root.join("inputs.json"), &input_bytes).unwrap();
    fs::write(root.join("documents.f32"), &document_bytes).unwrap();
    fs::write(root.join("queries.f32"), &query_bytes).unwrap();

    let manifest = json!({
        "format_version": 1,
        "dtype": "float32_le",
        "config": config,
        "inputs": file_entry("inputs.json", &input_bytes),
        "documents": file_entry("documents.f32", &document_bytes),
        "queries": file_entry("queries.f32", &query_bytes),
    });
    let path = root.join("vectors.json");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    path
}

fn matrix_bytes(matrix: &[[f32; 2]]) -> Vec<u8> {
    matrix
        .iter()
        .flat_map(|row| row.iter().flat_map(|value| value.to_le_bytes()))
        .collect()
}

fn file_entry(name: &str, bytes: &[u8]) -> Value {
    json!({
        "file": name,
        "sha256": format!("{:x}", Sha256::digest(bytes)),
    })
}

fn read_inputs(path: &Path) -> EmbeddingInputs {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn repo(root: &Path, contents: &str) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("lib.rs"), contents).unwrap();
}

fn source_vectors(inputs: &EmbeddingInputs) -> Vec<[f32; 2]> {
    inputs
        .documents
        .iter()
        .map(|input| {
            if input.text.contains("read_settings") || input.text.contains("alpha") {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            }
        })
        .collect()
}

fn parse_stdout_json(output: Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn repository_inputs_index_to_persistent_ipc_and_search_symbols() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    repo(
        &root,
        "pub fn read_settings() -> usize { 1 }\npub fn find_config() -> usize { 2 }\n",
    );
    let input_file = temp.path().join("export.json");
    success(&[
        "vectors",
        "prepare",
        root.to_str().unwrap(),
        "--output",
        input_file.to_str().unwrap(),
        "--query",
        "find config",
        "--query-id",
        "q-config",
    ]);
    let inputs = read_inputs(&input_file);
    assert_eq!(inputs.documents.len(), 2);
    assert_eq!(inputs.queries[0].id, "q-config");

    let bundle = write_bundle(
        &temp.path().join("bundle"),
        &fixture_config(),
        &inputs,
        &source_vectors(&inputs),
        &[[1.0, 0.0]],
    );
    let index = temp.path().join("index.arrow");
    success(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        bundle.to_str().unwrap(),
        "--output",
        index.to_str().unwrap(),
    ]);
    assert!(index.is_file(), "index must be persisted as Arrow IPC");

    let search = parse_stdout_json(success(&[
        "search",
        index.to_str().unwrap(),
        "find config",
        "--vectors",
        bundle.to_str().unwrap(),
        "--query-id",
        "q-config",
        "--strategy",
        "semantic",
        "--format",
        "json",
    ]));
    assert_eq!(search["hits"][0]["symbol"], "read_settings");

    let symbols = parse_stdout_json(success(&[
        "search",
        index.to_str().unwrap(),
        "find_config",
        "--strategy",
        "symbol",
        "--format",
        "json",
    ]));
    assert!(
        symbols["hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["symbol"] == "find_config")
    );
}

#[test]
fn no_op_update_reuses_all_rows_and_changed_source_exports_only_one_pending_input() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    repo(
        &root,
        "pub fn alpha() -> usize { 1 }\npub fn beta() -> usize { 2 }\n",
    );
    let export = temp.path().join("initial.json");
    success(&[
        "vectors",
        "prepare",
        root.to_str().unwrap(),
        "--output",
        export.to_str().unwrap(),
        "--query",
        "alpha",
        "--query-id",
        "q-alpha",
    ]);
    let inputs = read_inputs(&export);
    let config = fixture_config();
    let bundle = write_bundle(
        &temp.path().join("initial-bundle"),
        &config,
        &inputs,
        &source_vectors(&inputs),
        &[[1.0, 0.0]],
    );
    let first_index = temp.path().join("first.arrow");
    success(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        bundle.to_str().unwrap(),
        "--output",
        first_index.to_str().unwrap(),
    ]);

    let noop_index = temp.path().join("noop.arrow");
    let noop = parse_stdout_json(success(&[
        "update",
        first_index.to_str().unwrap(),
        "--output",
        noop_index.to_str().unwrap(),
        "--format",
        "json",
    ]));
    assert_eq!(noop["report"]["reused"], 2);
    assert_eq!(noop["report"]["embedded"], 0);

    // The leading comment shifts alpha's line numbers without changing its source; beta changes.
    repo(
        &root,
        "// moved down\npub fn alpha() -> usize { 1 }\npub fn beta() -> usize { 3 }\n",
    );
    let previous = CodeIndex::load_ipc(&first_index).unwrap();
    let options = BuildOptions::for_embedding(&config).unwrap();
    let plan = prepare_repository(&root, &options).unwrap();
    let pending = plan.pending_inputs(Some(&previous)).unwrap();
    assert_eq!(pending.len(), 1);
    assert!(pending[0].text.contains("beta"));

    let pending_path = temp.path().join("pending.json");
    success(&[
        "vectors",
        "prepare",
        root.to_str().unwrap(),
        "--previous",
        first_index.to_str().unwrap(),
        "--output",
        pending_path.to_str().unwrap(),
    ]);
    let pending_inputs = read_inputs(&pending_path);
    assert_eq!(pending_inputs.documents.len(), 1);
    assert!(pending_inputs.queries.is_empty());
    assert_eq!(pending_inputs.documents[0], pending[0]);
    let changed_bundle = write_bundle(
        &temp.path().join("changed-bundle"),
        &config,
        &pending_inputs,
        &[[0.0, 1.0]],
        &[],
    );
    let updated_index = temp.path().join("updated.arrow");
    let updated = parse_stdout_json(success(&[
        "update",
        first_index.to_str().unwrap(),
        "--vectors",
        changed_bundle.to_str().unwrap(),
        "--output",
        updated_index.to_str().unwrap(),
        "--format",
        "json",
    ]));
    assert_eq!(updated["report"]["reused"], 1);
    assert_eq!(updated["report"]["embedded"], 1);
}

#[test]
fn stale_missing_or_reordered_document_exports_fail_without_creating_an_index() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    repo(
        &root,
        "pub fn alpha() -> usize { 1 }\npub fn beta() -> usize { 2 }\n",
    );
    let export = temp.path().join("export.json");
    success(&[
        "vectors",
        "prepare",
        root.to_str().unwrap(),
        "--output",
        export.to_str().unwrap(),
    ]);
    let inputs = read_inputs(&export);
    let config = fixture_config();
    let vectors = source_vectors(&inputs);

    let stale_bundle = write_bundle(
        &temp.path().join("stale-bundle"),
        &config,
        &inputs,
        &vectors,
        &[],
    );
    repo(
        &root,
        "pub fn alpha() -> usize { 1 }\npub fn beta() -> usize { 9 }\n",
    );
    let stale_output = temp.path().join("stale.arrow");
    failed(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        stale_bundle.to_str().unwrap(),
        "--output",
        stale_output.to_str().unwrap(),
    ]);
    assert!(!stale_output.exists());

    let missing_inputs = EmbeddingInputs {
        format_version: 1,
        documents: vec![inputs.documents[0].clone()],
        queries: Vec::new(),
    };
    let missing_bundle = write_bundle(
        &temp.path().join("missing-bundle"),
        &config,
        &missing_inputs,
        &vectors[..1],
        &[],
    );
    let missing_output = temp.path().join("missing.arrow");
    failed(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        missing_bundle.to_str().unwrap(),
        "--output",
        missing_output.to_str().unwrap(),
    ]);
    assert!(!missing_output.exists());

    let mut reversed = inputs.clone();
    reversed.documents.reverse();
    let reversed_bundle = write_bundle(
        &temp.path().join("reversed-bundle"),
        &config,
        &reversed,
        &vectors.iter().copied().rev().collect::<Vec<_>>(),
        &[],
    );
    let reversed_output = temp.path().join("reversed.arrow");
    failed(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        reversed_bundle.to_str().unwrap(),
        "--output",
        reversed_output.to_str().unwrap(),
    ]);
    assert!(!reversed_output.exists());
}

#[test]
fn identified_query_requires_matching_text_and_full_prefix_contract() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    repo(&root, "pub fn alpha() -> usize { 1 }\n");
    let export = temp.path().join("export.json");
    success(&[
        "vectors",
        "prepare",
        root.to_str().unwrap(),
        "--output",
        export.to_str().unwrap(),
        "--query",
        "alpha",
        "--query-id",
        "q1",
    ]);
    let inputs = read_inputs(&export);
    let config = fixture_config();
    let bundle = write_bundle(
        &temp.path().join("bundle"),
        &config,
        &inputs,
        &source_vectors(&inputs),
        &[[1.0, 0.0]],
    );
    let index = temp.path().join("index.arrow");
    success(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        bundle.to_str().unwrap(),
        "--output",
        index.to_str().unwrap(),
    ]);

    failed(&[
        "search",
        index.to_str().unwrap(),
        "different text",
        "--vectors",
        bundle.to_str().unwrap(),
        "--query-id",
        "q1",
    ]);

    let query_only = EmbeddingInputs {
        format_version: 1,
        documents: Vec::new(),
        queries: inputs.queries.clone(),
    };
    let mut incompatible = config;
    incompatible.query_prefix = "different: ".into();
    let wrong_config_bundle = write_bundle(
        &temp.path().join("wrong-config-bundle"),
        &incompatible,
        &query_only,
        &[],
        &[[1.0, 0.0]],
    );
    failed(&[
        "search",
        index.to_str().unwrap(),
        "alpha",
        "--vectors",
        wrong_config_bundle.to_str().unwrap(),
        "--query-id",
        "q1",
    ]);
}

#[test]
fn benchmark_prepare_and_run_dispatch_identical_text_by_document_and_query_id() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = temp.path().join("dataset");
    fs::create_dir_all(dataset.join("qrels")).unwrap();
    fs::write(
        dataset.join("corpus.jsonl"),
        "{\"_id\":\"doc-a\",\"text\":\"identical document\"}\n{\"_id\":\"doc-b\",\"text\":\"identical document\"}\n",
    )
    .unwrap();
    fs::write(
        dataset.join("queries.jsonl"),
        "{\"_id\":\"q-a\",\"text\":\"same query\"}\n{\"_id\":\"q-b\",\"text\":\"same query\"}\n",
    )
    .unwrap();
    fs::write(
        dataset.join("qrels/test.tsv"),
        "query-id\tcorpus-id\tscore\nq-a\tdoc-a\t1\nq-b\tdoc-b\t1\n",
    )
    .unwrap();

    let export = temp.path().join("benchmark-inputs.json");
    success(&[
        "benchmark",
        "prepare",
        dataset.to_str().unwrap(),
        "--output",
        export.to_str().unwrap(),
    ]);
    let inputs = read_inputs(&export);
    let exported: Value = serde_json::from_slice(&fs::read(&export).unwrap()).unwrap();
    assert!(exported.get("qrels").is_none());
    assert_eq!(inputs.documents[0].text, inputs.documents[1].text);
    assert_eq!(inputs.queries[0].text, inputs.queries[1].text);

    let bundle = write_bundle(
        &temp.path().join("benchmark-bundle"),
        &fixture_config(),
        &inputs,
        &[[1.0, 0.0], [0.0, 1.0]],
        &[[1.0, 0.0], [0.0, 1.0]],
    );
    let run_dir = temp.path().join("run");
    success(&[
        "benchmark",
        "run",
        dataset.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        bundle.to_str().unwrap(),
        "--output",
        run_dir.to_str().unwrap(),
        "--top-k",
        "2",
    ]);
    let results: Value =
        serde_json::from_slice(&fs::read(run_dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(results["q-a"]["doc-a"], 1.0);
    assert_eq!(results["q-a"]["doc-b"], 0.0);
    assert_eq!(results["q-b"]["doc-a"], 0.0);
    assert_eq!(results["q-b"]["doc-b"], 1.0);
}

#[test]
fn repository_evaluator_preserves_distinct_ids_for_identical_query_text() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    repo(&root, "fn alpha() -> u32 { 1 }\nfn beta() -> u32 { 2 }\n");
    let config = fixture_config();
    let plan = prepare_repository(&root, &BuildOptions::for_embedding(&config).unwrap()).unwrap();
    let inputs = EmbeddingInputs {
        format_version: 1,
        documents: plan.pending_inputs(None).unwrap(),
        queries: vec![
            EmbeddingInput::new("q-alpha", "same question"),
            EmbeddingInput::new("q-beta", "same question"),
        ],
    };
    let bundle = write_bundle(
        &temp.path().join("bundle"),
        &config,
        &inputs,
        &source_vectors(&inputs),
        &[[1., 0.], [0., 1.]],
    );
    let index = temp.path().join("code.arrow");
    success(&[
        "index",
        root.to_str().unwrap(),
        "--embedding",
        "precomputed",
        "--vectors",
        bundle.to_str().unwrap(),
        "--output",
        index.to_str().unwrap(),
    ]);
    let queries = temp.path().join("queries.json");
    let cases = json!({"version":1,"name":"identified query control","queries":[
        {"id":"q-alpha","category":"conceptual","text":"same question","relevant":[{"path":"lib.rs","symbol":"alpha"}]},
        {"id":"q-beta","category":"conceptual","text":"same question","relevant":[{"path":"lib.rs","symbol":"beta"}]}
    ]});
    fs::write(&queries, serde_json::to_vec(&cases).unwrap()).unwrap();
    let output = success(&[
        "evaluate",
        index.to_str().unwrap(),
        queries.to_str().unwrap(),
        "--vectors",
        bundle.to_str().unwrap(),
        "--candidate-k",
        "10",
        "--repeats",
        "1",
        "--format",
        "json",
    ]);
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["evaluation"]["embedding_execution"],
        "precomputed-vector-lookup"
    );
    assert_eq!(
        report["evaluation"]["variants"][0]["overall"]["semantic"]["mrr"],
        1.0
    );
}

#[test]
fn existing_live_encoder_evaluation_keeps_non_token_query_ids() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    repo(&root, "fn alpha() -> u32 { 1 }\n");
    let index = temp.path().join("code.arrow");
    success(&[
        "index",
        root.to_str().unwrap(),
        "--output",
        index.to_str().unwrap(),
    ]);
    let cases = json!({"version":1,"name":"live compatibility","queries":[
        {"id":"question with spaces","category":"identifier","text":"alpha","relevant":[{"path":"lib.rs","symbol":"alpha"}]}
    ]});
    let queries = temp.path().join("cases.json");
    fs::write(&queries, serde_json::to_vec(&cases).unwrap()).unwrap();
    success(&[
        "evaluate",
        index.to_str().unwrap(),
        queries.to_str().unwrap(),
        "--repeats",
        "1",
    ]);
}
