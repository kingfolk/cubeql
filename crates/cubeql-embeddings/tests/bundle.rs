use std::{fs, path::Path, path::PathBuf};

use cubeql_embeddings::{
    BundleProvider, EmbeddingConfig, EmbeddingInput, EmbeddingInputs, EmbeddingProvider,
    HashProvider, Normalization, VectorBundle,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

struct Fixture {
    dir: tempfile::TempDir,
    config: EmbeddingConfig,
    documents: Vec<EmbeddingInput>,
    queries: Vec<EmbeddingInput>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            config: EmbeddingConfig {
                model: "test-model".into(),
                revision: "r1".into(),
                dimensions: 2,
                normalization: Normalization::None,
                implementation: "test-encoder".into(),
                pooling: "mean".into(),
                max_tokens: Some(64),
                document_prefix: "doc: ".into(),
                query_prefix: "query: ".into(),
                artifacts: Default::default(),
            },
            documents: vec![
                EmbeddingInput::new("d1", "same text"),
                EmbeddingInput::new("d2", "same text"),
            ],
            queries: vec![EmbeddingInput::new("q1", "find it")],
        }
    }

    fn write(&self, documents: &[f32], queries: &[f32]) -> PathBuf {
        let root = self.dir.path();
        let inputs = EmbeddingInputs {
            format_version: 1,
            documents: self.documents.clone(),
            queries: self.queries.clone(),
        };
        let inputs_bytes = serde_json::to_vec(&inputs).unwrap();
        let document_bytes = floats_to_bytes(documents);
        let query_bytes = floats_to_bytes(queries);
        fs::write(root.join("inputs.json"), &inputs_bytes).unwrap();
        fs::write(root.join("documents.f32"), &document_bytes).unwrap();
        fs::write(root.join("queries.f32"), &query_bytes).unwrap();
        let manifest = json!({
            "format_version": 1,
            "dtype": "float32_le",
            "config": self.config,
            "inputs": file_entry("inputs.json", &inputs_bytes),
            "documents": file_entry("documents.f32", &document_bytes),
            "queries": file_entry("queries.f32", &query_bytes),
        });
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        root.join("manifest.json")
    }

    fn read_manifest(&self) -> Value {
        serde_json::from_slice(&fs::read(self.dir.path().join("manifest.json")).unwrap()).unwrap()
    }

    fn save_manifest(&self, value: &Value) -> PathBuf {
        let path = self.dir.path().join("manifest.json");
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        path
    }
}

fn file_entry(name: &str, bytes: &[u8]) -> Value {
    json!({"file": name, "sha256": format!("{:x}", Sha256::digest(bytes))})
}

fn floats_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn assert_invalid<T>(result: Result<T, cubeql_embeddings::EmbeddingError>) {
    assert!(matches!(
        result,
        Err(cubeql_embeddings::EmbeddingError::Invalid(_))
    ));
}

#[test]
fn loads_ordered_vectors_and_keeps_duplicate_texts_distinct_by_id() {
    let fixture = Fixture::new();
    let manifest = fixture.write(&[1.0, 0.0, 0.0, 1.0], &[0.25, 0.75]);
    let bundle = VectorBundle::load(&manifest).unwrap();

    bundle.ensure_documents(&fixture.documents).unwrap();
    bundle.ensure_queries(&fixture.queries).unwrap();
    assert_eq!(bundle.document(&fixture.documents[0]).unwrap(), [1.0, 0.0]);
    assert_eq!(bundle.document(&fixture.documents[1]).unwrap(), [0.0, 1.0]);
    assert_eq!(
        bundle.query(&fixture.queries[0]).unwrap().vector(),
        [0.25, 0.75]
    );
    assert_eq!(bundle.config().document_prefix, "doc: ");
    assert_eq!(bundle.config().query_prefix, "query: ");
}

#[test]
fn rejects_reordered_or_changed_expected_inputs_and_changed_lookup_content() {
    let fixture = Fixture::new();
    let manifest = fixture.write(&[1.0, 0.0, 0.0, 1.0], &[0.25, 0.75]);
    let bundle = VectorBundle::load(manifest).unwrap();

    let mut reordered = fixture.documents.clone();
    reordered.swap(0, 1);
    assert_invalid(bundle.ensure_documents(&reordered));
    let mut changed_text = fixture.documents[0].clone();
    changed_text.text = "different".into();
    assert_invalid(bundle.ensure_documents(&[changed_text.clone(), fixture.documents[1].clone()]));
    assert_invalid(bundle.document(&changed_text));
    assert_invalid(bundle.document_by_id("d1", "different"));
}

#[test]
fn rejects_corrupt_hashes_and_unrecognized_manifest_schema() {
    let fixture = Fixture::new();
    let manifest = fixture.write(&[1.0, 0.0, 0.0, 1.0], &[0.25, 0.75]);
    let bundle = VectorBundle::load(&manifest).unwrap();
    let mut bad = fixture.documents[0].clone();
    bad.content_hash = "0".repeat(64);
    assert_invalid(bad.validate());
    assert_invalid(bundle.ensure_documents(&[bad, fixture.documents[1].clone()]));

    let mut json = fixture.read_manifest();
    json["surprise"] = json!(true);
    assert!(VectorBundle::load(fixture.save_manifest(&json)).is_err());
}

#[test]
fn rejects_bad_sha_and_unsafe_filenames() {
    let fixture = Fixture::new();
    fixture.write(&[1.0, 0.0, 0.0, 1.0], &[0.25, 0.75]);

    let mut json = fixture.read_manifest();
    json["documents"]["sha256"] = json!("f".repeat(64));
    assert_invalid(VectorBundle::load(fixture.save_manifest(&json)));

    let mut json = fixture.read_manifest();
    json["documents"]["file"] = json!("../documents.f32");
    assert_invalid(VectorBundle::load(fixture.save_manifest(&json)));
}

#[test]
fn rejects_wrong_matrix_shape_non_finite_values_and_non_unit_l2_vectors() {
    let fixture = Fixture::new();
    let manifest = fixture.write(&[1.0, 0.0, 0.0], &[0.25, 0.75]);
    assert_invalid(VectorBundle::load(manifest));

    let manifest = fixture.write(&[f32::NAN, 0.0, 0.0, 1.0], &[0.25, 0.75]);
    assert_invalid(VectorBundle::load(manifest));

    let mut l2 = Fixture::new();
    l2.config.normalization = Normalization::L2;
    let manifest = l2.write(&[2.0, 0.0, 0.0, 1.0], &[0.25, 0.75]);
    assert_invalid(VectorBundle::load(manifest));
}

#[test]
fn rejects_duplicate_ids_per_role_and_blank_queries() {
    let mut inputs = EmbeddingInputs {
        format_version: 1,
        documents: vec![
            EmbeddingInput::new("dup", "a"),
            EmbeddingInput::new("dup", "b"),
        ],
        queries: vec![EmbeddingInput::new("q", "query")],
    };
    assert_invalid(inputs.validate());

    inputs.documents = vec![EmbeddingInput::new("d", "document")];
    inputs.queries = vec![EmbeddingInput::new("q", " \n\t ")];
    assert_invalid(inputs.validate());

    inputs.format_version = 2;
    inputs.queries = vec![EmbeddingInput::new("q", "query")];
    assert_invalid(inputs.validate());
}

#[test]
fn rejects_invalid_configuration_and_validates_input_ids() {
    let fixture = Fixture::new();
    let manifest = fixture.write(&[1.0, 0.0, 0.0, 1.0], &[0.25, 0.75]);
    let mut json = fixture.read_manifest();
    json["config"]["dimensions"] = json!(0);
    assert_invalid(VectorBundle::load(fixture.save_manifest(&json)));

    let invalid_id = EmbeddingInput::new("bad\nid", "text");
    assert_invalid(invalid_id.validate());
    let spaced_id = EmbeddingInput::new("two words", "text");
    assert_invalid(spaced_id.validate());
    let blank_id = EmbeddingInput::new(" \t", "text");
    assert_invalid(blank_id.validate());
    assert!(Path::new(&manifest).exists());
}

#[test]
fn bundle_provider_dispatches_by_id_and_rejects_ambiguous_text_lookup() {
    let mut fixture = Fixture::new();
    fixture.queries = vec![
        EmbeddingInput::new("q1", "same query"),
        EmbeddingInput::new("q2", "same query"),
        EmbeddingInput::new("q3", "unique query"),
    ];
    let manifest = fixture.write(&[1.0, 0.0, 0.0, 1.0], &[0.2, 0.8, 0.6, 0.4, 0.75, 0.25]);
    let bundle = VectorBundle::load(&manifest).unwrap();
    let mut provider = BundleProvider::new(bundle);

    let documents = provider
        .embed_documents_with_ids(&fixture.documents)
        .unwrap();
    assert_eq!(documents, [vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_invalid(provider.embed_documents(&["same text"]));

    assert_eq!(
        provider.embed_query_with_id(&fixture.queries[0]).unwrap(),
        [0.2, 0.8]
    );
    assert_eq!(
        provider.embed_query_with_id(&fixture.queries[1]).unwrap(),
        [0.6, 0.4]
    );
    assert_eq!(provider.embed_query("unique query").unwrap(), [0.75, 0.25]);
    assert_invalid(provider.embed_query("same query"));
    assert!(matches!(
        provider.embed_query("absent query"),
        Err(cubeql_embeddings::EmbeddingError::Missing { kind: "query" })
    ));
}

#[test]
fn convenience_input_constructor_hashes_raw_utf8_and_hash_provider_remains_available() {
    let input = EmbeddingInput::new("id", "é");
    assert_eq!(
        input.content_hash,
        blake3::hash("é".as_bytes()).to_hex().as_str()
    );
    input.validate().unwrap();
    let provider = HashProvider::new(8).unwrap();
    assert_eq!(provider.config().document_prefix, "");
}
