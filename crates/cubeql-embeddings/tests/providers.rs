use std::collections::{BTreeMap, HashMap};

use cubeql_embeddings::{
    EmbeddedQuery, EmbeddingConfig, EmbeddingProvider, HashProvider, Normalization,
    PrecomputedProvider, embed_query,
};

fn config() -> EmbeddingConfig {
    EmbeddingConfig {
        model: "fixture".into(),
        revision: "v1".into(),
        dimensions: 2,
        normalization: Normalization::L2,
        implementation: "precomputed-v1".into(),
        pooling: "external".into(),
        max_tokens: None,
        document_prefix: "document: ".into(),
        query_prefix: "query: ".into(),
        artifacts: BTreeMap::new(),
    }
}

#[test]
fn complete_contract_is_stable_and_detects_same_dimension_mismatches() {
    let original = config();
    let json = serde_json::to_string(&original).unwrap();
    let restored: EmbeddingConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(
        original.fingerprint().unwrap(),
        restored.fingerprint().unwrap()
    );
    let mutations: Vec<fn(&mut EmbeddingConfig)> = vec![
        |c| c.model.push('x'),
        |c| c.revision.push('x'),
        |c| c.dimensions = 3,
        |c| c.normalization = Normalization::None,
        |c| c.implementation.push('x'),
        |c| c.pooling.push('x'),
        |c| c.max_tokens = Some(128),
        |c| c.document_prefix.push('x'),
        |c| c.query_prefix.push('x'),
        |c| {
            c.artifacts.insert("weights".into(), "new-checksum".into());
        },
    ];
    for mutate in mutations {
        let mut other = original.clone();
        mutate(&mut other);
        assert!(original.ensure_compatible(&other).is_err());
        assert_ne!(
            original.fingerprint().unwrap(),
            other.fingerprint().unwrap()
        );
    }
    let mut invalid = original.clone();
    invalid.dimensions = 0;
    assert!(invalid.fingerprint().is_err());
    invalid = original;
    invalid.max_tokens = Some(0);
    assert!(invalid.validate().is_err());
}

#[test]
fn precomputed_documents_and_queries_have_separate_ordered_maps() {
    let mut provider = PrecomputedProvider::new(
        config(),
        HashMap::from([("a".into(), vec![1.0, 0.0]), ("b".into(), vec![0.0, 1.0])]),
        HashMap::from([("a".into(), vec![-1.0, 0.0])]),
    )
    .unwrap();
    assert_eq!(
        provider.embed_documents(&["b", "a", "b"]).unwrap(),
        vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]
    );
    let query = embed_query(&mut provider, "a").unwrap();
    assert_eq!(query.vector(), &[-1.0, 0.0]);
    assert_eq!(query.config(), &config());
    assert!(provider.embed_query("b").is_err());
    assert!(provider.embed_documents(&["missing"]).is_err());
    assert!(provider.embed_documents(&[]).unwrap().is_empty());
}

#[test]
fn invalid_values_counts_dimensions_and_normalization_are_rejected() {
    let config = config();
    for vector in [
        vec![1.0],
        vec![f32::NAN, 0.0],
        vec![f32::INFINITY, 0.0],
        vec![2.0, 0.0],
    ] {
        assert!(EmbeddedQuery::new(config.clone(), vector.clone()).is_err());
        assert!(
            PrecomputedProvider::new(
                config.clone(),
                HashMap::from([("x".into(), vector)]),
                HashMap::new()
            )
            .is_err()
        );
    }
    assert!(config.validate_vectors(&[], 1).is_err());
    assert!(EmbeddedQuery::new(config, vec![0.0, 0.0]).is_ok());
}

#[test]
fn hash_fixture_has_same_document_query_encoding_and_unit_norm() {
    let mut provider = HashProvider::new(128).unwrap();
    let documents = provider.embed_documents(&["HELLO world", ""]).unwrap();
    assert_eq!(documents[0], provider.embed_query("hello WORLD").unwrap());
    assert_eq!(documents[1], vec![0.0; 128]);
    provider.config().validate_vectors(&documents, 2).unwrap();
    assert!(HashProvider::new(0).is_err());
}

#[cfg(feature = "minilm")]
#[test]
fn conflicting_local_model_files_are_never_replaced_or_used() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("config.json");
    std::fs::write(&file, "user data").unwrap();
    assert!(cubeql_embeddings::MiniLm::from_dir(dir.path()).is_err());
    // Fails on the existing first artifact before any HTTP request.
    assert!(cubeql_embeddings::download_minilm(dir.path()).is_err());
    assert_eq!(std::fs::read_to_string(file).unwrap(), "user data");
}

#[cfg(feature = "minilm")]
#[test]
#[ignore = "requires explicitly downloaded local MiniLM files; set CUBEQL_MINILM_DIR"]
fn local_minilm_is_normalized_padding_invariant_and_semantic() {
    let directory =
        std::env::var("CUBEQL_MINILM_DIR").expect("set CUBEQL_MINILM_DIR to local model files");
    let mut model = cubeql_embeddings::MiniLm::from_dir(directory).unwrap();
    let texts = [
        "A person is reading a book.",
        "Someone is enjoying a novel in the library.",
        "A database server stores vectors on a graphics processor.",
    ];
    let batch = model.embed_documents(&texts).unwrap();
    model.config().validate_vectors(&batch, 3).unwrap();
    for (text, vector) in texts.iter().zip(&batch) {
        let single = model.embed_query(text).unwrap();
        let max_error = single
            .iter()
            .zip(vector)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error < 5e-4,
            "batch padding changed embedding: {max_error}"
        );
    }
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    assert!(dot(&batch[0], &batch[1]) > dot(&batch[0], &batch[2]));
    assert!(model.embed_documents(&[]).unwrap().is_empty());
    let long = "long source text ".repeat(500);
    let edge = model.embed_documents(&["", &long]).unwrap();
    model.config().validate_vectors(&edge, 2).unwrap();
}
