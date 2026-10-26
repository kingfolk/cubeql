use std::{collections::BTreeSet, fs, io::Cursor, path::PathBuf, process::Command};

use arrow_array::{Array, FixedSizeListArray, Float16Array};
use cubeql::{Metric, SearchHit};
use cubeql_code::{
    BuildOptions, CodeIndex, CodeSearchIndex, VectorPrecision, build_repository,
    build_repository_with_provider, evaluation::*, reporting::*,
};
use cubeql_embeddings::{
    EmbeddingConfig, EmbeddingError, EmbeddingProvider, HashProvider, embed_query,
};

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/retrieval-benchmark/corpus")
}
fn labels() -> QuerySet {
    serde_json::from_str(include_str!("fixtures/retrieval-benchmark/queries.json")).unwrap()
}
fn fixture() -> (CodeIndex, HashProvider) {
    let mut provider = HashProvider::new(32).unwrap();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let index = build_repository_with_provider(corpus(), &options, None, &mut provider, 4)
        .unwrap()
        .index;
    assert_eq!(index.chunks().len(), 24);
    (index, provider)
}

#[test]
fn metrics_use_all_labels_one_based_ranks_and_zero_for_missing_results() {
    let relevant = BTreeSet::from([4, 8, 99]);
    let metrics = ranking_metrics(&(0..12).collect::<Vec<_>>(), &relevant).unwrap();
    assert_eq!(metrics.recall_at_5, 1.0 / 3.0);
    assert_eq!(metrics.recall_at_10, 2.0 / 3.0);
    assert_eq!(metrics.first_relevant_rank, Some(5));
    assert_eq!(metrics.reciprocal_rank, 0.2);
    let empty = ranking_metrics(&[], &relevant).unwrap();
    assert_eq!(empty.recall_at_5, 0.0);
    assert_eq!(empty.first_relevant_rank, None);
    assert_eq!(empty.reciprocal_rank, 0.0);
    assert!(ranking_metrics(&[1, 1], &relevant).is_err());
    assert!(ranking_metrics(&[], &BTreeSet::new()).is_err());
}

#[test]
fn fusion_combines_ranks_not_incompatible_scores_and_breaks_ties_by_row() {
    let a = [
        SearchHit {
            row: 9,
            score: -100.0,
        },
        SearchHit {
            row: 2,
            score: 9000.0,
        },
    ];
    let b = [
        SearchHit { row: 2, score: 0.1 },
        SearchHit { row: 9, score: 0.9 },
    ];
    let fused = reciprocal_rank_fusion(&[&a, &b], 10);
    assert_eq!(fused.iter().map(|h| h.row).collect::<Vec<_>>(), [2, 9]);
    assert!((fused[0].score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-7);
    let duplicate = [a[0].clone(), a[0].clone()];
    assert!((reciprocal_rank_fusion(&[&duplicate], 1)[0].score - 1.0 / 61.0).abs() < 1e-7);
    assert!(reciprocal_rank_fusion(&[], 10).is_empty());
}

#[test]
fn literal_baseline_and_labels_respect_snapshot_filters_and_reject_bad_labels() {
    let (code, _) = fixture();
    let exact = literal_search(&code, "load_config", &CodeFilter::default(), 10);
    assert_eq!(exact.len(), 1);
    assert_eq!(code.chunks()[exact[0].row].symbol, "load_config");
    assert_eq!(exact[0].score, 3.0);
    assert!(literal_search(&code, "LOAD_CONFIG", &CodeFilter::default(), 10).is_empty());
    let filter = CodeFilter {
        path_prefix: Some("vectors".into()),
        language: Some("rust".into()),
    };
    assert!(literal_search(&code, "load_config", &filter, 10).is_empty());
    let dataset = labels();
    let resolved = dataset.resolve(&code).unwrap();
    assert_eq!(resolved.len(), 8);
    assert_eq!(resolved[6].len(), 2);
    for bad in 0..6 {
        let mut dataset = dataset.clone();
        match bad {
            0 => dataset.queries[0].relevant[0].symbol = "missing".into(),
            1 => dataset.queries[1].id = dataset.queries[0].id.clone(),
            2 => dataset.queries[0].filter = filter.clone(),
            3 => dataset.queries[6].relevant.pop().map(|_| ()).unwrap(),
            4 => {
                let duplicate = dataset.queries[0].relevant[0].clone();
                dataset.queries[0].relevant.push(duplicate);
            }
            _ => dataset.version = 2,
        }
        assert!(dataset.resolve(&code).is_err());
    }
}

#[test]
fn fp16_roundtrip_keeps_model_metadata_rows_and_filtered_device_scores() {
    let (code, mut provider) = fixture();
    let original = SnapshotInfo::new(&code).unwrap();
    let code = code.with_precision(VectorPrecision::Fp16).unwrap();
    assert_eq!(code.precision(), VectorPrecision::Fp16);
    let info = SnapshotInfo::new(&code).unwrap();
    assert_eq!(info.vector_storage_bytes * 2, original.vector_storage_bytes);
    assert_eq!(info.corpus_fingerprint, original.corpus_fingerprint);
    assert_eq!(info.embedding, original.embedding);
    let mut bytes = Vec::new();
    code.write_ipc(&mut bytes).unwrap();
    assert_eq!(bytes.len() as u64, info.ipc_bytes);
    let code = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    let query = embed_query(&mut provider, "directional similarity of vectors").unwrap();
    let filter = CodeFilter {
        language: Some("rust".into()),
        path_prefix: Some("vectors".into()),
    };
    // Independent scalar reference over the actual FP16 Arrow values.
    let arrays = code
        .batch()
        .column_by_name("embedding")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let mut expected = code
        .chunks()
        .iter()
        .enumerate()
        .filter(|(_, c)| filter.matches(c))
        .map(|(row, _)| {
            let vector = arrays.value(row);
            let values = vector.as_any().downcast_ref::<Float16Array>().unwrap();
            let values = values
                .values()
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>();
            let q = query.vector();
            let dot: f32 = values.iter().zip(q).map(|(a, b)| a * b).sum();
            let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt()
                * q.iter().map(|v| v * v).sum::<f32>().sqrt();
            SearchHit {
                row,
                score: if norm == 0.0 { 0.0 } else { dot / norm },
            }
        })
        .collect::<Vec<_>>();
    expected.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.row.cmp(&b.row)));
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code).unwrap();
    let selected = search.prepare_filter(&filter.predicates()).unwrap();
    let hits = selected.query(&query, 10, Metric::Cosine).unwrap();
    assert_eq!(hits.len(), 6);
    for (got, reference) in hits.iter().zip(&expected) {
        assert_eq!(got.row, reference.row);
        assert!((got.score - reference.score).abs() < 1e-5);
    }
    let materialized = materialize(search.code(), &hits, true).unwrap();
    assert!(
        materialized
            .iter()
            .all(|h| h.path == "vectors.rs" && h.source.is_some())
    );
    let mut wrong = query.config().clone();
    wrong.revision.push_str("different");
    let wrong = cubeql_embeddings::EmbeddedQuery::new(wrong, query.vector().to_vec()).unwrap();
    assert!(selected.query(&wrong, 10, Metric::Cosine).is_err());
    let no_matches = CodeFilter {
        language: Some("python".into()),
        ..Default::default()
    };
    assert!(
        search
            .prepare_filter(&no_matches.predicates())
            .unwrap()
            .query(&query, 5, Metric::Cosine)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn incremental_updates_preserve_fp16_without_embedding_unchanged_documents() {
    let (code, mut provider) = fixture();
    let code = code.with_precision(VectorPrecision::Fp16).unwrap();
    let options = code.build_options().unwrap();
    let result =
        build_repository_with_provider(corpus(), &options, Some(&code), &mut provider, 4).unwrap();
    assert_eq!(result.report.embedded, 0);
    assert_eq!(result.report.reused, 24);
    assert_eq!(result.index.precision(), VectorPrecision::Fp16);
    assert_eq!(result.index.batch(), code.batch());
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("lib.rs"), "fn overflow() {}\n").unwrap();
    let options = BuildOptions {
        dimensions: 1,
        ..Default::default()
    };
    let huge = build_repository(dir.path(), &options, None, |_| Ok(vec![100_000.0]))
        .unwrap()
        .index;
    assert!(huge.with_precision(VectorPrecision::Fp16).is_err());
}

#[test]
fn empty_fp16_snapshots_and_ambiguous_labels_have_defined_behavior() {
    let dir = tempfile::tempdir().unwrap();
    let mut provider = HashProvider::new(2).unwrap();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let code = build_repository_with_provider(dir.path(), &options, None, &mut provider, 2)
        .unwrap()
        .index
        .with_precision(VectorPrecision::Fp16)
        .unwrap();
    assert_eq!(SnapshotInfo::new(&code).unwrap().vector_storage_bytes, 0);
    let mut bytes = Vec::new();
    code.write_ipc(&mut bytes).unwrap();
    let code = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    assert_eq!(code.precision(), VectorPrecision::Fp16);
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code).unwrap();
    assert!(
        search
            .query_text(&mut provider, "anything", 5, Metric::Cosine)
            .unwrap()
            .is_empty()
    );

    // syn can parse repeated definitions even though rustc would reject their namespace.
    fs::write(
        dir.path().join("config.rs"),
        "fn load_config() {}\nfn load_config() {}\n",
    )
    .unwrap();
    let code = build_repository_with_provider(dir.path(), &options, None, &mut provider, 2)
        .unwrap()
        .index;
    let mut dataset = labels();
    dataset.queries.truncate(1);
    assert!(dataset.resolve(&code).is_err());
}

struct QuerySpy {
    inner: HashProvider,
    queries: usize,
}
impl EmbeddingProvider for QuerySpy {
    fn config(&self) -> &EmbeddingConfig {
        self.inner.config()
    }
    fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        panic!("evaluation must not re-embed documents")
    }
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.queries += 1;
        self.inner.embed_query(text)
    }
}

#[test]
fn evaluation_reuses_queries_for_both_precisions_and_reports_every_category() {
    let (code, inner) = fixture();
    let mut provider = QuerySpy { inner, queries: 0 };
    let options = EvaluationOptions {
        candidate_k: 24,
        repeats: 2,
        compare_fp16: true,
    };
    let report = evaluate(
        code.clone(),
        &cubecl::Device::default(),
        &mut provider,
        &labels(),
        options,
    )
    .unwrap();
    assert_eq!(provider.queries, 8);
    assert_eq!(report.variants.len(), 2);
    let fp32 = &report.variants[0];
    let fp16 = &report.variants[1];
    assert_eq!(
        fp32.snapshot.corpus_fingerprint,
        fp16.snapshot.corpus_fingerprint
    );
    assert_eq!(
        fp32.snapshot.vector_storage_bytes,
        fp16.snapshot.vector_storage_bytes * 2
    );
    for variant in &report.variants {
        assert_eq!(variant.by_category.len(), 4);
        assert_eq!(variant.overall["literal"].recall_at_5, 0.25);
        assert_eq!(variant.overall["literal"].mrr, 0.25);
        for query in &variant.queries {
            assert_eq!(query.semantic.retrieval.samples, 2);
            assert!(query.semantic.top_hits.len() <= 10);
            assert!(query.semantic.retrieval.min_ms <= query.semantic.retrieval.median_ms);
        }
    }
    assert_eq!(
        fp32.queries[0]
            .semantic
            .top_hits
            .iter()
            .map(|h| h.row)
            .collect::<Vec<_>>(),
        fp16.queries[0]
            .semantic
            .top_hits
            .iter()
            .map(|h| h.row)
            .collect::<Vec<_>>()
    );
    assert!(
        evaluate(
            code.clone().with_precision(VectorPrecision::Fp16).unwrap(),
            &cubecl::Device::default(),
            &mut provider,
            &labels(),
            options
        )
        .is_err()
    );
    assert!(
        evaluate(
            code.clone(),
            &cubecl::Device::default(),
            &mut provider,
            &labels(),
            EvaluationOptions {
                repeats: 0,
                ..options
            }
        )
        .is_err()
    );
    let mut bad = labels();
    bad.queries[0].relevant[0].symbol = "missing".into();
    assert!(
        evaluate(
            code,
            &cubecl::Device::default(),
            &mut provider,
            &bad,
            options
        )
        .is_err()
    );
    assert_eq!(provider.queries, 8); // All failures rejected before inference.
    serde_json::to_value(report).unwrap();
}

fn cli(args: &[&str]) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
        .args(args)
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("stdout must contain only JSON")
}

#[test]
fn cli_search_inspect_update_and_evaluate_are_json_and_snapshot_safe() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("v1.arrow");
    let second = dir.path().join("v2.arrow");
    let first = first.to_str().unwrap();
    let second = second.to_str().unwrap();
    let corpus = corpus();
    let built = cli(&[
        "index",
        corpus.to_str().unwrap(),
        "--output",
        first,
        "--dimensions",
        "32",
        "--precision",
        "fp16",
        "--fallback-lines",
        "12",
        "--max-file-bytes",
        "65536",
    ]);
    assert_eq!(built["snapshot"]["precision"], "fp16");
    let inspected = cli(&["inspect", first]);
    assert_eq!(inspected["snapshot"]["max_file_bytes"], 65536);
    assert_eq!(
        inspected["input_file_bytes"],
        inspected["snapshot"]["ipc_bytes"]
    );
    let search = cli(&[
        "search",
        first,
        "load_config",
        "--language",
        "rust",
        "--path-prefix",
        "config",
        "--top-k",
        "1",
        "--include-source",
    ]);
    // Hashes are not a semantic/symbol ranker: check filters and saved source, not relevance.
    assert_eq!(search["hits"][0]["path"], "config.rs");
    assert!(
        search["hits"][0]["source"]
            .as_str()
            .unwrap()
            .contains("pub fn")
    );
    assert!(search["timings"]["embedding_ms"].is_number());
    let alias = cli(&["query", first, "load_config", "--top-k", "1"]);
    assert!(alias["hits"][0].get("source").is_none());
    let empty = cli(&["search", first, "anything", "--language", "python"]);
    assert!(empty["hits"].as_array().unwrap().is_empty());
    let updated = cli(&["update", first, "--output", second]);
    assert_eq!(updated["report"]["embedded"], 0);
    assert_eq!(updated["report"]["reused"], 24);
    // Arrow allocation capacity depends on whether arrays were just built or loaded from IPC.
    let mut updated_info = updated["snapshot"].clone();
    let mut inspected_info = inspected["snapshot"].clone();
    updated_info
        .as_object_mut()
        .unwrap()
        .remove("arrow_array_memory_bytes");
    inspected_info
        .as_object_mut()
        .unwrap()
        .remove("arrow_array_memory_bytes");
    assert_eq!(updated_info, inspected_info);
    let dataset = corpus.parent().unwrap().join("queries.json");
    let evaluated = cli(&[
        "evaluate",
        first,
        dataset.to_str().unwrap(),
        "--repeats",
        "1",
        "--candidate-k",
        "24",
    ]);
    assert_eq!(
        evaluated["evaluation"]["variants"][0]["queries"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
    let before = fs::read(first).unwrap();
    for args in [
        vec!["update", first, "--output", first],
        vec!["index", corpus.to_str().unwrap(), "--output", first],
    ] {
        assert!(
            !Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
                .args(args)
                .status()
                .unwrap()
                .success()
        );
    }
    let invalid = dir.path().join("invalid.arrow");
    assert!(
        !Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
            .args([
                "index",
                corpus.to_str().unwrap(),
                "--output",
                invalid.to_str().unwrap(),
                "--previous",
                first,
                "--precision",
                "fp32"
            ])
            .status()
            .unwrap()
            .success()
    );
    assert!(!invalid.exists());
    assert_eq!(fs::read(first).unwrap(), before);
}

#[test]
fn cli_update_can_relocate_a_repository_without_changing_build_options() {
    let dir = tempfile::tempdir().unwrap();
    let old_root = dir.path().join("old");
    let new_root = dir.path().join("new");
    fs::create_dir(&old_root).unwrap();
    fs::write(
        old_root.join("example.py"),
        "first\nsecond\nthird\nfourth\n",
    )
    .unwrap();
    let first = dir.path().join("first.arrow");
    let next = dir.path().join("next.arrow");
    cli(&[
        "index",
        old_root.to_str().unwrap(),
        "--output",
        first.to_str().unwrap(),
        "--dimensions",
        "16",
        "--fallback-lines",
        "2",
        "--max-file-bytes",
        "256",
        "--precision",
        "fp16",
    ]);
    fs::rename(&old_root, &new_root).unwrap();
    let updated = cli(&[
        "update",
        first.to_str().unwrap(),
        "--root",
        new_root.to_str().unwrap(),
        "--output",
        next.to_str().unwrap(),
    ]);
    assert_eq!(updated["report"]["embedded"], 0);
    assert_eq!(updated["report"]["reused"], 2);
    assert_eq!(updated["snapshot"]["precision"], "fp16");
    assert_eq!(updated["snapshot"]["fallback_lines"], 2);
    assert_eq!(updated["snapshot"]["max_file_bytes"], 256);
    assert_eq!(
        updated["snapshot"]["root"],
        new_root.canonicalize().unwrap().to_str().unwrap()
    );
}
