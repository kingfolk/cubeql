use std::{collections::HashMap, fs, process::Command};

use cubeql::Metric;
use cubeql_code::{
    BuildOptions, CodeIndex, CodeSearchIndex, VectorPrecision, build_repository_with_provider,
    reporting::CodeFilter,
    retrieval::{
        RetrievalOptions, RetrievalSource, SearchStrategy, SnapshotTextIndex, materialize_hits,
    },
};
use cubeql_embeddings::{
    EmbeddedQuery, EmbeddingProvider, HashProvider, Normalization, PrecomputedProvider, embed_query,
};

const TARGET: &str = "fn target() { needle_api(); }";
const DISTRACTOR: &str = "fn distractor() { unrelated(); }";
const OTHER: &str = "fn other() { something(); }";

fn fixture() -> (tempfile::TempDir, CodeIndex, PrecomputedProvider) {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("lib.rs"),
        format!("{TARGET}\n{DISTRACTOR}\n{OTHER}\n"),
    )
    .unwrap();
    let mut config = HashProvider::new(2).unwrap().config().clone();
    config.model = "fixture-not-implemented-by-cli".into();
    config.normalization = Normalization::None;
    let mut provider = PrecomputedProvider::new(
        config.clone(),
        HashMap::from([
            (TARGET.into(), vec![0.0, 1.0]),
            (DISTRACTOR.into(), vec![1.0, 0.0]),
            (OTHER.into(), vec![0.9, 0.1]),
        ]),
        HashMap::from([
            ("target".into(), vec![1.0, 0.0]),
            ("needle api".into(), vec![1.0, 0.0]),
        ]),
    )
    .unwrap();
    let built = build_repository_with_provider(
        dir.path(),
        &BuildOptions::for_embedding(&config).unwrap(),
        None,
        &mut provider,
        2,
    )
    .unwrap();
    (dir, built.index, provider)
}
fn options(strategy: SearchStrategy) -> RetrievalOptions {
    RetrievalOptions {
        strategy,
        top_k: 3,
        candidate_k: 3,
        deduplicate: true,
    }
}

#[test]
fn lexical_and_symbol_queries_need_no_model_or_device_and_search_saved_source() {
    let (dir, code, _) = fixture();
    let index = SnapshotTextIndex::new(&code);
    let prepared = index.prepare(&CodeFilter::default(), None).unwrap();
    // No whole-query source match: identifier splitting connects these two words to needle_api.
    let hits = prepared
        .query("needle API", None, options(SearchStrategy::Lexical))
        .unwrap();
    assert_eq!(hits.hits.len(), 1);
    assert_eq!(code.chunks()[hits.hits[0].row].symbol, "target");
    assert_eq!(hits.hits[0].provenance[0].source, RetrievalSource::Lexical);
    assert_eq!(hits.timings.semantic_ms, 0.0);
    let hits = prepared
        .query("find target", None, options(SearchStrategy::Symbol))
        .unwrap();
    assert_eq!(code.chunks()[hits.hits[0].row].symbol, "target");
    assert!(
        prepared
            .query("TARGET", None, options(SearchStrategy::Symbol))
            .unwrap()
            .hits
            .is_empty()
    );
    assert!(
        !prepared
            .query("TARGET", None, options(SearchStrategy::Lexical))
            .unwrap()
            .hits
            .is_empty()
    );
    fs::write(dir.path().join("lib.rs"), "fn completely_changed() {}\n").unwrap();
    let hits = prepared
        .query("needle_api", None, options(SearchStrategy::Lexical))
        .unwrap();
    let code_hits = materialize_hits(&code, &hits.hits, true).unwrap();
    assert_eq!(code_hits[0].code.source.as_deref(), Some(TARGET));
    let filter = CodeFilter {
        path_prefix: Some("absent/".into()),
        ..Default::default()
    };
    assert!(
        index
            .prepare(&filter, None)
            .unwrap()
            .query("target", None, options(SearchStrategy::Symbol))
            .unwrap()
            .hits
            .is_empty()
    );
}

#[test]
fn hybrid_can_recover_an_exact_identifier_missed_by_vector_ranking_and_explains_scores() {
    for precision in [VectorPrecision::Fp32, VectorPrecision::Fp16] {
        let (_dir, code, mut provider) = fixture();
        let search = CodeSearchIndex::new(
            &cubecl::Device::default(),
            code.with_precision(precision).unwrap(),
        )
        .unwrap();
        let text = SnapshotTextIndex::new(search.code());
        let prepared = text.prepare(&CodeFilter::default(), Some(&search)).unwrap();
        let query = embed_query(&mut provider, "target").unwrap();
        let semantic = prepared
            .query("target", Some(&query), options(SearchStrategy::Semantic))
            .unwrap();
        assert_eq!(
            search.code().chunks()[semantic.hits[0].row].symbol,
            "distractor"
        );
        let hybrid = prepared
            .query("target", Some(&query), options(SearchStrategy::Hybrid))
            .unwrap();
        let target = &hybrid.hits[0];
        assert_eq!(search.code().chunks()[target.row].symbol, "target");
        assert_eq!(
            target
                .provenance
                .iter()
                .map(|p| p.source)
                .collect::<Vec<_>>(),
            [
                RetrievalSource::Semantic,
                RetrievalSource::Lexical,
                RetrievalSource::Symbol
            ]
        );
        assert_eq!(
            target.provenance.iter().map(|p| p.rank).collect::<Vec<_>>(),
            [3, 1, 1]
        );
        let expected: f64 = target
            .provenance
            .iter()
            .map(|p| 1.0 / (60.0 + p.rank as f64))
            .sum();
        assert!((target.score as f64 - expected).abs() < 1e-7);
        let raw = search.query(&query, 3, Metric::Cosine).unwrap();
        assert_eq!(
            raw.iter().map(|h| h.row).collect::<Vec<_>>(),
            semantic.hits.iter().map(|h| h.row).collect::<Vec<_>>()
        );
        assert!(target.provenance.iter().all(|p| p.rank > 0));
    }
}

#[test]
fn every_branch_obeys_the_same_filter_and_bad_models_or_snapshots_are_rejected() {
    let (_dir, code, mut provider) = fixture();
    let clone = code.clone();
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code).unwrap();
    let wrong_snapshot = SnapshotTextIndex::new(&clone);
    assert!(
        wrong_snapshot
            .prepare(&CodeFilter::default(), Some(&search))
            .is_err()
    );
    let text = SnapshotTextIndex::new(search.code());
    let query = embed_query(&mut provider, "target").unwrap();
    let filter = CodeFilter {
        language: Some("python".into()),
        ..Default::default()
    };
    let prepared = text.prepare(&filter, Some(&search)).unwrap();
    for strategy in [
        SearchStrategy::Semantic,
        SearchStrategy::Lexical,
        SearchStrategy::Symbol,
        SearchStrategy::Hybrid,
    ] {
        assert!(
            prepared
                .query("target", Some(&query), options(strategy))
                .unwrap()
                .hits
                .is_empty()
        );
    }
    let mut wrong = query.config().clone();
    wrong.revision.push_str("wrong");
    let wrong = EmbeddedQuery::new(wrong, query.vector().to_vec()).unwrap();
    assert!(
        prepared
            .query("target", Some(&wrong), options(SearchStrategy::Hybrid))
            .is_err()
    );
    assert!(
        prepared
            .query("target", None, options(SearchStrategy::Hybrid))
            .is_err()
    );
    let host = text.prepare(&CodeFilter::default(), None).unwrap();
    assert!(
        host.query("target", Some(&query), options(SearchStrategy::Hybrid))
            .is_err()
    );
    assert!(
        host.query(" ", None, options(SearchStrategy::Lexical))
            .is_err()
    );
    let mut invalid = options(SearchStrategy::Lexical);
    invalid.candidate_k = 2;
    assert!(host.query("target", None, invalid).is_err());
    invalid.top_k = 0;
    assert!(host.query("target", None, invalid).unwrap().hits.is_empty());
}

#[test]
fn exact_symbols_are_case_sensitive_qualified_and_distinct_from_line_fallbacks() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("lib.rs"),
        "mod config {\n fn load() {}\n}\nmod other {\n fn load() {}\n}\n",
    )
    .unwrap();
    fs::write(dir.path().join("script.py"), "def load(): pass\n").unwrap();
    let mut provider = HashProvider::new(8).unwrap();
    let code = build_repository_with_provider(
        dir.path(),
        &BuildOptions::for_embedding(provider.config()).unwrap(),
        None,
        &mut provider,
        2,
    )
    .unwrap()
    .index;
    let index = SnapshotTextIndex::new(&code);
    let prepared = index.prepare(&CodeFilter::default(), None).unwrap();
    let qualified = prepared
        .query("config::load", None, options(SearchStrategy::Symbol))
        .unwrap();
    assert_eq!(qualified.hits.len(), 1);
    assert_eq!(code.chunks()[qualified.hits[0].row].symbol, "config::load");
    let unqualified = prepared
        .query("load", None, options(SearchStrategy::Symbol))
        .unwrap();
    assert_eq!(unqualified.hits.len(), 2);
    assert!(
        unqualified
            .hits
            .iter()
            .all(|h| code.chunks()[h.row].language == "rust")
    );
    assert!(unqualified.hits[0].row < unqualified.hits[1].row);
    assert!(
        prepared
            .query("Config::load", None, options(SearchStrategy::Symbol))
            .unwrap()
            .hits
            .is_empty()
    );
    assert!(
        !prepared
            .query("def load", None, options(SearchStrategy::Lexical))
            .unwrap()
            .hits
            .is_empty()
    );
}

#[test]
fn overlapping_ranges_are_suppressed_before_top_k_and_can_be_retained_explicitly() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("lib.rs"),
        "fn first() {} fn second() {}\nfn third() {}\n",
    )
    .unwrap();
    let mut provider = HashProvider::new(8).unwrap();
    let code = build_repository_with_provider(
        dir.path(),
        &BuildOptions::for_embedding(provider.config()).unwrap(),
        None,
        &mut provider,
        2,
    )
    .unwrap()
    .index;
    let index = SnapshotTextIndex::new(&code);
    let prepared = index.prepare(&CodeFilter::default(), None).unwrap();
    let mut opts = options(SearchStrategy::Lexical);
    opts.top_k = 2;
    let hits = prepared.query("fn", None, opts).unwrap().hits;
    assert_eq!(hits.iter().map(|h| h.row).collect::<Vec<_>>(), [0, 2]);
    assert_eq!(hits[0].suppressed_overlaps.len(), 1);
    let suppressed = &hits[0].suppressed_overlaps[0];
    assert_eq!(suppressed.symbol, "second");
    assert_eq!((suppressed.start_line, suppressed.end_line), (1, 1));
    assert_eq!(suppressed.provenance[0].source, RetrievalSource::Lexical);
    opts.deduplicate = false;
    let hits = prepared.query("fn", None, opts).unwrap().hits;
    assert_eq!(hits.iter().map(|h| h.row).collect::<Vec<_>>(), [0, 1]);
    assert!(hits.iter().all(|h| h.suppressed_overlaps.is_empty()));
}

#[test]
fn cli_host_strategies_work_with_an_unavailable_model_and_report_provenance() {
    let (dir, code, _) = fixture();
    let path = dir.path().join("index.arrow");
    code.save_ipc(&path).unwrap();
    let bin = env!("CARGO_BIN_EXE_cubeql-code");
    for strategy in ["lexical", "symbol"] {
        let output = Command::new(bin)
            .args([
                "search",
                path.to_str().unwrap(),
                "target",
                "--strategy",
                strategy,
                "--model-dir",
                "/unavailable-model-files",
                "--include-source",
                "--top-k",
                "1",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["format_version"], 2);
        assert_eq!(report["strategy"], strategy);
        assert_eq!(report["metric"], strategy);
        assert_eq!(report["backend"], "Host");
        assert_eq!(report["hits"][0]["symbol"], "target");
        assert_eq!(report["hits"][0]["source"], TARGET);
        assert_eq!(report["hits"][0]["provenance"][0]["source"], strategy);
        assert_eq!(report["timings"]["embedding_ms"], 0.0);
        assert_eq!(report["timings"]["model_load_ms"], 0.0);
        assert_eq!(report["timings"]["upload_ms"], 0.0);
        assert_eq!(report["timings"]["scoring_topk_readback_ms"], 0.0);
    }
    let output = Command::new(bin)
        .args([
            "search",
            path.to_str().unwrap(),
            "target",
            "--strategy",
            "hybrid",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("does not implement"));
}

#[test]
fn current_evaluation_reuses_baseline_embeddings_and_preserves_all_six_strategies() {
    use cubeql_code::evaluation::{
        EvaluationOptions, QueryCase, QueryCategory, QuerySet, RelevantChunk,
    };
    use cubeql_embeddings::{EmbeddingConfig, EmbeddingError};
    struct CountQueries {
        inner: PrecomputedProvider,
        calls: usize,
    }
    impl EmbeddingProvider for CountQueries {
        fn config(&self) -> &EmbeddingConfig {
            self.inner.config()
        }
        fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            panic!("evaluation must not encode documents")
        }
        fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            self.calls += 1;
            self.inner.embed_query(text)
        }
    }
    let (_dir, code, provider) = fixture();
    let mut provider = CountQueries {
        inner: provider,
        calls: 0,
    };
    let dataset = QuerySet {
        version: 1,
        name: "controlled-complementarity".into(),
        queries: vec![QueryCase {
            id: "identifier".into(),
            category: QueryCategory::Identifier,
            text: "target".into(),
            relevant: vec![RelevantChunk {
                path: "lib.rs".into(),
                symbol: "target".into(),
            }],
            filter: CodeFilter::default(),
        }],
    };
    let options = EvaluationOptions {
        candidate_k: 10,
        repeats: 2,
        compare_fp16: true,
    };
    let report = cubeql_code::hybrid_evaluation::evaluate(
        code.clone(),
        &cubecl::Device::default(),
        &mut provider,
        &dataset,
        options,
    )
    .unwrap();
    assert_eq!(provider.calls, 1);
    assert_eq!(report.format_version, 2);
    assert_eq!(report.variants.len(), 2);
    for variant in &report.variants {
        assert_eq!(variant.overall.len(), 6);
        assert_eq!(variant.by_category[&QueryCategory::Identifier].len(), 6);
        let query = &variant.queries[0];
        // The v1 baseline must stay separate: flattening another hybrid evaluator here
        // would serialize duplicate lexical/symbol/hybrid keys and run those paths twice.
        let baseline_json = serde_json::to_value(&query.baseline).unwrap();
        for name in ["lexical", "symbol", "hybrid"] {
            assert!(baseline_json.get(name).is_none());
        }
        assert_eq!(query.baseline.semantic.metrics.first_relevant_rank, Some(3));
        assert_eq!(query.hybrid.metrics.first_relevant_rank, Some(1));
        assert_eq!(query.hybrid.retrieval.samples, 2);
        assert_eq!(query.hybrid.top_hits[0].provenance.len(), 3);
        assert_eq!(query.hybrid.top_hits[0].code.symbol, "target");
    }
    assert_eq!(
        report.variants[0].snapshot.corpus_fingerprint,
        report.variants[1].snapshot.corpus_fingerprint
    );
    assert_eq!(
        report.variants[0].snapshot.vector_storage_bytes,
        report.variants[1].snapshot.vector_storage_bytes * 2
    );
    let json = serde_json::to_value(&report).unwrap();
    assert!(json["variants"][0]["queries"][0]["semantic"].is_object());
    assert!(json["variants"][0]["queries"][0]["hybrid"].is_object());
    assert!(
        cubeql_code::hybrid_evaluation::evaluate(
            code,
            &cubecl::Device::default(),
            &mut provider,
            &dataset,
            EvaluationOptions {
                repeats: 0,
                ..options
            }
        )
        .is_err()
    );
    assert_eq!(provider.calls, 1);
}
