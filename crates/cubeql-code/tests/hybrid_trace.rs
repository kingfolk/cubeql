use std::{fs, path::PathBuf};

use cubeql_code::{
    BuildOptions, CodeIndex, CodeSearchIndex, build_repository_with_provider,
    evaluation::{EvaluationOptions, QuerySet},
    hybrid_evaluation,
    reporting::CodeFilter,
    retrieval::{RetrievalOptions, SearchStrategy, SnapshotTextIndex},
};
use cubeql_embeddings::{
    EmbeddingConfig, EmbeddingError, EmbeddingProvider, HashProvider, embed_query,
};

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/retrieval-benchmark/corpus")
}

fn dataset() -> QuerySet {
    serde_json::from_str(include_str!("fixtures/retrieval-benchmark/queries.json")).unwrap()
}

fn fixture() -> (CodeIndex, HashProvider) {
    let mut provider = HashProvider::new(32).unwrap();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let index = build_repository_with_provider(corpus(), &options, None, &mut provider, 4)
        .unwrap()
        .index;
    (index, provider)
}

struct QueryCounter {
    inner: HashProvider,
    queries: usize,
}

impl EmbeddingProvider for QueryCounter {
    fn config(&self) -> &EmbeddingConfig {
        self.inner.config()
    }
    fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        panic!("evaluation must not encode documents")
    }
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.queries += 1;
        self.inner.embed_query(text)
    }
}

#[test]
fn trace_keeps_overlap_suppression_witnesses_beyond_final_window() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("nested.rs"),
        "mod outer { pub fn unrelated_example() { let value = 1; } }\n",
    )
    .unwrap();
    fs::write(dir.path().join("target.rs"), "pub fn needle_target() {}\n").unwrap();
    fs::write(dir.path().join("other.rs"), "pub fn needle_other() {}\n").unwrap();
    let mut provider = HashProvider::new(8).unwrap();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let code = build_repository_with_provider(dir.path(), &options, None, &mut provider, 2)
        .unwrap()
        .index;
    assert!(code.chunks().len() >= 3);
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code).unwrap();
    let text_index = SnapshotTextIndex::new(search.code());
    text_index.warm_up(SearchStrategy::Hybrid);
    let prepared = text_index
        .prepare(&CodeFilter::default(), Some(&search))
        .unwrap();
    let query = embed_query(&mut provider, "needle_target").unwrap();
    let options = RetrievalOptions {
        strategy: SearchStrategy::Hybrid,
        candidate_k: 10,
        top_k: 1,
        deduplicate: true,
    };
    let traced = prepared
        .query_with_trace("needle_target", Some(&query), options)
        .unwrap();
    assert_eq!(
        search.code().chunks()[traced.trace.final_rows[0]].symbol,
        "needle_target"
    );
    assert!(
        traced.trace.suppressions.iter().any(|suppression| {
            traced
                .trace
                .fused
                .iter()
                .any(|candidate| candidate.row == suppression.row)
                && traced
                    .trace
                    .deduplicated
                    .iter()
                    .any(|candidate| candidate.row == suppression.kept_row)
                && !traced.trace.final_rows.contains(&suppression.kept_row)
                && !traced.trace.final_rows.contains(&suppression.row)
        }),
        "expected the complete kept-list witness outside final top_k"
    );

    let mut keep_overlaps = options;
    keep_overlaps.deduplicate = false;
    let kept = prepared
        .query_with_trace("needle_target", Some(&query), keep_overlaps)
        .unwrap();
    assert!(kept.trace.suppressions.is_empty());
    assert!(kept.trace.deduplicated.len() >= traced.trace.deduplicated.len());
    assert!(
        kept.trace
            .deduplicated
            .iter()
            .any(|c| c.path == "nested.rs")
    );
    assert!(kept.trace.deduplicated.iter().any(|c| c.path == "other.rs"));
    assert_eq!(
        kept.trace
            .deduplicated
            .iter()
            .filter(|candidate| candidate.path == "nested.rs")
            .count(),
        2
    );
}

#[test]
fn trace_is_captured_from_one_hybrid_query_and_preserves_rank_contract() {
    let (code, mut provider) = fixture();
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code).unwrap();
    let index = SnapshotTextIndex::new(search.code());
    index.warm_up(SearchStrategy::Hybrid);
    let prepared = index
        .prepare(&CodeFilter::default(), Some(&search))
        .unwrap();
    let query = embed_query(&mut provider, "configuration reader").unwrap();
    let options = RetrievalOptions {
        strategy: SearchStrategy::Hybrid,
        candidate_k: 24,
        top_k: 2,
        deduplicate: true,
    };
    let ordinary = prepared
        .query("configuration reader", Some(&query), options)
        .unwrap();
    let traced = prepared
        .query_with_trace("configuration reader", Some(&query), options)
        .unwrap();
    assert_eq!(
        traced
            .result
            .hits
            .iter()
            .map(|hit| (
                hit.row,
                hit.score,
                serde_json::to_value(&hit.provenance).unwrap()
            ))
            .collect::<Vec<_>>(),
        ordinary
            .hits
            .iter()
            .map(|hit| (
                hit.row,
                hit.score,
                serde_json::to_value(&hit.provenance).unwrap()
            ))
            .collect::<Vec<_>>(),
    );
    let trace = traced.trace;
    assert_eq!(trace.format_version, 1);
    assert_eq!(
        (trace.candidate_k, trace.top_k, trace.rrf_rank_constant),
        (24, 2, 60)
    );
    assert_eq!(
        trace
            .sources
            .iter()
            .map(|source| source.source)
            .collect::<Vec<_>>(),
        [
            cubeql_code::retrieval::RetrievalSource::Semantic,
            cubeql_code::retrieval::RetrievalSource::Lexical,
            cubeql_code::retrieval::RetrievalSource::Symbol,
        ]
    );
    assert!(
        trace
            .sources
            .iter()
            .all(|source| source.candidates.len() <= 24)
    );
    assert!(trace.fused.len() > options.top_k);
    assert!(trace.fused.len() <= 3 * options.candidate_k);
    assert!(trace.fused.windows(2).all(|pair| {
        pair[0].score > pair[1].score
            || (pair[0].score == pair[1].score && pair[0].row < pair[1].row)
    }));
    assert_eq!(
        trace.final_rows,
        ordinary.hits.iter().map(|hit| hit.row).collect::<Vec<_>>()
    );
    assert!(trace.fused.iter().enumerate().all(|(i, c)| c.rank == i + 1));
    assert!(
        trace
            .deduplicated
            .iter()
            .enumerate()
            .all(|(i, c)| c.rank == i + 1)
    );
}

#[test]
fn empty_branches_zero_top_k_and_evaluation_schema_are_explicit() {
    let (code, mut provider) = fixture();
    let search = CodeSearchIndex::new(&cubecl::Device::default(), code.clone()).unwrap();
    let index = SnapshotTextIndex::new(search.code());
    let filter = CodeFilter {
        path_prefix: Some("missing/".into()),
        ..Default::default()
    };
    let prepared = index.prepare(&filter, Some(&search)).unwrap();
    let query = embed_query(&mut provider, "################").unwrap();
    let mut options = RetrievalOptions {
        strategy: SearchStrategy::Hybrid,
        candidate_k: 10,
        top_k: 0,
        deduplicate: true,
    };
    let traced = prepared
        .query_with_trace("################", Some(&query), options)
        .unwrap();
    assert!(traced.result.hits.is_empty());
    assert_eq!(traced.trace.sources.len(), 3);
    assert!(
        traced
            .trace
            .sources
            .iter()
            .all(|source| source.candidates.is_empty())
    );
    assert!(traced.trace.fused.is_empty() && traced.trace.deduplicated.is_empty());
    assert!(traced.trace.final_rows.is_empty());
    options.top_k = 10;
    let traced = prepared
        .query_with_trace("################", Some(&query), options)
        .unwrap();
    assert!(
        traced
            .trace
            .sources
            .iter()
            .all(|source| source.candidates.is_empty())
    );

    let dataset = dataset();
    let options = EvaluationOptions {
        candidate_k: 24,
        repeats: 1,
        compare_fp16: true,
    };
    let mut default_provider = HashProvider::new(32).unwrap();
    let default = hybrid_evaluation::evaluate(
        code.clone(),
        &cubecl::Device::default(),
        &mut default_provider,
        &dataset,
        options,
    )
    .unwrap();
    let mut trace_provider = QueryCounter {
        inner: HashProvider::new(32).unwrap(),
        queries: 0,
    };
    let traced = hybrid_evaluation::evaluate_with_trace(
        code,
        &cubecl::Device::default(),
        &mut trace_provider,
        &dataset,
        options,
    )
    .unwrap();
    assert_eq!(trace_provider.queries, dataset.queries.len());
    assert_eq!((default.format_version, traced.format_version), (2, 4));
    let json = serde_json::to_value(&traced).unwrap();
    for variant in json["variants"].as_array().unwrap() {
        for query in variant["queries"].as_array().unwrap() {
            assert!(query.get("hybrid_trace").is_some());
            for strategy in [
                "semantic",
                "literal",
                "experimental_rrf",
                "lexical",
                "symbol",
                "hybrid",
            ] {
                assert!(query[strategy].get("candidate_ranking").is_some());
            }
        }
    }
    let default_json = serde_json::to_value(default).unwrap();
    assert!(
        default_json["variants"][0]["queries"][0]
            .get("hybrid_trace")
            .is_none()
    );
    assert!(
        default_json["variants"][0]["queries"][0]["hybrid"]
            .get("candidate_ranking")
            .is_none()
    );
    let mut candidates_provider = HashProvider::new(32).unwrap();
    let candidates = hybrid_evaluation::evaluate_with_candidates(
        fixture().0,
        &cubecl::Device::default(),
        &mut candidates_provider,
        &dataset,
        options,
    )
    .unwrap();
    assert_eq!(candidates.format_version, 3);
    let candidates_json = serde_json::to_value(candidates).unwrap();
    assert!(
        candidates_json["variants"][0]["queries"][0]["hybrid"]
            .get("candidate_ranking")
            .is_some()
    );
    assert!(
        candidates_json["variants"][0]["queries"][0]
            .get("hybrid_trace")
            .is_none()
    );
}

#[test]
fn cli_trace_implies_candidate_export_and_rejects_text_before_index_read() {
    let dir = tempfile::tempdir().unwrap();
    let index = dir.path().join("index.arrow");
    let (code, _) = fixture();
    code.write_ipc(fs::File::create(&index).unwrap()).unwrap();
    let queries = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/retrieval-benchmark/queries.json");
    let binary = env!("CARGO_BIN_EXE_cubeql-code");
    let rejected = std::process::Command::new(binary)
        .args([
            "evaluate",
            "/does/not/exist",
            queries.to_str().unwrap(),
            "--trace-hybrid",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("--format json"));
    let output = std::process::Command::new(binary)
        .args([
            "evaluate",
            index.to_str().unwrap(),
            queries.to_str().unwrap(),
            "--trace-hybrid",
            "--format",
            "json",
            "--candidate-k",
            "24",
            "--repeats",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["format_version"], 1);
    assert_eq!(report["evaluation"]["format_version"], 4);
    assert!(report["evaluation"]["variants"][0]["queries"][0]["hybrid_trace"].is_object());
    assert!(
        report["evaluation"]["variants"][0]["queries"][0]["hybrid"]["candidate_ranking"].is_array()
    );
}
