use std::{fs, path::PathBuf};

use cubeql_code::{
    BuildOptions, CodeIndex, build_repository_with_provider,
    evaluation::{self, EvaluationOptions, QuerySet},
    hybrid_evaluation,
};
use cubeql_embeddings::{EmbeddingConfig, EmbeddingError, EmbeddingProvider, HashProvider};

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

fn assert_candidates<T: serde::Serialize>(strategy: &T, limit: usize) {
    let value = serde_json::to_value(strategy).unwrap();
    let candidates = value["candidate_ranking"].as_array().unwrap();
    let top_hits = value["top_hits"].as_array().unwrap();
    assert!(candidates.len() <= limit);
    assert_eq!(
        candidates
            .iter()
            .map(|hit| hit["rank"].as_u64().unwrap() as usize)
            .collect::<Vec<_>>(),
        (1..=candidates.len()).collect::<Vec<_>>()
    );
    for (candidate, top_hit) in candidates.iter().zip(top_hits) {
        for field in ["row", "path", "symbol", "score"] {
            assert_eq!(candidate[field], top_hit[field]);
        }
    }
    assert!(candidates.iter().all(|candidate| {
        candidate.get("source").is_none()
            && candidate.get("provenance").is_none()
            && candidate["path"].is_string()
            && candidate["symbol"].is_string()
            && candidate["score"].is_number()
    }));
}

fn assert_metrics_cover_candidate_window<T: serde::Serialize>(
    strategy: &T,
    relevant: &std::collections::BTreeSet<usize>,
) {
    let value = serde_json::to_value(strategy).unwrap();
    let rows = value["candidate_ranking"]
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| candidate["row"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let expected = evaluation::ranking_metrics(&rows, relevant).unwrap();
    assert_eq!(
        value["metrics"]["recall_at_5"].as_f64().unwrap(),
        expected.recall_at_5
    );
    assert_eq!(
        value["metrics"]["recall_at_10"].as_f64().unwrap(),
        expected.recall_at_10
    );
    assert_eq!(
        value["metrics"]["reciprocal_rank"].as_f64().unwrap(),
        expected.reciprocal_rank
    );
}

#[test]
fn candidate_export_is_opt_in_complete_and_keeps_metrics_and_default_json() {
    let (code, _provider) = fixture();
    let options = EvaluationOptions {
        candidate_k: 24,
        repeats: 1,
        compare_fp16: true,
    };
    let dataset = dataset();
    let mut default_provider = HashProvider::new(32).unwrap();
    let default_report = hybrid_evaluation::evaluate(
        code.clone(),
        &cubecl::Device::default(),
        &mut default_provider,
        &dataset,
        options,
    )
    .unwrap();
    assert_eq!(default_report.format_version, 2);
    let default_json = serde_json::to_value(&default_report).unwrap();
    for variant in default_json["variants"].as_array().unwrap() {
        for query in variant["queries"].as_array().unwrap() {
            for strategy in [
                "semantic",
                "literal",
                "experimental_rrf",
                "lexical",
                "symbol",
                "hybrid",
            ] {
                assert!(query[strategy].get("candidate_ranking").is_none());
            }
        }
    }

    let mut counter = QueryCounter {
        inner: HashProvider::new(32).unwrap(),
        queries: 0,
    };
    let exported = hybrid_evaluation::evaluate_with_candidates(
        code.clone(),
        &cubecl::Device::default(),
        &mut counter,
        &dataset,
        options,
    )
    .unwrap();
    assert_eq!(exported.format_version, 3);
    assert_eq!(counter.queries, dataset.queries.len());
    assert_eq!(exported.variants.len(), 2);
    let relevant = dataset.resolve(&code).unwrap();
    for (old_variant, variant) in default_report.variants.iter().zip(&exported.variants) {
        for (query_index, (old_query, query)) in
            old_variant.queries.iter().zip(&variant.queries).enumerate()
        {
            for (old_strategy, strategy) in [
                (&old_query.baseline.semantic, &query.baseline.semantic),
                (&old_query.baseline.literal, &query.baseline.literal),
                (
                    &old_query.baseline.experimental_rrf,
                    &query.baseline.experimental_rrf,
                ),
            ] {
                assert_eq!(
                    old_strategy.metrics.recall_at_5,
                    strategy.metrics.recall_at_5
                );
                assert_eq!(
                    old_strategy.metrics.recall_at_10,
                    strategy.metrics.recall_at_10
                );
                assert_eq!(
                    old_strategy.metrics.reciprocal_rank,
                    strategy.metrics.reciprocal_rank
                );
            }
            for (old_strategy, strategy) in [
                (&old_query.lexical, &query.lexical),
                (&old_query.symbol, &query.symbol),
                (&old_query.hybrid, &query.hybrid),
            ] {
                assert_eq!(
                    old_strategy.metrics.recall_at_5,
                    strategy.metrics.recall_at_5
                );
                assert_eq!(
                    old_strategy.metrics.recall_at_10,
                    strategy.metrics.recall_at_10
                );
                assert_eq!(
                    old_strategy.metrics.reciprocal_rank,
                    strategy.metrics.reciprocal_rank
                );
            }
            let baseline = &query.baseline;
            assert_candidates(&baseline.semantic, options.candidate_k);
            assert_candidates(&baseline.literal, options.candidate_k);
            assert_candidates(&baseline.experimental_rrf, options.candidate_k);
            assert_candidates(&query.lexical, options.candidate_k);
            assert_candidates(&query.symbol, options.candidate_k);
            assert_candidates(&query.hybrid, options.candidate_k);
            let relevant = &relevant[query_index];
            assert_metrics_cover_candidate_window(&baseline.semantic, relevant);
            assert_metrics_cover_candidate_window(&baseline.literal, relevant);
            assert_metrics_cover_candidate_window(&baseline.experimental_rrf, relevant);
            assert_metrics_cover_candidate_window(&query.lexical, relevant);
            assert_metrics_cover_candidate_window(&query.symbol, relevant);
            assert_metrics_cover_candidate_window(&query.hybrid, relevant);
        }
    }
}

#[test]
fn baseline_export_includes_full_windows_and_empty_literal_is_valid() {
    let (code, mut provider) = fixture();
    let mut dataset = dataset();
    dataset.queries.truncate(1);
    dataset.queries[0].text = "no literal match anywhere in this snapshot".into();
    let options = EvaluationOptions {
        candidate_k: 24,
        repeats: 1,
        compare_fp16: false,
    };
    let default = evaluation::evaluate(
        code.clone(),
        &cubecl::Device::default(),
        &mut provider,
        &dataset,
        options,
    )
    .unwrap();
    let exported = evaluation::evaluate_with_candidates(
        code,
        &cubecl::Device::default(),
        &mut provider,
        &dataset,
        options,
    )
    .unwrap();
    assert_eq!(default.format_version, 1);
    assert_eq!(exported.format_version, 2);
    let default_json = serde_json::to_value(&default).unwrap();
    for strategy in ["semantic", "literal", "experimental_rrf"] {
        assert!(
            default_json["variants"][0]["queries"][0][strategy]
                .get("candidate_ranking")
                .is_none()
        );
    }
    let old = &default.variants[0].queries[0];
    let new = &exported.variants[0].queries[0];
    assert_eq!(
        old.literal.metrics.reciprocal_rank,
        new.literal.metrics.reciprocal_rank
    );
    assert!(new.literal.candidate_ranking.as_ref().unwrap().is_empty());
    assert!(new.semantic.candidate_ranking.as_ref().unwrap().len() > 10);
    assert_eq!(new.semantic.top_hits.len(), 10);
    assert_eq!(
        new.semantic.candidate_ranking.as_ref().unwrap()[0].row,
        new.semantic.top_hits[0].row
    );
    assert_eq!(new.semantic.candidate_ranking.as_ref().unwrap()[0].rank, 1);
}

#[test]
fn hybrid_candidate_export_accepts_empty_lexical_and_symbol_lists() {
    let (code, mut provider) = fixture();
    let mut dataset = dataset();
    dataset.queries.truncate(1);
    dataset.queries[0].text = "########################".into();
    let report = hybrid_evaluation::evaluate_with_candidates(
        code,
        &cubecl::Device::default(),
        &mut provider,
        &dataset,
        EvaluationOptions {
            candidate_k: 24,
            repeats: 1,
            compare_fp16: false,
        },
    )
    .unwrap();
    let query = &report.variants[0].queries[0];
    assert!(query.lexical.candidate_ranking.as_ref().unwrap().is_empty());
    assert!(query.symbol.candidate_ranking.as_ref().unwrap().is_empty());
    assert!(query.lexical.top_hits.is_empty());
    assert!(query.symbol.top_hits.is_empty());
}

#[test]
fn cli_requires_json_and_keeps_wrapper_version_one() {
    let dir = tempfile::tempdir().unwrap();
    let index = dir.path().join("index.arrow");
    let mut provider = HashProvider::new(32).unwrap();
    let options = BuildOptions::for_embedding(provider.config()).unwrap();
    let code = build_repository_with_provider(corpus(), &options, None, &mut provider, 4)
        .unwrap()
        .index;
    code.write_ipc(fs::File::create(&index).unwrap()).unwrap();
    let queries = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/retrieval-benchmark/queries.json");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
        .args([
            "evaluate",
            index.to_str().unwrap(),
            queries.to_str().unwrap(),
            "--export-candidates",
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
    assert_eq!(report["evaluation"]["format_version"], 3);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
        .args([
            "evaluate",
            index.to_str().unwrap(),
            queries.to_str().unwrap(),
            "--export-candidates",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}
