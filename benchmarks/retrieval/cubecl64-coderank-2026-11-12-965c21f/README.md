# Original 64-query CubeCL CodeRank CPU development run

The fixed local CodeRankEmbed condition completed once on the original 64-query development set. Hybrid Recall@10 is **0.729167**; semantic Recall@10 is **0.559896**. These are positive symbol-label observations, not independently reviewed validation or coding-agent success. Existing ranking defaults are unchanged.

## Frozen condition

- Run date: 2026-11-12. Recorded checkout at launch: `965c21f9f42d012a3a98beb82e5f99a0faa49ea7`; the existing executable is identified separately by SHA-256. Its build revision is not independently attested.
- Corpus: CubeCL `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`; 1,127 indexed files, 27,289 chunks, 8,075,854 source-text bytes. All 1,175 input files were inventoried before and after execution.
- Original query bytes: SHA-256 `8c2d3eb7da1560a98f4d9ee844cfa842f53c3f9a1109affa71fed9cfad49c091`. Four categories with 16 questions each; positive `(path, symbol)` anchors, AI-authored, independent human review pending. Labels and filters were preserved.
- Model: `nomic-ai/CodeRankEmbed`, revision `3c4b60807d71f79b43f3c4363786d9493691f8b1`, FP32 CPU, CLS pooling, L2 normalization, 768 dimensions, max_tokens=256; required query prefix retained. Exact model hashes and provider implementation/version identity are in `experiment.json` and the archived recipe.
- Candidate budget: 100 per native source/final window; three warm retrieval repeats; one FP32 variant. The same encoded query vectors are reused across native strategies. Warm repeats are timing observations, not three quality samples.
- Bounds fixed before execution: 3,600 seconds per native command, provider startup 120,000 ms, provider request 30,000 ms. Both stages succeeded. No paid model API or coding-agent evaluation was run.
- Source preparation materialized 20 internal license symlinks to their exact target bytes in the generated archive copy. The original checkout and source code were untouched. The initial strict-inventory preparation failure and per-link conversions remain archived.

The source-chunk fingerprint `3ad4afd5220b95c3987c463f6ebe009c1c5adab2ccac8bc256da69bf8d7176cb` matches the [historical MiniLM run](../cubecl-2026-10-29-77b2d6af8514/REPORT.md). This is a new model/product condition with a different executable and provider; it does not isolate a causal model or engine improvement and does not replace that historical report.

## Native quality

| Strategy | Queries | Recall@5 | Recall@10 | MRR@100 |
| --- | ---: | ---: | ---: | ---: |
| semantic | 64 | 0.473958 | 0.559896 | 0.385504 |
| literal | 64 | 0.250000 | 0.250000 | 0.250000 |
| experimental_rrf | 64 | 0.520833 | 0.591146 | 0.470404 |
| lexical | 64 | 0.419271 | 0.549479 | 0.383519 |
| symbol | 64 | 0.250000 | 0.250000 | 0.250465 |
| hybrid | 64 | 0.627604 | 0.729167 | 0.545637 |

| Category | Queries | Semantic Recall@10 | Hybrid Recall@10 |
| --- | ---: | ---: | ---: |
| behavioral | 16 | 0.625000 | 0.843750 |
| conceptual | 16 | 0.375000 | 0.562500 |
| cross_file | 16 | 0.364583 | 0.510417 |
| identifier | 16 | 0.875000 | 1.000000 |

## Paired observations

Compared with semantic Recall@10, hybrid recovers 10 zero-recall queries, improves three queries with partial recall, reduces recall on one query, loses none completely, shares 14 zero-recall misses and ties on 36 positive-recall queries. These counts describe this frozen development set; they are not significance tests.

The retained regression is `v2-behavior-biased-first-timing`: semantic Recall@10 is 1.0 and hybrid is 0.5, while the first relevant rank stays 2. A first-hit MRR alone would hide this loss of a required anchor. Twenty-one queries have incomplete hybrid anchor coverage at ten. An absent displayed hit does not prove absence from the internal candidate pool.

The next evidence requirement is an independently reviewed multi-repository validation set. The conceptual/cross-file misses and the retained regression justify inspecting source evidence and candidate exports before selecting routing or structural-context experiments. They do not establish a failure cause or authorize a new ranking policy. RQ4/RQ5 ranking changes remain deferred; the RQ2 validation target is not claimed.

## Timing boundaries

Index CLI wall time was 2,053,704 ms (34.23 minutes), including provider startup, source processing, encoding and serialization. Evaluate CLI wall time was 46,237 ms. Within evaluation, provider startup was 5,272.722 ms, snapshot read was 648.116 ms and the native query-embedding total was 3,049.388 ms. Raw reports retain first-invocation, filter, materialization and warm retrieval timers.

These boundaries differ from resident MCP request latency and from native Semble payload work. No speedup or equal-work comparison is claimed. The runtime records macOS 15.5 arm64, eight logical CPUs and Python 3.14.6; no controlled hardware comparison or measured peak-memory result is asserted. Arrow allocation fields are native byte-accounting observations, not whole-process peak RSS.

## Verify without inference

```sh
python3 -B benchmarks/retrieval/cubecl64-coderank-2026-11-12-965c21f/verify.py
```

The standalone verifier checks exact publication/archive membership and SHA-256 values, the recipe fingerprint, original label bytes, successful execution/condition records, producer hashes, displayed-hit Recall@5/10 and overall/category means. MRR beyond the ten displayed hits is validated against the reported first-relevant rank; a complete Top100 ranking is not exported.

The full diagnostic was generated by the [native symbol-report consumer](../../../tools/repository_native_diagnostics.py) from the saved native report, with complete query/category denominators, and is preserved at `diagnostics/diagnostics.json` inside the evidence archive. The top-level `diagnostics.json` selects its overall/category summaries, paired counts, regression and incomplete-coverage IDs for review; the verifier checks this projection against the full archived output. The archive includes its exact producer source and canonical input, plus native index/evaluate stdout/stderr, execution records, frozen recipe/source inventory, input profiles, source-preparation receipts and a dependency lock observed after execution. Checksums identify bytes; they do not attest training provenance or a hermetic runtime.

The 99,489,234-byte Arrow snapshot, model weights, source checkout and runtime caches are excluded from this publication. The original local snapshot remains preserved, with its hash in the condition and experiment records. Archived paths are historical local evidence, not commands to run unchanged.

To execute a new condition, use the [baseline workflow](../../../docs/repository-baseline-workflow.md): point a new configuration at a pinned local source/model/binary, retain these original labels and fixed options, prepare a new recipe and run into a new directory with `--execute`. Preparation rechecks identities. Do not overwrite this completed run or relabel queries to improve its score.
