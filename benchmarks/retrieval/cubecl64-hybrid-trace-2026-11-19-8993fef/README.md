# Original64 observed Hybrid-stage evidence

The instrumented Hybrid run reproduced all 384 prior query/strategy metrics, Top10 displays and complete final rankings exactly. The new trace locates each required symbol relative to the actual filtered branch windows, RRF fusion, overlap deduplication and final truncation. Ranking defaults and budgets were unchanged.

Of the ten questions with an anchor absent from the final Top100 window, seven involve absence from all bounded branch windows and four involve final truncation; one question has both kinds of gaps. No overlap suppression occurred in these 64 exported traces. This is a measured development observation, not evidence that overlap handling is unnecessary in other corpora.

## Fixed condition

- Date: 2026-11-19. Exporter source revision: `8993fef469cd31125310d7f4698e4e3e9734fdac`; binary/source hashes recorded separately. Build revision is not independently attested.
- Same preserved CubeCL `5037c0639ce3b1205392bdbe43dcf25e2ea09e29` Arrow snapshot: 1,127 files and 27,289 chunks. Snapshot SHA-256 `b162e27c1155c4c3983bffd226e77f5e3a75610e47eb76ce447e16338d7a40bb` was checked before and after execution.
- Same original 64 AI-authored positive symbol-label queries and Rust filters. Query SHA-256 `8c2d3eb7da1560a98f4d9ee844cfa842f53c3f9a1109affa71fed9cfad49c091`; 89 per-query anchor judgments across four categories with 16 questions each. Independent human review and a separate multi-repository validation set remain outstanding.
- CodeRankEmbed256 FP32 CPU, candidate_k=100, repeats=3, no FP16 comparison; exact source/model/provider/runner identities matched the old frozen recipe except for the intended new executable. Document vectors were reused. The local encoder encoded queries live; this is not query-vector replay.
- One evaluate-only command with `--trace-hybrid --format json`, a 3,600-second deadline and a predeclared 128 MiB captured-output bound. Actual stdout was 25,783,499 bytes. CLI wall time was 183,811 ms.
- Trace capture happens on the final Hybrid repeat using its actual branch arrays and deduplication decisions, with no additional branch searches or query-embedding calls. The native deduplication timer and that warm-repeat sample may include capture work; CLI time includes startup, retrieval and export. This is not a controlled latency comparison with the prior uninstrumented run.
- No paid API benchmark arm, coding-agent task evaluation, repository re-indexing or document encoding was performed. Original experiments, source-preparation receipts and this new create-new run are preserved separately.

## Observed label positions

Hybrid Recall@10 remains 0.729167; final-window Recall@100 remains 0.878906; independently recomputed MRR@100 remains 0.545637. The trace adds stage evidence rather than a quality improvement.

| First observed stage | Anchor judgments | Questions containing this stage |
| --- | ---: | ---: |
| not_in_filtered_branch_windows | 11 | 7 |
| suppressed_by_overlap | 0 | 0 |
| truncated_after_deduplication | 5 | 4 |
| retained_below_top10 | 15 | 15 |
| retained_top10 | 58 | 50 |

Anchor-stage counts partition the 89 judgments, including labels repeated across questions. Question counts overlap when a question requires several symbols; they must not be added as disjoint question totals. In particular, `v2-cross-layout-and-contiguity` has one anchor outside all branch windows and another lost at final truncation.

| Category | Questions | Judgments | Absent from branches | Suppressed | Truncated | Retained ranks 11–100 | Retained within ten |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| behavioral | 16 | 18 | 1 | 0 | 0 | 2 | 15 |
| conceptual | 16 | 16 | 1 | 0 | 1 | 5 | 9 |
| cross_file | 16 | 39 | 9 | 0 | 4 | 8 | 18 |
| identifier | 16 | 16 | 0 | 0 | 0 | 0 | 16 |

## Concrete stage evidence

- `concept-shader-language-selection`: `<AutoCompiler as WgpuCompiler>::init` enters the lexical branch at rank 94, reaches fused/deduplicated rank 183, and is cut by the final 100-result window.
- `v2-cross-layout-and-contiguity`: `is_contiguous` is absent from all three filtered branch windows; `is_contiguous_pitched` enters lexical rank 79, reaches fused/deduplicated rank 171, and is truncated. A single query-level missing label would hide this distinction.
- `v2-behavior-biased-first-timing`: `SampleSet::reliable` appears only in the semantic branch at rank 4, then at fused/deduplicated/final rank 25. `DISCARD_THRESHOLD` appears at semantic rank 2 and lexical rank 13, remaining final rank 2. The required second location is lost from ten while first-hit MRR stays unchanged.

These positions explain the observed pipeline transitions under this fixed condition. They do not establish that changing weights would improve held-out quality. The eight-test offline diagnostic suite includes all five stages, observed suppression with its kept witness beyond the final window, and a float64-order/float32-score tie case. Four Rust integration tests also exercise actual suppression separately from this real dataset, which had zero suppression events.

## Limits and next evidence

Every exported branch is already filtered and bounded to 100 candidates. Absence there does not prove absence from the repository, from an unfiltered list, or below a branch cutoff. Source scoring correctness and unbounded recall are not established by replaying the trace. The trace validates actual RRF/IoU transitions and final truncation; suppression is a geometric policy decision, not a relevance judgment.

The observed truncation cases and rank-4-to-25 example provide concrete hypotheses for later fusion/routing experiments. Before tuning a default policy, freeze an independently reviewed multi-repository validation set. A branch-budget or structural-context experiment should separately declare changed work and preserve these original labels/results. This batch makes no held-out 80% target, peer-superiority, speedup or coding-agent success claim.

## Verify without inference

```sh
python3 -B benchmarks/retrieval/cubecl64-hybrid-trace-2026-11-19-8993fef/verify.py
```

The standalone verifier reads evidence without extracting files or executing archived producers. It checks exact membership/hashes, original labels and frozen input receipts, unchanged prior complete rankings/metrics, full-window recall/MRR, the float64 RRF union order with float32 scores, greedy inclusive-range IoU witnesses, final truncation, per-label first-loss stages, category counts and summary projections.

`trace-summary.json` selects the overall/category summaries, all ten missing-final-window queries and the retained regression. Complete 64-query diagnostics and native traces remain in the 30-member evidence archive, alongside the prior full-ranking report, raw execution/stdio, original recipe and snapshot receipt, before/after identities, producer sources and an after-run dependency lock. The large Arrow index, model weights, source checkout and caches are excluded; their relevant identities remain recorded. Byte hashes are not a hermetic build or training-provenance attestation.

For new traces use [the observed-stage consumer](../../../docs/repository-hybrid-trace.md). See the [prior final-window report](../cubecl64-candidates-2026-11-12-6104689/README.md) for the original 43/11/10 question partition and complete historical evidence. Existing version-2/3 reports and tools remain available.
