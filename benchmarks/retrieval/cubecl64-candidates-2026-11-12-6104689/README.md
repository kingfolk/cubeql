# Complete retained-window evidence on original64

The opt-in candidate export reproduced all 384 original query/strategy metric and ten-hit display cases exactly while exposing the complete final retained rankings. The preserved document index was reused; no repository document re-encoding was performed. Live local query encoding remained enabled.

Hybrid Recall@10 remains **0.729167**. Its newly observable final-window Recall@100 is **0.878906**, with all labeled anchors present somewhere in the final window for 54/64 questions. These are different cutoffs on the same AI-authored development labels; no independently reviewed validation or coding-agent result is claimed.

## Fixed condition and identity

- Run date: 2026-11-12; candidate exporter source revision `6104689`. Existing ranking, fusion, deduplication, chunking and candidate budgets were retained. The executable SHA-256 is recorded separately; the build revision is not independently attested.
- Preserved snapshot: CubeCL `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`, 1,127 indexed files, 27,289 chunks; Arrow SHA-256 `b162e27c1155c4c3983bffd226e77f5e3a75610e47eb76ce447e16338d7a40bb`. Its bytes were checked before and after execution.
- Original 64 queries, filters and positive `(path, symbol)` labels remained unchanged. Query SHA-256: `8c2d3eb7da1560a98f4d9ee844cfa842f53c3f9a1109affa71fed9cfad49c091`. Each of the four categories has 16 questions.
- Same fixed CodeRankEmbed256 FP32 CPU provider, local model artifacts and implementation identity as the [prior completed run](../cubecl64-coderank-2026-11-12-965c21f/README.md). Query vectors were encoded live once per question and reused across the six strategies; this is not query-vector replay.
- Native candidate_k=100, repeats=3, compare_fp16=false. One evaluate-only command with `--export-candidates --format json`; 3,600-second deadline and a predeclared 64 MiB captured-output bound. All source/provider/model/label/runner identities matched the prior frozen inputs; the new exporter executable was the intended change.
- Evaluation CLI wall time: 49,867 ms, including model startup, retrieval, candidate mapping and JSON serialization. Existing retrieval and ten-hit materialization timers exclude candidate export. No controlled speed comparison is made.
- No paid API benchmark arm or coding-agent task evaluation was launched. The original source-preparation license-link materializations and the original completed experiment remain preserved.

## What the final window shows

| Strategy | Recall@10 | Final-window Recall@100 | All required in window | Independently recomputed MRR@100 |
| --- | ---: | ---: | ---: | ---: |
| semantic | 0.559896 | 0.763021 | 0.718750 | 0.385504 |
| literal | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| experimental_rrf | 0.591146 | 0.763021 | 0.718750 | 0.470404 |
| lexical | 0.549479 | 0.884115 | 0.828125 | 0.383519 |
| symbol | 0.250000 | 0.273438 | 0.265625 | 0.250465 |
| hybrid | 0.729167 | 0.878906 | 0.843750 | 0.545637 |

For hybrid, the 64 questions partition into 43 with all anchors in ten, 11 with all anchors in the exported window but at least one below ten, and 10 with an anchor missing from the exported window. A query may have both a missing anchor and another anchor below ten; both lists are retained, and the missing-window class takes precedence.

| Category | Questions | All anchors within ten | All exported, some below ten | Missing from final window | Final-window recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| behavioral | 16 | 13 | 2 | 1 | 0.937500 |
| conceptual | 16 | 9 | 5 | 2 | 0.875000 |
| cross_file | 16 | 5 | 4 | 7 | 0.703125 |
| identifier | 16 | 16 | 0 | 0 | 1.000000 |

### Retained regression

For `v2-behavior-biased-first-timing`, `SampleSet::reliable` is semantic rank 4 and hybrid rank 25. `DISCARD_THRESHOLD` stays at rank 2 for both. Thus hybrid loses one anchor from ten while preserving the first-hit MRR; both anchors remain in the hybrid final window. The new export locates the loss relative to the display cutoff without assigning a causal explanation.

## Interpretation and next evidence

The arrays are the final retained rankings used for metrics, after existing fusion/deduplication. They are not the internal pre-fusion or pre-deduplication candidate unions. An anchor absent here may have existed earlier in a branch, been filtered, or been removed by deduplication. The ten missing-window cases do not prove a candidate-generation failure.

The eleven below-ten cases provide concrete ranking examples for future routing or reranking experiments. Seven of the ten missing-window cases are cross-file questions, which motivates examining broader evidence coverage. Before changing ranking defaults, freeze and independently review a separate multi-repository validation set; inspect internal branch exports where a causal attribution is needed. No new ranking policy or held-out target is claimed by this batch.

## Verify without inference

```sh
python3 -B benchmarks/retrieval/cubecl64-candidates-2026-11-12-6104689/verify.py
```

The standalone verifier reads the archive without extracting or executing its producer code. It checks exact file inventories and hashes, frozen input/model/source identities, the snapshot receipt, original label bytes, original/new report agreement, complete rank order/budgets/prefixes, independently recomputed Recall@5/10, final-window recall/MRR, category aggregates and diagnostic projections. The actual Arrow bytes were checked during execution; the large index and weights are excluded from publication.

`window-summary.json` contains category/strategy summaries plus all 21 incomplete hybrid Top10 cases and their labeled ranks. The evidence archive retains the full 384-case rankings, complete per-query diagnostics and canonical input, raw command output, before/after identities, the prior native report, comparison records, producer sources and an observed-after-run dependency lock. Historical local paths are evidence, not commands to execute unchanged. The full ranking beyond ten was unavailable in the original report; only its metrics and displayed results can be compared directly.

Use [the native diagnostic consumer](../../../docs/repository-native-diagnostics.md) on any new format-version-3 report to perform the same strict model-free checks. Format-version-2 consumers retain their old limitations; this batch verified the prior four-file diagnostic bundle remains byte-identical under the updated consumer.
