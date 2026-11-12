# Paired native repository diagnostics

`tools/repository_diagnostics.py` produces descriptive JSON and English Markdown from saved
rankings. It launches no tool or model. All input runs must pass the existing frozen-plan
scorer, including exact query coverage, source spans, filters, statuses and plan fingerprint.

```sh
python3 -B tools/repository_diagnostics.py \
  --plan /runs/plan.json \
  --run /runs/cubeql/run.hybrid.json \
  --run /runs/semble/run.json \
  --output /runs/diagnostics-001
```

The first run is the paired baseline. The output directory must be new; the tool retains
the plan, all input run objects, original scorer output, `diagnostics.json`, `README.md`
and artifact hashes in `manifest.json`. It does not rewrite labels or reorder hits.

## What the diagnoses mean

For every query, the report retains first ranks for every required anchor and distinguishes:

| Diagnosis | Meaning |
| --- | --- |
| `execution_failure` | The condition failed, timed out or was unsupported; quality remains zero in the frozen denominator |
| `missing_from_exported_window` | At least one anchor is absent from the exported top-k list |
| `present_below_top10` | All anchors are somewhere in the export, but at least one ranks below ten |
| `all_locations_in_top10` | Every labeled anchor is covered by a returned location in the first ten |

The separate lists `missing_anchor_ids` and `below_top10_anchor_ids` retain mixed cases.
Exported-window recall is the fraction of anchors present in that run's entire exported
window, with query and equal-repository averages. Its cutoff is recorded as `candidate_k`.
This is **not internal candidate-union recall**: an absent anchor might still have been
generated internally and then excluded by native ordering or deduplication.

Paired outcomes are `recovered` (zero to positive Recall@10), `lost` (positive to zero),
`improved`, `regressed`, `shared_miss` (both zero), or `tied`. The report preserves the
existing scorer's metric deltas. These are descriptive per-query changes, without a
significance test, statistical independence claim, or explanation of implementation intent.
Inspect the original source and judgments before attributing a miss to declarations,
references, candidate generation, or reranking.

## Timing interpretation

Each condition retains its declared timing boundary. Min/median/max use successful query
rows with `elapsed_ms`; counts and completeness are explicit, so failing difficult queries
cannot silently make a condition appear faster over the full query set. Startup, indexing,
model load and source reads remain in the native collectors' evidence rather than being
silently folded into these request observations.

The paired report records whether timing-boundary names and exported windows match and
whether both conditions have complete latency observations. It does not compute a speedup
ratio or attest equal internal work. For example, Semble's default pipeline may over-fetch
five times top-k per branch and return full chunks in the search response, while CubeQL
may use another candidate budget and retrieve source separately through `read_code`.
Identical request-boundary names do not establish equal context, computation or serialization.

Native MCP collection is documented in [repository-mcp-collection.md](repository-mcp-collection.md);
the real local peer runner is documented in [repository-semble-collection.md](repository-semble-collection.md).
Keep the CLI cold-process control from [repository-collection.md](repository-collection.md) separate.

## Evaluation limits

Anchor and all-required metrics measure **source locations**. A location covering a label
does not establish that all source bytes were returned, read or used by an agent. Incomplete
judgments also mean unjudged results are not automatically false positives. Native product
conditions can differ in models, chunks, routing and budgets; improvements cannot be assigned
to a single algorithm without a separate control.

Begin with a clearly labeled development pilot, then freeze an independently reviewed
multi-repository validation set and its primary averaging rule before tuning. Preserve the
original CubeCL64 labels and reference outputs. Running a real native engine on a tiny
comment-rich fixture establishes pipeline behavior, not held-out relevance, the 80% engineering
target, or coding-task success. The latter requires separately scoped agent outcomes.
