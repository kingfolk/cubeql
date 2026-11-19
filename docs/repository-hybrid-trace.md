# Observed Hybrid-stage diagnostics

The complete final window locates misses relative to a display cutoff. The opt-in
Hybrid trace additionally exposes the bounded, already-filtered branch windows and
observed fusion/deduplication stages. It preserves the ranking policy and budgets.

```sh
cubeql-code evaluate saved.arrow queries.json --encoder-config /profiles/local.json \
  --candidate-k 100 --repeats 3 --trace-hybrid --format json > native-trace.json
python3 -B tools/repository_hybrid_trace.py \
  --report native-trace.json --output /path/to/new-trace-diagnostics
```

`--trace-hybrid` implies complete candidate export. The outer wrapper stays version 1;
the nested evaluation is version 4. Every query has a `hybrid_trace`, and all six
strategies retain the version-3 complete final ranking. Without tracing, ordinary
reports and candidate-only reports keep their versions 2 and 3 respectively. The
original native diagnostic consumer remains available for those reports.

The new model-free consumer validates the complete version-3 projection first, including
all queries/strategies/precisions, independent full-window recall/MRR and category
arithmetic. It then verifies each trace before creating output:

- All three filtered branches, in semantic/lexical/symbol order, respect `candidate_k`.
- Source-free records have contiguous ranks, stable snapshot rows, inclusive line
  ranges, finite float32 scores and consistent paths/symbols across stages.
- The complete RRF union follows the fixed k=60 equal-weight policy. Ordering uses
  float64 accumulated scores and row tie order before conversion to float32, matching
  the producer; equal serialized float32 scores do not justify reordering rows.
- The full kept list and every suppression witness match greedy, rank-preserving
  same-path inclusive-range IoU >= 0.8. Deduplication is replayed before truncation.
- The final ranking is exactly the first `top_k` kept rows, with matching scores and
  displayed snapshot metadata. A suppression witness may itself lie beyond this prefix.

For each positive `(path, symbol)` label, diagnostics retain branch, fused, deduplicated
and final ranks, plus any suppression witness. The first observed loss stage is:

| Stage | Meaning within this observed condition |
| --- | --- |
| `not_in_filtered_branch_windows` | No branch exported the label within its filtered candidate budget. |
| `suppressed_by_overlap` | The union contained the label, and observed overlap deduplication suppressed it. |
| `truncated_after_deduplication` | The label survived deduplication but fell beyond the final window. |
| `retained_below_top10` | The label reached the final window at rank greater than ten. |
| `retained_top10` | The label reached the final window within ten. |

Counts distinguish repeated per-query anchor judgments from distinct queries. The
anchor stage counts partition judgments; queries containing a stage can overlap.
Category summaries retain complete question/label denominators. `diagnostics.json`
contains all findings; `README.md` is generated from them. The bundle includes a
canonical input copy, source digests and a byte-hash manifest. Existing output
folders are rejected, and malformed/incomplete traces fail before creating a folder.

An anchor absent from these filtered, bounded branches may exist elsewhere in the
repository, earlier in a filter pipeline, or below a branch cutoff. The trace does
not establish unbounded recall or a filter-specific cause. Geometric suppression is
an observed policy decision, not a relevance judgment. This is development evidence;
independent human review, multi-repository validation and coding-agent outcomes remain
separate requirements. Trace capture can affect the native deduplication timer and the last Hybrid warm-repeat timer;
these instrumented timings do not establish a speed improvement.
