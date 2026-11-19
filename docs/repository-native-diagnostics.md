# Saved native symbol-label diagnostics

`tools/repository_native_diagnostics.py` is a model-free consumer for one completed
native `cubeql-code evaluate --format json` report wrapped in the repository baseline
runner's format-version-1 output. It reads the report, recomputes and validates its
quality summaries, and writes a create-new diagnostic bundle. It does not index a
repository, launch an evaluator or model, or mutate Git state.

```sh
python3 -B tools/repository_native_diagnostics.py \
  --report /path/to/native-evaluation.json \
  --output /path/to/new-diagnostics-directory
```

The output directory must not already exist. `input-report.json` is a canonical JSON
copy of the parsed wrapper so the analysis can be replayed. `diagnostics.json` retains
all six strategies and every precision variant in dataset/query order, including
recomputed overall and per-category summaries and paired semantic-versus-lexical,
semantic-versus-symbol, and semantic-versus-hybrid per-query comparisons. `README.md`
is a compact summary table generated from the validated report, and `manifest.json`
contains SHA-256 checksums for the other three files. The diagnostic records both the
canonical report digest and the input-byte digest; the CLI byte digest distinguishes
different source encodings of the same parsed JSON.

Validation calls the native query-set validator and the repository baseline's native
report validator. It also independently checks the exact `fp32` or `fp32`/`fp16`
variant set, complete strategy and query coverage, query order and category echoes,
numeric/timer fields, recall and rank consistency, top-hit uniqueness, summary
denominators, and summary arithmetic. Because the Rust report displays at most ten hits,
Recall@5 and Recall@10 are independently recomputed from those displayed native
`(path, symbol)` labels. MRR beyond rank ten cannot be reconstructed from the display
window; reciprocal rank is checked against the reported first-relevant rank instead.
No Top100 anchor recall is inferred from top-ten hits, and labels are not converted to
source spans.

Pair outcomes use per-query Recall@10: `recovered` means semantic has zero recall and
the compared strategy has positive recall; `lost` is the reverse; `improved` and
`regressed` compare positive values; `shared_miss` means both are zero; `tied` means
equal positive recall. Each query includes Recall@5/10 and reciprocal-rank deltas plus
the two first-relevant ranks. The report contains native quality only; native latency
is intentionally omitted.

These are observations on the original symbol-label development set, not independently
reviewed validation quality. Warm repeats are timing repetitions, not independent
quality observations. The diagnostic provides no causal diagnosis, equal-work speedup,
or agent-outcome result.

## Complete final candidate windows

Opt into native evaluation format version 3 with `evaluate --export-candidates --format json`.
Each strategy then carries a `candidate_ranking` array with contiguous one-based `rank`,
snapshot `row`, `path`, `symbol` and finite `score`. The array is the complete final retained
ranking, bounded by the existing `candidate_k`; it is not the internal branch union before
fusion or deduplication. Empty and short lexical/symbol rankings are valid. The ten displayed
hits remain an exact prefix of the array.

The same diagnostic command accepts this new report. Before creating output it checks every
strategy/query/precision, array budgets and ranks, unique rows within snapshot bounds, stable
row identities across queries, exact displayed prefixes, and independently recomputes the
first relevant rank and reciprocal rank from the full array. It still recomputes displayed
Recall@5/10 and all native aggregate/category summaries. Missing or partial candidate exports
are rejected; old format-version-2 reports retain their original diagnostics and limits.

For version 3, `diagnostics.json` uses diagnostic format version 2. Each variant additionally
contains `candidate_window`, with per-strategy query-weighted overall and category summaries,
final-window recall, all-required coverage, recomputed MRR, and every label's exported rank.
Each query is classified as one of:

- `all_locations_in_top10`: every labeled location appears within ten.
- `present_below_top10`: all labels appear in the exported window, at least one below ten.
- `missing_from_exported_window`: one or more labels are absent from the final array.

The missing and below-ten lists remain separate when a query has both kinds of gaps. A
missing label may have existed in an internal pool and later been removed by fusion,
filtering or deduplication. This report cannot attribute the cause or prove an internal
candidate-generation failure. Arithmetic validation also cannot prove that an unobserved
engine pool was exported completely; it validates the declared final window.

The new Markdown summary labels its final-window bound and recomputed metrics. These remain
positive-label development observations; no labels, ranking policy or default budget changes.
Candidate export work happens outside the existing retrieval and display-materialization
timers. CLI wall time includes the extra export/serialization work and should not be compared
as a ranking-speed improvement.
