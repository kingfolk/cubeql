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
