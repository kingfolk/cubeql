# Retrieval benchmark reports

These reports measure CubeQL code retrieval against a pinned source corpus. They are
developer diagnostics, not coding-agent success rates or comparisons with other products.

| Run date | CubeQL revision recorded for the run | Corpus | Report |
|---|---|---|---|
| 2026-11-12 | `965c21f` at launch; executable SHA recorded separately | CubeCL `5037c0639ce3` | [Original64 CodeRank CPU development run](cubecl64-coderank-2026-11-12-965c21f/README.md) |
| 2026-11-11 | `ca31444` base; executable SHA recorded separately | Four-file synthetic Rust development fixture | [Real CodeRank MCP / Semble pilot](native-pilot-2026-11-11-ca31444/README.md) |
| 2026-10-29 | `77b2d6af8514` | CubeCL `5037c0639ce3` | [64-query MiniLM / CPU evaluation](cubecl-2026-10-29-77b2d6af8514/REPORT.md) |

## Recording another version

Create a new directory named `<corpus>-<YYYY-MM-DD>-<cubeql-short-commit>`.
Use the experiment date and **evaluated CubeQL revision**, not the commit that publishes
the report. Record the source-corpus revision separately in `metadata.json`. Use a suffix
for additional runs of the same revision on the same date; never replace an earlier run.

Each directory should contain an English report, machine-readable metadata and summary,
query labels, per-query results and original evidence. Record the executable SHA-256 as
well as the checkout revision; a checkout revision alone does not attest how an existing
binary was built. Record model artifacts, source fingerprint, filters, precision,
candidate windows, backend and timing boundaries.

To compare code revisions, reuse the same frozen queries, corpus, model and parameters.
Keep an expanded query set separate from its predecessor. Compare matching cutoffs
(MRR@50 is different from MRR@100), and distinguish retrieval quality from latency.
Changes in corpus, model, chunking or precision need explicit paired experiments rather
than being attributed to an engine revision.

The first report includes `artifacts.tar.gz`: original JSON results, diagnostics, logs,
validation records and the scripts used at the time. Large Arrow indexes, model weights
and a duplicate source checkout are excluded. The archive contains historical local
paths; use the portable command recipe in the report instead of executing the archived
machine-specific orchestration scripts. Verify it without loading a model:

```sh
python3 -B benchmarks/retrieval/cubecl-2026-10-29-77b2d6af8514/verify.py
```

## Frozen product comparison inputs

[Repository product evaluation](../../docs/repository-product-evaluation.md) freezes reviewed
source-span anchors and scores native tool rankings with complete query coverage, explicit
failures and paired changes. Its synthetic fixture checks the interchange/scoring pipeline;
it does not add a peer-quality result to the table above.

The [multi-repository fixture](fixtures/multi-repository/spec.json) exercises version 2 plans
with colliding relative source paths and unequal query counts. It is a development smoke
fixture. Reports keep query-weighted and repository-weighted means distinct; neither is a new
held-out validation or peer-product result.

[Native CubeQL collection](../../docs/repository-collection.md) automates new snapshot builds,
query execution and frozen-plan scoring while preserving raw diagnostics and complete coverage.
