# RQ0 local acceptance report — 2026-11-01

RQ0 makes the existing external benchmark evidence portable. It adds staged commands for
input preparation, offline validation, resumable CodeRankEmbed encoding, ID-bound vector
replay, an independent numerical reference, native scoring and historical archive checks.
It does not change a retrieval algorithm or the production embedding-provider interface.

These acceptance checks ran locally before publication and made no paid model calls.
The measured checkout is `3b9e3e0cf9ecf65ab00abccccb7ac31db5cff4d0` plus the local RQ0
changes. CubeCL is `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`. The historical experiments
have their own original source identities; this new validation does not relabel their builds.

## Acceptance checks

| Check | Result |
|---|---|
| Python recipe tests, with NumPy and verified native evaluator | 13 passed |
| Rust vector replay example tests | 3 passed |
| Existing external benchmark regression tests | 7 passed |
| `cargo fmt --all -- --check` | Passed |
| Clippy on the new example with `-D warnings` | Passed |
| Release build of the replay example | Passed |
| Offline CLI fixture: three documents, three queries | JSON/TREC/report agree; one unjudged query retained |
| Reuse of a completed smoke output directory | Rejected; original seal still verifies |
| Pinned APPS preparation using the local download cache | Exact historical converted file hashes matched |
| Actual CodeRankEmbed encoding of the tiny fixture | Three document and three query vectors produced offline |
| Tiny saved-vector replay, NumPy reference and native scoring | Completed; both judged positives ranked first |
| Full APPS saved-vector replay and numerical comparison | Completed for all 3,765 queries |
| Historical rankings rescored through the pinned evaluator | All eight conditions matched |

The tiny fixture validates the pipeline, not public semantic quality. Encoder tests interrupt
a batch callback and confirm that resume reuses the 16 completed rows while encoding only
the remaining four, preserving the original row order. Other tests check altered resume
contracts, duplicate IDs, invalid grades, vector/data hashes, export corruption, sparse
results, judgment coverage and self-ID policy. Native metric calibration uses hand-computable
graded nDCG and MRR examples.

Tests and fixture commands are documented in [the portable recipe](README.md). Full-model
inference is an explicit job rather than a normal test. The local Python stack used Python
3.12, NumPy 2.2.6, PyArrow 20.0.0, PyTorch 2.7.1, sentence-transformers 4.1.0,
transformers 4.51.3 and pytrec-eval-terrier 0.5.11. Each stage retains actual versions,
hardware, recipe sources, tracked patches and executable hashes where applicable.

## Full APPS replay

The new preparation command reconstructed the same 8,765-document corpus and 3,765 official
test queries from the pinned Parquet cache. All corpus train/test records remain searchable;
test qrels select query IDs without filtering candidate documents. Labels are not embedded.

The new Rust example loaded copies of the previously saved CodeRankEmbed-8192 FP32 vectors.
The conversion retained the original embedding contract and verified raw vector checksums;
it bound the new manifest to dataset bytes and exact document/query ID order. No full-corpus
model encoding was repeated. The fresh release example then called CubeQL's existing
benchmark library and exported Top-100 cosine results on the CPU backend.

| Engine, identical saved vectors | nDCG@10 | Recall@10 | Recall@100 | MRR@100 |
|---|---:|---:|---:|---:|
| CubeQL FP32 | 0.2345962350 | 0.3288180611 | 0.5667994688 | 0.2138692292 |
| Exhaustive NumPy float64 cosine | 0.2345962350 | 0.3288180611 | 0.5667994688 | 0.2138692292 |

The table uses unrounded direct pytrec values; native CoIR rounds its aggregate output.
Native nDCG and MRR also agree between the engines at all requested cutoffs (1, 5, 10, 100).
These values are normalized metrics, not accuracy percentages.

Rank identity is a separate diagnostic:

- Top-10 order agrees for 3,761 / 3,765 queries.
- Top-100 order agrees for 3,681 / 3,765 queries.
- Top-100 candidate sets agree for 3,764 / 3,765 queries.
- Maximum returned-score error is `1.2152044149e-6`, below the local acceptance bound `5e-6`.
- The largest reference-score gap among differently ordered positions is `5.9752186476e-7`.

All 84 queries with Top-100 order differences remain in `parity.json`; scores were neither
perturbed nor removed. The results support numerical agreement for this workload. They do
not establish bitwise equivalence for every input, model quality improvements, agent gains,
or a controlled CPU/Metal speed comparison. Replay embedding timers measure vector lookup;
the original model-inference timings remain separate.

## Historical rescoring

The local [frozen bundle](baselines/coir-apps-2026-11-01-89c431e/README.md) contains 35 archive
members and approximately 61 MiB of compressed evidence. It includes original ID/score
rankings, per-query metrics, provenance, failed preflight diagnostics and source amendments.
It excludes corpus/query text, qrels, vectors, weights, indexes and third-party evaluator
source. Archive metadata and every member checksum were audited before rescoring.

All eight conditions matched their recorded native nDCG@10, Recall@100, native MRR@100 and
unrounded pytrec reciprocal rank, within the declared `6e-6` historical-rounding tolerance.

| Historical condition | Native nDCG@10 | Recall@10 | Native MRR@100 |
|---|---:|---:|---:|
| MiniLM, 256 tokens | 0.06596 | 0.09907 | 0.06082 |
| BM25, word tokenizer | 0.01152 | 0.01620 | 0.01153 |
| BM25, code tokenizer | 0.01501 | 0.02284 | 0.01452 |
| Experimental RRF | 0.06354 | 0.09641 | 0.05325 |
| CodeRankEmbed-256, CubeQL | 0.19840 | 0.27703 | 0.18205 |
| CodeRankEmbed-256, NumPy | 0.19840 | 0.27703 | 0.18205 |
| CodeRankEmbed-8192, CubeQL | 0.23460 | 0.32882 | 0.21387 |
| CodeRankEmbed-8192, NumPy | 0.23460 | 0.32882 | 0.21387 |

RRF has many equal scores. Its unrounded pytrec reciprocal rank is `0.0568851227`, whereas
native CoIR MRR@100 is `0.05325`: the evaluators use different tie ordering. Both are retained.
The bundle's BM25 controls are historical Python experiments, not a new supported lexical
strategy in `cubeql-code`. No peer search product was independently run in RQ0.

## Local artifacts and limits

This acceptance run's local artifact directory is
`rq0-checks/validation-2026-11-01-3b9e3e0`, under the task workspace. It contains:

- `acceptance.json`, command stdout/stderr and exit records;
- `smoke/`, `tiny-vectors/` and the complete tiny replay/reference/scoring stages;
- `saved-vectors-8192/origin.json` documenting the copy and manifest conversion;
- `full-saved-8192/` with fresh raw rankings, native scores and all parity diagnostics;
- `historical-rescores/` with eight extracted ranking/scoring/comparison directories.

Generated datasets, vectors, models and these acceptance artifacts are outside the Git
working tree. New commands accept explicit paths, use create-new output directories and
seal completed stages. Interrupted encoders can resume only with the same contract; failed
attempts are preserved. The baseline's source snapshot records its packaging state, while
new run snapshots record the recipe used for each new stage. A checksum is an integrity
check, not a binary-build attestation.

Only one public CoIR task was exercised. There is no fresh untouched validation split,
ten-task suite result, independent competitor reproduction or coding-agent outcome here.
The next milestone, RQ1, still owns reusable production support for code models/precomputed
vectors and an independently run stronger-model reference.
