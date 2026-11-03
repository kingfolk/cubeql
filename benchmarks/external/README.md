# Portable external retrieval benchmarks (RQ0)

This recipe turns the local CoIR APPS experiments into independently runnable stages:
input validation, pinned preparation, encoding, CubeQL replay, exact-vector reference,
native scoring and archive verification. See the [local validation report](VALIDATION.md)
for completed checks and their scope. It uses the merged external benchmark library;
it does not add a production model provider, peer product adapter or new ranking algorithm.

The recipe runs on Python 3.12 on macOS/Linux; checkpoint locking uses POSIX `flock`.
CubeQL's compute backends retain their own platform requirements. No command invokes a
paid API, coding agent, or account-configuration change. Downloads occur only with explicit
`--download`; model inference uses a fully verified local snapshot with network access disabled.

## Build and run an offline smoke test

CubeQL currently uses a sibling CubeCL checkout through Cargo path dependencies. Use the
same sibling layout described in the project README. The frozen baseline used CubeCL
`5037c0639ce3b1205392bdbe43dcf25e2ea09e29`; record the actual dependency revision for new runs.
Initial Cargo dependency/LLVM setup may require downloads. The smoke command itself uses
an already-built binary, the checked-in three-document fixture and Python's standard library.

```sh
cargo build --locked -p cubeql-code --bin cubeql-code

python3 -B benchmarks/external/runner.py smoke \
  --binary target/debug/cubeql-code \
  --output .benchmark-data/smoke-001

python3 -B benchmarks/external/runner.py verify .benchmark-data/smoke-001
```

This executes all three fixture queries, preserves the unjudged query, validates JSON/TREC
scores and order against the completed report, and seals artifact checksums. Hash embeddings
are a pipeline fixture; no semantic-quality or public-benchmark score is claimed.
The output must not already exist. Repeat the command with a new output path.

## Install experiment dependencies

Use a dedicated environment; nothing is installed into the core library:

```sh
python3.12 -m venv .benchmark-data/venv
.benchmark-data/venv/bin/python -m pip install -r benchmarks/external/requirements.txt
```

For CodeRankEmbed inference, install `requirements-coderank.txt` instead. It includes the
scoring dependencies and pins the historical CPU reference stack. The run records actual
package versions; cache compatibility checks include those versions. Capture a full
environment lock when publishing experiments on another platform.

The commands below use `python` to mean that environment's interpreter. Run them from
the repository root. `.benchmark-data/` is ignored by Git so corpora, weights and generated
indexes do not become source changes accidentally.

## Prepare exactly the frozen APPS task

```sh
python -B benchmarks/external/runner.py prepare-apps \
  --cache .benchmark-data/cache/apps \
  --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs \
  --download

python -B benchmarks/external/runner.py validate \
  .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset
```

Replace the date/commit placeholders with the experiment date and measured source commit,
not a later publication commit. Dirty source requires the recorded patch and helper sources;
the directory suffix and checkout HEAD are not binary build attestations.

`specs/coir-apps.json` pins both Hugging Face dataset revisions, the three Parquet files
and their SHA-256 values. Preparation keeps all 8,765 corpus documents and selects exactly
the 3,765 query IDs in official test qrels, following the CoIR loader. It checks the converted
JSONL/TSV bytes against the historical dataset fingerprint. IDs and metadata do not become
embedding text; titles join text with one newline only when nonempty. No rechunking or
positive-document filtering occurs. APPS titles are empty in this release.

Without `--download`, a verified local cache is required. A conflicting existing cache file
causes an error and is never replaced automatically. Derived dataset output is create-new.

## Reproduce historical scores without running models

The [frozen evidence bundle](baselines/coir-apps-2026-11-01-89c431e/README.md) contains original
ID/score rankings, timings/provenance and per-query metrics. It excludes corpus/query text,
qrels, vectors, weights and third-party source. Archived metadata can contain historical
machine paths; they are evidence, not the portable command recipe.

```sh
python3 -B benchmarks/external/runner.py audit-baseline \
  benchmarks/external/baselines/coir-apps-2026-11-01-89c431e

python -B benchmarks/external/runner.py fetch evaluator \
  --cache .benchmark-data/cache/evaluator --download

python -B benchmarks/external/runner.py score-baseline \
  benchmarks/external/baselines/coir-apps-2026-11-01-89c431e \
  --mode coderank_8192_cubeql \
  --dataset .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset \
  --evaluator-cache .benchmark-data/cache/evaluator \
  --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/rescore-coderank
```

Available modes are listed by `audit-baseline`, including MiniLM, both BM25 controls,
offline RRF, and the two CodeRank contexts with CubeQL and NumPy rankings. Rescoring must
match the recorded native/direct metrics; a mismatch retains diagnostics and fails.
The native evaluator is pinned by commit and three file checksums, downloaded into the
explicit cache rather than vendored here. See its [upstream project](https://github.com/CoIR-team/coir).

## Run a new MiniLM condition

The existing CLI/provider is reused, with separate model loading/inference/retrieval timings:

```sh
cargo build --locked --release -p cubeql-code --features local-embeddings

python -B benchmarks/external/runner.py minilm \
  .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset \
  --binary target/release/cubeql-code --model-dir /path/to/pinned-minilm \
  --top-k 100 --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/minilm
```

Use the existing explicit model-download workflow documented in
[embedding providers](../../docs/embedding-providers.md); this command performs no model download.

## Encode CodeRankEmbed with resumable batches

```sh
python -B benchmarks/external/runner.py fetch coderank \
  --cache .benchmark-data/cache/coderank --download

python -B benchmarks/external/runner.py encode-coderank \
  .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset \
  --model-dir .benchmark-data/cache/coderank --max-tokens 8192 \
  --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/vectors-8192
```

The pinned model snapshot uses CLS pooling, a required query prefix, L2 normalization and
FP32 CPU inference. Contexts 256 and 8,192 are separate conditions. Length-bounded batches
preserve document/query row alignment; the final manifest binds IDs, dataset bytes, the
embedding contract and little-endian FP32 vector hashes. Encoding is a long-running process
with no inherited 600-second harness timeout.

Resume an **incomplete** encoding by repeating its exact command with `--resume`.
The input/model/version/recipe contract must match. Completed batch files are atomically
published and reused; a conflicting or corrupted checkpoint fails. An OS process lock prevents
two encoders from using one run. Completed runs are verified, not restarted. A changed
configuration/source needs a new output directory. All failed attempt records remain.

## Replay vectors and compare the exact engine

```sh
cargo build --locked --release -p cubeql-code --example benchmark_replay

python -B benchmarks/external/runner.py replay \
  .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset \
  --binary target/release/examples/benchmark_replay \
  --vectors .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/vectors-8192/vectors.json \
  --top-k 100 --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/cubeql-8192

python -B benchmarks/external/runner.py reference \
  .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset \
  --vectors .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/vectors-8192/vectors.json \
  --compare-report .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/cubeql-8192/retrieval/report.json \
  --top-k 100 --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/numpy-8192
```

The Rust example verifies vector checksums, dataset bytes and ID order before creating a
run, then calls the existing `benchmark::run` library API. Duplicate document text retains
distinct rows/vectors. Its embedding timers measure **vector lookup**, not model inference;
the encoder's timing records are separate. This example is an experiment tool, not a new
production CLI provider. RQ1 still owns that product integration.

The NumPy control exhaustively ranks the identical FP32 bytes using float64 cosine and
corpus-row tie order. It retains all score/rank differences and near-tie gaps. Do not demand
bitwise agreement between different floating-point reduction paths or perturb scores to
force matching ranks. The reference is a correctness diagnostic, not a controlled CPU/GPU
speed comparison.

## Score a new ranking export

```sh
python -B benchmarks/external/runner.py score \
  .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/inputs/dataset \
  --results .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/cubeql-8192/retrieval/results.json \
  --trec .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/cubeql-8192/retrieval/run.trec \
  --evaluator-cache .benchmark-data/cache/evaluator --cutoffs 1 5 10 100 \
  --output .benchmark-data/coir-apps-YYYY-MM-DD-COMMIT/scores-cubeql-8192
```

For the NumPy export, pass its `results.json`/`run.trec` and explicit `--candidate-window 100`.
Other exports without a completed sibling `report.json` also require a declared window.
This prevents accidental claims of MRR@100 from a Top-10 export. Sparse/empty result sets
within a declared budget remain valid misses; missing result query IDs are rejected.

Scoring selects all **judged** input queries, preserving zero/negative integer grades and
the evaluator's semantics. Unjudged queries remain in retrieval exports but do not enter
the quality denominator; they are not labeled no-answer examples. By default, identical
query/document IDs are removed consistently for every metric, matching native CoIR policy;
`--include-identical-ids` explicitly changes that policy. APPS namespaces do not overlap.
The original rankings and judgments are never rewritten.

Native CoIR nDCG/MAP/recall/precision and MRR are retained alongside unrounded direct
pytrec metrics. Native MRR uses stable JSON score ordering; pytrec can order equal-score
documents differently. TREC/report preserve engine ranks, while JSON object order is not
a portable rank contract. Report both tie-sensitive MRR values and their candidate window.

## Tests and evidence boundaries

```sh
python3 -B -m unittest discover -s benchmarks/external/tests -v
cargo test --locked -p cubeql-code --example benchmark_replay

# Enable real native-evaluator calibration tests after fetching the small pinned cache:
CUBEQL_COIR_EVALUATOR=.benchmark-data/cache/evaluator \
  python -B -m unittest discover -s benchmarks/external/tests -v
```

Standard-library tests run without downloads. Vector/checkpoint tests additionally use the
benchmark environment's NumPy; native tests require the verified local evaluator cache.
They cover hand-computable graded metrics, unjudged/empty queries, export corruption,
changed resume contracts and interrupted-batch reuse. Real models/full corpora are explicit
jobs, not ordinary CI tests.

A final `manifest.json` seals each completed stage. `verify` checks exact artifact membership
and bytes; hashes detect changes, they do not attest a model's training provenance or a
binary's build. Failed stages retain attempts and have no completion manifest. Retries for
non-encoding stages need a new directory. Runtime import-cache files and the process lock
are explicitly excluded from the encoder's immutable payload.

One APPS task is not the ten-task CoIR average, a product leaderboard or agent success.
Record full/pilot coverage, cutoff/denominator changes, cold versus resident timing, hardware
and measured versus estimated memory. Warm timing repeats are not independent quality
samples. Do not tune on observed test results or overwrite a less favorable run.
