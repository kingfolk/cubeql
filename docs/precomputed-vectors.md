# External code embeddings and reusable vector bundles (RQ1)

CodeRankEmbed and other code models can now feed the supported `cubeql-code` CLI. Python
turns identified texts into vectors; Rust validates those vectors, builds an Arrow snapshot
and searches it. Neither the core `cubeql` crate nor its users acquire a Python/model
runtime dependency. Hash and optional MiniLM inference continue to work.
The Python encoder currently targets macOS/Linux and uses POSIX checkpoint locks; Rust
bundle loading does not depend on that encoder runtime.

## Small end-to-end example

Run from the repository root with the sibling CubeCL checkout available:

```sh
cargo build --release -p cubeql-code
mkdir -p .benchmark-data
python3.12 -m venv .benchmark-data/code-env
.benchmark-data/code-env/bin/python -m pip install -r tools/requirements-code-embeddings.txt

.benchmark-data/code-env/bin/python -B tools/encode_vectors.py fetch \
  --model coderank --model-dir .benchmark-data/models/coderank --download
```

The downloader pins the model revision and checks every file. Omit `--download` to verify
an existing cache. Encoding and querying never download missing artifacts or fall back to
hashing. CodeRankEmbed's pinned model card declares MIT; it is an optional external model.
The card describes CoRNStack contrastive training. This workflow does not establish that
APPS was excluded from all of its training/pretraining sources; unknown exposure stays unknown.

Export the actual three function bodies plus one natural-language query:

```sh
target/release/cubeql-code vectors prepare \
  crates/cubeql-code/tests/fixtures/semantic-demo \
  --query "Where do we read configuration from a file?" \
  --output .benchmark-data/demo-inputs.json

.benchmark-data/code-env/bin/python -B tools/encode_vectors.py encode \
  .benchmark-data/demo-inputs.json --model coderank \
  --model-dir .benchmark-data/models/coderank --max-tokens 256 \
  --output .benchmark-data/demo-vectors

target/release/cubeql-code index \
  crates/cubeql-code/tests/fixtures/semantic-demo \
  --embedding precomputed --vectors .benchmark-data/demo-vectors/vectors.json \
  --output .benchmark-data/demo.arrow

target/release/cubeql-code search .benchmark-data/demo.arrow \
  "Where do we read configuration from a file?" \
  --vectors .benchmark-data/demo-vectors/vectors.json --query-id query \
  --top-k 3 --include-source --format json
```

The result includes the saved source path, symbol, line range and cosine score. The source
snapshot contains the model contract and document vectors; it does not require the original
document bundle or a running Python process to stay available. Semantic search still needs
a vector for its query. Lexical/symbol search needs neither a query vector nor a model.

Use new output paths for fresh jobs; only an incomplete encoder job may reuse its directory
with `--resume` as described below. Generated corpora, weights and vectors belong under the
ignored `.benchmark-data/` or another explicit local artifact directory, outside source control.

## Ask another question

Encode only the new query using the same model, context limit, implementation and runtime:

```sh
target/release/cubeql-code vectors query "Where are configuration settings saved?" \
  --id save-settings --output .benchmark-data/save-inputs.json

.benchmark-data/code-env/bin/python -B tools/encode_vectors.py encode \
  .benchmark-data/save-inputs.json --model coderank \
  --model-dir .benchmark-data/models/coderank --max-tokens 256 \
  --output .benchmark-data/save-vectors

target/release/cubeql-code search .benchmark-data/demo.arrow \
  "Where are configuration settings saved?" --query-id save-settings \
  --vectors .benchmark-data/save-vectors/vectors.json --format json
```

This does not encode the document corpus again. It imports a **query-only** bundle: its
document matrix has zero rows. A bundle supplies already computed vectors; it cannot
generate an arbitrary unseen query by itself. An application can automate the query
encoder separately and use `EmbeddedQuery` or the provider API in a resident service.

`--query-id` disambiguates identical text assigned to different IDs. Without it, text lookup
must find exactly one query; duplicate query text never silently picks a row. Imported
query text and ID must both match, and the full saved model contract must be compatible.

## Incremental updates

For a real repository, export only chunks whose source has no reusable previous vector:

```sh
target/release/cubeql-code vectors prepare /path/to/repository \
  --previous .benchmark-data/repo-v1.arrow --output .benchmark-data/update-inputs.json

.benchmark-data/code-env/bin/python -B tools/encode_vectors.py encode \
  .benchmark-data/update-inputs.json --model coderank \
  --model-dir .benchmark-data/models/coderank --max-tokens 256 \
  --output .benchmark-data/update-vectors

target/release/cubeql-code update .benchmark-data/repo-v1.arrow \
  --vectors .benchmark-data/update-vectors/vectors.json \
  --output .benchmark-data/repo-v2.arrow --format json
```

If only one of 100 unchanged function bodies changes, only that function needs a new
embedding. Line shifts alone can reuse exact unchanged source. The builder rewrites the
complete Arrow snapshot with current locations and accurate add/change/remove statistics;
it does not update an index in place.

When every document can be reused (including removal-only changes), skip encoding and run
`update OLD --output NEW` without `--vectors`. An empty preparation with no queries is
rejected rather than creating a meaningless encoder job. New/changed source without a
bundle fails explicitly. `--root` can relocate an existing repository as before.

Preparation and construction share one scanner/chunker and the same duplicate-aware reuse
algorithm. If the source changes between export and import, the expected pending IDs/text
will differ and the import fails. Export again instead of relabeling old vectors. First-time
preparation accepts `--fallback-lines` and `--max-file-bytes`; use matching index options.

## Data contract and library API

The input file has exactly these fields:

```json
{
  "format_version": 1,
  "documents": [{"id": "code:...", "text": "fn ...", "content_hash": "<BLAKE3 of raw UTF-8 text>"}],
  "queries": [{"id": "query", "text": "Where ...?", "content_hash": "<BLAKE3 of raw UTF-8 text>"}]
}
```

IDs are nonempty tokens and unique within each role. The two roles have separate ID spaces.
One role may be empty. Repository IDs bind the chunker version, path, language, kind, symbol,
line range and source hash. They identify exported inputs rather than permanent device rows.
Unchanged source can be reused across updates even when its exported ID changes after a line shift.
`--queries FILE` accepts JSONL records with `_id` and `text` instead of `--query TEXT`.

The encoder writes `inputs.json`, little-endian `documents.f32` and `queries.f32`, and:

```json
{
  "format_version": 1,
  "dtype": "float32_le",
  "config": {"...": "the complete EmbeddingConfig"},
  "inputs": {"file": "inputs.json", "sha256": "..."},
  "documents": {"file": "documents.f32", "sha256": "..."},
  "queries": {"file": "queries.f32", "sha256": "..."}
}
```

Rust checks file paths, checksums, exact ordered input coverage, text hashes, matrix shape,
finite values and normalization. Identical document text with different IDs retains separate
rows and vectors. Model revision, role-specific prompts, pooling, context limit, normalization,
artifact hashes and the output-affecting runtime/device identity must match. Equal dimensions
alone do not establish compatibility. The Python encoder applies each role's prefix exactly
once; raw source/query text stays unprefixed in the input contract.

The library exposes `EmbeddingInput`, `EmbeddingInputs`, `VectorBundle` and `BundleProvider`
in `cubeql-embeddings`. `VectorBundle::ensure_documents` / `ensure_queries` compare independently
prepared inputs; `document` / `query` look up by ID and verify exact text. `BundleProvider`
implements the new ID-aware provider methods while existing live providers inherit text
dispatch. `prepare_repository`, `RepositoryPlan::pending_inputs` and
`build_repository_with_provider` support the same flow without invoking the CLI.

The older exact-text `PrecomputedProvider` remains available for fixtures. Production bundle
imports use IDs, so duplicate text cannot collapse two documents into one vector. A checksum
protects file identity; it cannot prove that a third party truthfully encoded its stated texts.

## Resume and optional Jina reference

Append `--resume` to the same incomplete encoding command after interruption. Completed batch
checkpoints are reused only with the identical input/model/configuration/runtime/recipe.
Completed jobs are never restarted in place. Failure records and command attempt history
remain. The CPU encoder uses four PyTorch threads and FP32 inference; model load and actual
encoding durations are separate from CubeQL's vector-lookup/upload/retrieval durations.

An optional Jina reference uses a separate environment:

```sh
python3.12 -m venv .benchmark-data/jina-env
.benchmark-data/jina-env/bin/python -m pip install -r tools/requirements-jina.txt
.benchmark-data/jina-env/bin/python -B tools/encode_vectors.py fetch \
  --model jina --model-dir .benchmark-data/models/jina --download
.benchmark-data/jina-env/bin/python -B tools/encode_vectors.py encode \
  .benchmark-data/demo-inputs.json --model jina \
  --model-dir .benchmark-data/models/jina --max-tokens 1024 \
  --output .benchmark-data/demo-jina-vectors
```

This pins `jinaai/jina-code-embeddings-0.5b` revision
`4db235132dafbe56a8b9c5f59b59795ecf58a4a7`: 896 dimensions, NL2Code role prompts,
last nonpadding-token pooling and L2 normalization. FP32/SDPA is explicit; FlashAttention
and CUDA are not required. On supported Apple hardware, add `--device mps` for Jina inference;
unavailable MPS fails rather than falling back. Device choice is part of query/index
compatibility. This is independent of the CubeCL retrieval backend.

Jina's weights are **CC-BY-NC-4.0**, so this optional model is a noncommercial reference,
not CubeQL's default distributable model. Its paper discloses APPS training data; APPS test
scores must be described as partly in-domain, with context and precision differences recorded.
No gated terms are accepted automatically. Sources: [Jina model card](https://huggingface.co/jinaai/jina-code-embeddings-0.5b),
[training/benchmark paper](https://arxiv.org/html/2508.21290v2),
[CodeRankEmbed model card](https://huggingface.co/nomic-ai/CodeRankEmbed).

## Public benchmark integration and checks

`cubeql-code benchmark prepare DATASET --output INPUTS.json` exports the canonical corpus
and queries without relevance judgments. Encode those inputs, then use the supported
`benchmark run DATASET --embedding precomputed --vectors VECTORS.json --output NEW --top-k 100`.
The report labels embedding timers as `precomputed-vector-lookup`. Model inference is timed
only by the encoder. The independent `benchmarks/external/runner.py reference` command accepts
both RQ0 experiment manifests and the new production bundle format; native scoring is unchanged.

The repository `evaluate` command also accepts `--vectors`. Its query bundle must match the
evaluation query set's IDs, text and order; query IDs survive the shared semantic/hybrid
capture path, including different vectors for identical text. Relevance labels remain outside
the embedding inputs.

Offline tests:

```sh
cargo test -p cubeql-embeddings -p cubeql-code
.benchmark-data/code-env/bin/python -B -m unittest discover -s tools/tests -v
```

Tests cover corrupted/missing/reordered inputs, vector dimensions/configuration, NaNs,
duplicate text IDs, source edits after preparation, full persistence and incremental reuse.
Real-model inference and full-corpus comparisons remain explicit local jobs.

## Local validation

The [2026-11-03 RQ1 report](../benchmarks/external/reports/rq1-coir-apps-jina-2026-11-03-a4b7abe/README.md) records the
complete APPS reference, same-vector CubeQL/NumPy comparison, real-model product examples
and source/binary identities. The long Jina MPS job needed two fresh-process resumes after
allocator-memory failures; completed checkpoints were retained and no memory safety limit
was disabled. This optional encoder is not a resident, memory-bounded inference service.
