# Code search CLI and labeled evaluation

Milestone 6 adds a usable snapshot workflow and a small, reproducible retrieval evaluator.
Everything here lives in `cubeql-code`; the generic `cubeql` Arrow/vector API still has no
code-parser, embedding-model, CLI or evaluation dependencies.

## Search a saved repository

From the workspace root, with the sibling CubeCL checkout required by this workspace:

```bash
# Offline pipeline smoke test. Hash vectors are NOT semantic embeddings.
cargo run -p cubeql-code -- index ../cubecl --output /tmp/code-v1.arrow
cargo run -p cubeql-code -- inspect /tmp/code-v1.arrow --format json
cargo run -p cubeql-code -- search /tmp/code-v1.arrow TriviallyUnrollable \
  --language rust --path-prefix crates/ --top-k 5 --format json
cargo run -p cubeql-code -- update /tmp/code-v1.arrow --output /tmp/code-v2.arrow
```

`query` remains an alias of `search`. `--format text` is the default; JSON stdout is one object
with `format_version: 1`. Warnings go to stderr. Search returns original snapshot row IDs, path,
language, symbol, kind, inclusive one-based line ranges and cosine score. Add `--include-source`
to include saved chunk text. No-match filters return `"hits": []`.

Both filters use AND semantics. Language equality is case-sensitive (`rust`, not `Rust`). A path
prefix is a case-sensitive literal prefix on repository-relative paths, not a glob; use `src/`
instead of `src` to restrict a directory. Metadata selection runs on the host using the existing
compact/before-scoring filter path. CubeCL scores only the selected rows, returning their original
row IDs; source locations are materialized on the host.

`update` restores the saved repository root, fallback chunk size, maximum file size, full provider
contract and FP16/FP32 storage. Supply `--root /new/location` if the repository moved. Unchanged
chunks reuse their vectors; changed and new chunks are embedded in batches. Removed chunks
disappear. Every update writes a **new** snapshot; old files are never overwritten. Results refer
to saved source, which may differ from the current working tree. Row IDs are snapshot-local.
For pre-M6 snapshots that did not save a file-size limit, update uses the old default of 2 MiB.

`index --precision fp16` creates FP16 vector storage. Incremental updates keep that precision.
Changing precision through `index --previous` is rejected: rebuild without `--previous` to create
a new FP32 baseline, rather than silently widening previously rounded FP16 values.

`inspect` needs neither a model nor a compute device. It reports model/preprocessing identity,
dimensions, chunk/file counts, root, builder options, logical corpus fingerprint and size estimates.
The file count is files represented by chunks, not every file visited by the builder.

## Use real embeddings

```bash
cargo run -p cubeql-code --features local-embeddings -- model download \
  --directory /tmp/cubeql-minilm
cargo run -p cubeql-code --features local-embeddings -- index \
  crates/cubeql-code/tests/fixtures/retrieval-benchmark/corpus \
  --embedding minilm --model-dir /tmp/cubeql-minilm --output /tmp/benchmark-fp32.arrow
cargo run -p cubeql-code --features local-embeddings -- search /tmp/benchmark-fp32.arrow \
  "Where do we read application settings from a file?" \
  --model-dir /tmp/cubeql-minilm --language rust --top-k 3 --include-source --format json
```

Search and update select the provider from saved metadata and verify the entire embedding
contract. They never infer a different model from dimensions alone. The CLI implements the hash
fixture and pinned MiniLM; custom providers remain available through the library. MiniLM runs
on the CPU even when CubeCL retrieval runs on Metal. Only the explicit download command uses
the network; inference uses the local artifacts. See [embedding-providers.md](embedding-providers.md).

## Run the labeled evaluator

```bash
cargo run -p cubeql-code --features local-embeddings -- evaluate /tmp/benchmark-fp32.arrow \
  crates/cubeql-code/tests/fixtures/retrieval-benchmark/queries.json \
  --model-dir /tmp/cubeql-minilm --candidate-k 24 --repeats 5 --compare-fp16 --format json

# Run the exact same saved corpus and query labels on native Metal, with CPU disabled.
cargo run -p cubeql-code --no-default-features --features metal,local-embeddings -- \
  evaluate /tmp/benchmark-fp32.arrow \
  crates/cubeql-code/tests/fixtures/retrieval-benchmark/queries.json \
  --model-dir /tmp/cubeql-minilm --candidate-k 24 --repeats 5 --compare-fp16 --format json
```

For a fully offline smoke test, build the same corpus with default hash embeddings and omit
`--model-dir`/`local-embeddings`. That tests reporting and ranking mechanics, not semantic quality.
Backend feature selection is unchanged; the JSON report records the actual selected device.

The checked-in fixture has 24 documented Rust functions and 8 manually labeled queries, two
each for `identifier`, `conceptual`, `behavioral`, and `cross_file`. Cross-file queries require
multiple relevant chunks, not just the first hit. The query file is outside the corpus, so labels
are not accidentally embedded. Labels resolve by exact `(path, symbol)` against the snapshot.
Missing, ambiguous, duplicate or filtered-out labels fail before query inference. To evaluate a
real repository, supply your own query JSON with the same schema:

```json
{
  "version": 1,
  "name": "my-reviewed-queries-v1",
  "queries": [{
    "id": "find-config-reader",
    "category": "conceptual",
    "text": "Where are configuration files read?",
    "filter": {"language": "rust"},
    "relevant": [{"path": "src/config.rs", "symbol": "load_config"}]
  }]
}
```

Review relevance labels manually. Every label contributes equally; there is no graded relevance,
answer synthesis or graph traversal. Unknown JSON fields are rejected to catch misspelled filters.

## What is compared?

- **Semantic:** the saved embedding provider plus exact CubeCL cosine top-k. With hash embeddings
  this report key still describes the vector path, not a learned semantic model.
- **Literal:** case-sensitive saved-source substring search, ranking exact qualified symbols
  before unqualified symbols, then source matches; equal scores use ascending row IDs. This is a
  small baseline, not BM25, a tuned lexical retriever, or all of ripgrep's features.
- **Experimental RRF:** fuse the top `candidate_k` lists of both paths using equal weights and
  `sum(1 / (60 + one_based_rank))`, then return at most `candidate_k` results. Raw cosine and
  literal scores are not added together. See the [RRF definition](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion).

RRF exists **only in the evaluator**. The `search` command is still vector-only. Production
strategy routing, ripgrep/symbol adapters, source provenance and overlap deduplication belong to
Milestone 7. An experimental improvement on this fixture is not proof that hybrid search is ready.

Recall@5/10 is the fraction of *all labeled relevant chunks* found in the first 5/10 returned
results. Reciprocal rank is `1 / rank_of_first_relevant`, or zero if no relevant result appears in
the candidate window. `mrr` is their mean, therefore **MRR@candidate_k**, not unbounded MRR.
Aggregates are unweighted means over queries, both overall and per category. `candidate_k` must
be at least 10; repeats must be positive. Metrics use the full candidate list, while each query's
`top_hits` contains only the first 10 display results.

## Timing and memory interpretation

Search JSON separates IPC reading/validation, model loading, query embedding, index construction/
upload submission, filter preparation, scoring/top-k/readback, and result materialization.
Each CLI invocation reloads the model/index; embedding the library in a long-running process is
the way to avoid that repeated cost.

Evaluation embeds each query **once** and reuses those exact FP32 query vectors across precision
variants. FP16 vectors are rounded from the same original FP32 snapshot, keeping source rows,
labels and model identity fixed. Variants run sequentially with one resident device index at a
time. An FP16 input cannot be used with `--compare-fp16`, since widening cannot reconstruct an
original FP32 baseline. The corpus fingerprint hashes ordered `(path, symbol, content_hash)` and
does not depend on root, model or precision; it is a comparison aid, not a security attestation.

For every query the evaluator records the first semantic invocation separately, then min/median/
max over repeated warm retrievals. The first invocation for a later query is **not** a fresh-process
cold start: compiled kernels and runtime state can already be warm. The first invocation of the
FP16 variant may compile a different kernel. Warm semantic timing includes device scoring, top-k
and final readback, but excludes embedding, filter preparation and materialization. Warm RRF timing
includes rerunning both retrieval branches plus fusion, not just the merge. Literal timing includes
host filtering and scanning. Reusing a prepared device filter is part of the semantic path.

Upload and filter timers measure host-side setup/submission, not independently synchronized GPU
transfer time. Deferred work can appear in the following query. Conversion timing includes a
host snapshot clone. The evaluator is not an isolated kernel microbenchmark or an end-to-end
cold/warm serving benchmark. Use optimized builds and sufficiently large corpora before drawing
performance conclusions; the checked-in smoke corpus is much too small to amortize GPU overhead.

Size reporting deliberately distinguishes:

- `vector_storage_bytes`: logical `rows * dimensions * (2 or 4)` bytes; not allocator usage.
- `arrow_array_memory_bytes`: Arrow's buffer/allocation accounting; not process RSS, model
  weights, duplicated Rust strings, retained baseline snapshots, or GPU/runtime allocations.
- `source_text_bytes`: total chunk source UTF-8 bytes.
- `ipc_bytes`: bytes when serialized by the current writer (counted without writing another file).
- CLI `input_file_bytes`: actual saved input file size.

There is no peak-memory/RSS measurement. Halving vector bytes does not halve total application
memory or IPC size, since source and other metadata are unchanged. Arrow allocation capacities
can differ between freshly built and IPC-loaded arrays; shared buffers can be counted repeatedly.
Do not interpret this accounting as a deduplicated physical-memory measurement.

## Limits and verification

### Local smoke observation

An Apple M1 run using the pinned MiniLM, debug builds, `candidate_k=24` and 5 repeats produced
the following quality results on both CubeCL CPU and native Metal, for both FP32 and FP16:

| Strategy | Recall@5 | Recall@10 | MRR@24 |
| --- | ---: | ---: | ---: |
| Literal baseline | 0.2500 | 0.2500 | 0.2500 |
| MiniLM + device cosine | 0.9375 | 1.0000 | 1.0000 |
| Experimental RRF | 0.9375 | 1.0000 | 1.0000 |

Literal matching found the two identifier queries but none of the six natural-language queries.
Semantic and RRF Recall@5 were 1.0 for identifier/conceptual/behavioral categories and 0.75 for
cross-file queries, reaching 1.0 at 10. RRF did **not** improve over semantic-only on this fixture.

The 24 x 384 vector storage was 36,864 bytes in FP32 and 18,432 in FP16. With the same local root
metadata, IPC sizes were 51,258 and 32,754 bytes respectively. The logical corpus fingerprint was
`93a72055c16efefb00ea1ab6f9adea4f760a586d609469d4753c64f0875bdb4f`.

For timing sanity checks, the ranges of the eight per-query warm semantic medians were:

| Backend | FP32 | FP16 |
| --- | ---: | ---: |
| CubeCL CPU | 0.210–0.993 ms | 0.310–1.123 ms |
| CubeCL Metal | 0.577–0.696 ms | 0.659–1.866 ms |

These are debug smoke observations, not isolated performance benchmarks or a speedup claim.
The queries include different metadata selections. First semantic calls also incurred first-use
work (about 107 ms on CPU and 281 ms on Metal for the first FP32 query), unlike the warm numbers.
In these CLI runs, model loading took about 4.7–5.2 seconds and embedding all eight queries took
2.3–2.7 seconds; those phases are deliberately excluded from warm retrieval. Real service
decisions require optimized builds, larger representative datasets and repeated controlled runs.

### Scope

The fixture is a **synthetic, comment-rich smoke dataset**, not a representative code-retrieval
benchmark. Natural-language queries often paraphrase comments; cross-file labels are manually
chosen and incomplete for other tasks. Tiny candidate sets and narrow path filters make recall
easier. MiniLM is a general English sentence model, not code-specialized. Results cannot establish
production quality, gains over a real coding harness, or GPU speedups. Extend the corpus and review
labels before model selection or performance claims.

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-default-features --features metal
```

Tests cover metric arithmetic, missing results, RRF ranks/ties, invalid labels, FP16 IPC and
incremental reuse, filtered row mapping against scalar cosine, provider guards, shared query
encoding across precisions and CLI JSON/no-overwrite behavior. Default tests need no model files.
