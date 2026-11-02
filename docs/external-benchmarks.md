# External benchmark import and ranking export

`cubeql-code benchmark` bridges a canonical corpus/query/qrels dataset to CubeQL's exact
cosine engine. It preserves external document units and IDs, so a benchmark's own evaluator
can score the results. It is separate from `evaluate`, which still uses code-specific
`(path, symbol)` labels. Core `cubeql` and the existing code-search algorithms are unchanged.

This first adapter supports **BEIR-style JSONL plus headered TSV**, a shared interchange
format used by code-retrieval evaluations such as CoIR. It does not download datasets or
convert every benchmark's native release. See the upstream
[BEIR loader](https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py) and
[CoIR](https://github.com/CoIR-team/coir) for their formats and evaluation protocols.

## Input contract

A dataset directory contains:

```text
dataset/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

`corpus.jsonl` has one JSON object per document:

```json
{"_id":"doc-config","title":"Configuration","text":"fn load_config() { read_file(); }"}
{"_id":"doc-cache","text":"fn cache_value() { store(); }","metadata":{"language":"rust"}}
```

`queries.jsonl` has one object per query:

```json
{"_id":"q-config","text":"Where are application settings read from disk?"}
```

`qrels/test.tsv` has literal tabs and an exact header:

```text
query-id	corpus-id	score
q-config	doc-config	2
q-config	doc-cache	0
```

- IDs are opaque UTF-8 strings: they never become filesystem paths or syntax symbols.
  They must be nonempty and contain no whitespace/control characters, so they are safe
  as TREC columns. Document IDs and query IDs have separate namespaces and may coincide.
- Corpus and query IDs must each be unique. Duplicate qrels pairs, missing references,
  malformed JSON/TSV and non-integer grades fail validation. Error messages identify the
  file and line for malformed records. Blank JSONL/TSV records are rejected.
- `title` is optional or null; unknown JSON fields such as `metadata` are ignored.
  With a nonempty title, embedding input is exactly `title + "\n" + text`; otherwise it
  is `text`. Provider-specific preprocessing still applies and is recorded separately.
- Each input document remains **one Arrow row**. There is no parsing, rechunking,
  path-based filtering, deduplication or symbol synthesis. A benchmark document can be
  a function, file or prebuilt chunk; its size is not assumed to be a source-file size.
- Grades, including zero and negative integers, are preserved. This adapter applies no
  relevance threshold. A header-only qrels file is valid; a missing/empty file is not.
- Every input query is executed, even if it has no qrels or only zero grades. Missing
  judgments do **not** mean that the query has no answer. Native evaluators may select a
  judged subset; record that choice instead of changing the retrieval inputs silently.
- `--split` selects `qrels/<split>.tsv` and accepts only ASCII letters, digits, `_`, `-`.

The input files stay outside the index's content. Only corpus title/text are embedded;
query text and qrels cannot enter document vectors through this adapter.

## Validate without inference

```sh
cargo run -p cubeql-code -- benchmark validate \
  crates/cubeql-code/tests/fixtures/external-benchmark
```

The JSON output reports document/query/judgment counts, judged and unjudged query counts,
the grade histogram, and BLAKE3 fingerprints of the exact parsed bytes. Validation loads
neither an embedding model nor a compute device. It does not judge annotation quality.

## Run an offline pipeline smoke test

```sh
cargo run -p cubeql-code -- benchmark run \
  crates/cubeql-code/tests/fixtures/external-benchmark \
  --output /tmp/cubeql-external-smoke --dimensions 32 --top-k 3
```

The default is deterministic hash embeddings and stderr identifies them as a fixture.
This exercises file loading, Arrow/CubeCL execution and export, **not semantic quality**.
The checked-in three-document/three-query corpus is synthetic test data, not a public
benchmark score. One query intentionally has no judgment to test preservation.

Use the real local provider for quality experiments:

```sh
cargo run --release -p cubeql-code --features local-embeddings -- benchmark run \
  /path/to/frozen-dataset --split test \
  --output /tmp/cubeql-external-minilm \
  --embedding minilm --model-dir /path/to/pinned-minilm \
  --batch-size 8 --top-k 100 --run-tag cubeql-minilm
```

MiniLM loads existing validated local artifacts; this command performs no downloads.
Only inference runs on CPU; choose another existing backend feature for retrieval, for
example `--no-default-features --features metal,local-embeddings`. This is not a measured
CPU/Metal comparison. Library callers can use another `EmbeddingProvider` or precomputed
document/query vectors without changing the generic engine.

`--output` must be a **new directory with an existing parent**. Successful runs create:

| File | Contents |
|---|---|
| `dataset.json` | Counts, grade histogram and exact-byte input fingerprints |
| `qrels.json` | Original query/document IDs and unchanged grades as nested JSON |
| `request.json` | Requested provider, options, split and declared revision, retained even if model loading fails |
| `preparation.json` | Loaded model contract, model load time, package version, executable BLAKE3 and optional declared engine revision, recorded before inference |
| `index.arrow` | Generic CubeQL FP32 index with `document_id`, `title`, `text`, `embedding`, corpus hash and model contract |
| `results.json` | Native score mapping `{query_id: {document_id: cosine_score}}` |
| `run.trec` | Six-column `query-id Q0 document-id rank score run-tag` export |
| `report.json` | Completion marker, full ordered rankings, parameters and per-phase timings |

Every file uses create-new semantics. Existing runs are never overwritten. After an
inference/export failure the CLI preserves partial artifacts and writes `failure.json`;
a retry needs a new directory. Input/option errors fail before a run directory is made.
Only a final `report.json` marks a completed export. Interrupted processes can leave an
incomplete directory without `failure.json`; do not treat that as a completed run.

Supply `--engine-revision <commit>` if known. It is explicitly caller-provided, not a
build attestation. The executable fingerprint distinguishes binaries even when their
package version is unchanged. Record dependency commits, dirty state, hardware and the
external evaluator revision with any published experiment. Output names do not imply
that the current checkout built an already-existing executable.

`index.arrow` uses the generic `cubeql::read_ipc_batch` / `VectorIndex` format, **not**
`CodeIndex`; code-specific `inspect/search/update` commands do not accept this document
index. Its schema metadata contains only corpus/model provenance, not queries or qrels.
Custom consumers must verify the stored embedding contract before querying it.

## Score through an external evaluator

`results.json` is the nested score dictionary consumed by BEIR-style evaluators and
`pytrec_eval`. For example, in a Python environment with `pytrec-eval-terrier` installed:

```python
import json
import pytrec_eval

with open("/tmp/cubeql-external-smoke/qrels.json") as f:
    qrels = json.load(f)
with open("/tmp/cubeql-external-smoke/results.json") as f:
    results = json.load(f)
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels, {"ndcg_cut.3", "recall.1,3", "recip_rank"}
)
per_query = evaluator.evaluate(results)
print(per_query)
```

This example illustrates interoperability, not a replacement scoring protocol. Follow
the selected benchmark's relevance threshold, graded gains, self-match handling, query
denominator, cutoffs and missing-result policy. Some datasets give hard negatives a
nonzero grade. Natural no-answer cases need a separate selective-retrieval evaluation;
this runner always returns nearest neighbors and implements no abstention rule.

Scores are unmodified FP32 cosine values; non-finite output is rejected. The engine
orders ties by corpus row ID. `report.json` and TREC ranks preserve that order, but an
external evaluator may re-sort equal scores using document IDs. Report that policy;
scores are not perturbed to manufacture a common tie order. `results.json` object-key
order does not encode ranks. Top-k is a maximum, not a relevance threshold.

## Scope and next steps

The public library entry points are `benchmark::Dataset::load` and `benchmark::run`.
`BenchmarkRun::save` writes to a caller-created directory with the same create-new guards.
The runner uploads corpus vectors once and reuses the index for every query. It records
document embedding, index setup/upload, query embedding and retrieval separately.
Retrieval is one observation per query with first-use work potentially included; it is
not a repeated warm latency benchmark, synchronized GPU transfer measurement or peak-RSS
experiment. Parsing, Arrow construction and serialization are outside retrieval timers.

The first version is FP32 exact cosine only. BM25, lexical/symbol/hybrid, reranking,
precomputed-vector CLI imports, snapshot reuse and benchmark-specific converters are
separate future work. All document metadata/text and vectors must fit in memory. A saved
index is not automatically reused on the next invocation.

For ARB/CORE-Bench releases that bind each query to a repository/base-commit corpus,
convert and run each permitted corpus group separately. Flattening all snapshots into
one global corpus would change the benchmark and may expose future or unrelated code.
Document any chunk-to-file aggregation. A selected pilot subset is not a full benchmark
score. Pin the official dataset/evaluator before running experiments; the adapter alone
does not establish a score on any public leaderboard.
