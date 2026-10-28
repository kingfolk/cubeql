# Snapshot-local hybrid code retrieval

Milestone 7 adds explicit semantic, lexical, syntax-symbol and hybrid strategies to the library
and CLI. The generic `cubeql` crate is unchanged: all code-specific adapters, fusion, explanations
and overlap suppression live in `cubeql-code`. No new model or GPU kernel is introduced.

## CLI

Any existing code snapshot can be searched without rebuilding its Arrow data:

```bash
# No model loading or device initialization, even for a MiniLM snapshot.
cargo run -p cubeql-code -- search /tmp/benchmark-fp32.arrow \
  "read configuration file" --strategy lexical --language rust --top-k 5 --format json

cargo run -p cubeql-code -- search /tmp/benchmark-fp32.arrow \
  load_config --strategy symbol --format json

# The model directory is needed only for semantic/hybrid retrieval.
cargo run -p cubeql-code --features local-embeddings -- search /tmp/benchmark-fp32.arrow \
  "Where do we read application settings from a file?" \
  --strategy hybrid --model-dir /tmp/cubeql-minilm \
  --language rust --candidate-k 24 --top-k 5 --format json
```

`--strategy semantic` remains the default; `query` is still an alias. There is no automatic query
router. Select a strategy explicitly to inspect its behavior. `--candidate-k` is the per-source
budget before fusion/deduplication; by default it is `max(50, top_k)`, and it must be at least
`top_k`. `--no-dedup` preserves overlapping results for inspection. A zero `top_k` returns no
results without model/device initialization; empty search text is rejected.

Language/path filters apply to every branch, before candidate truncation. They retain the same
case-sensitive equality/literal-prefix semantics as before. Search only uses saved source, not
the live checkout. This prevents a text search from returning new files or line numbers that do
not belong to the resident vector snapshot. Updated snapshots require fresh resident/text indexes.

## Adapters

### Lexical: exact phrase plus identifier-aware keywords

The snapshot adapter is an in-process inverted index, not a shell call to ripgrep. It provides
literal substring search plus token retrieval without requiring a separately installed binary.
It does not implement regex, ripgrep CLI compatibility, BM25 or filesystem crawling.

Terms are extracted from each chunk's path, symbol and source. They are Unicode-alphanumeric
identifiers with underscores, normalized to lowercase. Original identifiers and their snake_case/
CamelCase/acronym components are indexed once per chunk. For example, `load_config` produces
`load_config`, `load`, `config`; `HTTPParser` also produces `http` and `parser`. A fixed small list
of English/Rust glue words is ignored. There is no stemming or synonym expansion.

Each query uses distinct terms. For term document frequency `df` and corpus chunk count `N`:

```text
idf(term) = ln((N + 1) / (df + 1)) + 1
coverage = sum(idf of matched query terms) / sum(idf of all query terms)
lexical_score = coverage + (1 if the exact query substring is present in source, else 0)
```

Coverage uses OR matching, favoring broader matches and rarer terms. The document frequencies
are snapshot-global, not recomputed for each metadata filter. Exact substring matching is
case-sensitive and scans allowed saved chunks; token matching is case-insensitive. Punctuation-
only queries can still produce exact source matches. Ties use ascending snapshot row IDs.
The postings accelerate keyword lookup, but the exact-phrase branch still scans source.

Unlike the M6 whole-query `literal` baseline, a query such as `needle api` can now match a chunk
containing `needle_api` even when the complete query sentence never appears in the code.

### Symbol: reuse the syntax index

The adapter indexes syntax-derived qualified and unqualified symbols already saved by the Rust
chunker. Exact names are case-sensitive. It checks the entire query and identifier-like atoms
within a sentence. Full-symbol matches score 3, whole-query unqualified matches score 2, and
symbol atoms in a longer query score 1. Equal scores use ascending row IDs.

`Config::load` does not degrade to an unqualified `load` query and accidentally retrieve
`Other::load`. Generic/trait symbol spelling follows the existing chunker's representation.
Line-fallback chunks are not treated as syntax symbols, so Python and other fallback languages
remain searchable lexically/semantically but do not acquire fictitious symbol support. This is
not an LSP or cross-reference/call-graph index; no fuzzy/prefix symbol matching is implemented.

### Semantic and hybrid

Semantic retrieval reuses the existing typed-query/model guard and exact device cosine top-k.
Hybrid requests run semantic, lexical and symbol branches with the same filter and candidate
budget, then fuse their lists using equal-weight RRF:

```text
hybrid_score(row) = sum(1 / (60 + one_based_source_rank(row)))
```

An absent row contributes zero. The existing evaluator and query API share one RRF
implementation. Raw cosine/lexical/symbol scores are not added or directly compared. The union
can contain up to three times the per-source budget before deduplication and final top-k.
There is no identifier pinning, score calibration, reranker or guarantee that fusion improves
relevance. Correlated lexical/symbol evidence may double-reward the same name; inspect provenance
and benchmark before selecting hybrid as a default.

## Output and overlap suppression

Search JSON uses `format_version: 2`. Existing location/source fields remain flattened in each
hit, with new `strategy`, strategy-dependent `metric`, and per-hit `provenance` entries:

```json
{
  "source": "lexical",
  "rank": 1,
  "raw_score": 1.5
}
```

The example score is illustrative. Sources can be `semantic`, `lexical`, or `symbol`. Each rank
is within its filtered source candidate list, before final fusion and overlap suppression. The
top-level hit score is cosine/lexical/symbol for standalone modes and RRF for hybrid. It is not
a confidence probability. Host-only modes report `backend: "Host"` and zero model/embedding/
device timings. Human-readable output also includes source ranks.

After ranking, greedy suppression keeps the first result when another result on the same path
has inclusive line-range intersection-over-union (IoU) >= 0.8. It runs before final truncation so
duplicate ranges do not waste top-k slots. Small headers contained in much larger functions
are not automatically duplicates. Distinct functions on the same source line can be suppressed;
the retained result records the other row, symbol, path, lines and original provenance in
`suppressed_overlaps`. Suppressed evidence is not falsely attributed to the representative row.
No source text is rewritten or deleted. Disable suppression with `--no-dedup` when separate
same-line symbols matter. Locations and row IDs are valid for the saved snapshot, not stable
identifiers across arbitrary future edits.

The Arrow snapshot format is unchanged. Index/update/inspect CLI envelopes retain their prior
format; the nested evaluation report also advances to version 2 and adds new strategy results.
The original `SearchResponse` library type remains available; the CLI now uses `RetrievalResponse`.

## Library composition

Host-only search needs no provider or device:

```rust
use cubeql_code::{CodeIndex, reporting::CodeFilter,
    retrieval::{RetrievalOptions, SearchStrategy, SnapshotTextIndex, materialize_hits}};

let code = CodeIndex::load_ipc("/tmp/benchmark-fp32.arrow")?;
let text_index = SnapshotTextIndex::new(&code);
text_index.warm_up(SearchStrategy::Lexical);
let prepared = text_index.prepare(&CodeFilter::default(), None)?;
let result = prepared.query("read configuration", None, RetrievalOptions {
    strategy: SearchStrategy::Lexical,
    ..Default::default()
})?;
let hits = materialize_hits(&code, &result.hits, false)?;
```

For hybrid retrieval, construct a `CodeSearchIndex`, borrow its `code()` to construct
`SnapshotTextIndex`, and pass `Some(&resident)` to `prepare`. Pass a compatible `EmbeddedQuery`
to `query`. The same prepared selection can serve repeated queries, and the same text index
reuses its lazily built postings/symbol maps. `warm_up(strategy)` builds the needed adapters
outside query timers; semantic-only mode builds neither map.

The text and semantic adapters must reference the **same CodeIndex object**, not independent
clones with matching row counts. This prevents accidental cross-snapshot vector/source mixing.
Both adapters are bound to identical filters by `prepare`. The semantic branch still checks the
full embedding configuration before scoring, including for empty selections. The caller must
embed the same text passed to the lexical/symbol branches; tagged vectors verify model identity,
not correspondence with an arbitrary text string supplied separately by the caller.

## Evaluation and limits

The existing `evaluate` command now reports six strategies on the same labels and shared query
embeddings: original `literal`, `semantic`, original two-way `experimental_rrf`, plus current
`lexical`, `symbol`, and three-way `hybrid`. The old baselines remain unchanged to show whether
gains come from stronger adapters or from fusion. Original semantic/literal baselines do not
perform the new overlap suppression; the current retrieval APIs do. The fixture has disjoint
function ranges, so this distinction does not affect its quality comparison.

The current CLI calls `hybrid_evaluation::evaluate`. It first runs the unchanged
`evaluation::evaluate` baseline through a provider wrapper that captures the query vectors, then
reuses those vectors for the current strategies. Duplicate query texts are captured per case, not
silently collapsed. Baseline and current variants run in separate passes, so their index conversion/
upload costs are reported separately (`baseline_conversion_ms`, `baseline_upload_ms`, and current
`conversion_ms`, `upload_ms`). Only one resident vector index is needed at a time, but host baseline
snapshots and report results remain allocated. This is not a minimal-memory benchmark. Later
strategies can reuse JIT/runtime state from the baseline; no fresh-process cold-start comparison
is claimed. Library callers can still run the original M6 evaluator independently.

On the existing 24-chunk/8-query MiniLM smoke dataset, local CPU and native Metal runs with
`candidate_k=24`, 3 repeats and paired FP32/FP16 produced the same quality across both devices
and storage precisions:

| Strategy | Recall@5 | Recall@10 | MRR@24 |
| --- | ---: | ---: | ---: |
| M6 literal | 0.2500 | 0.2500 | 0.2500 |
| Semantic | 0.9375 | 1.0000 | 1.0000 |
| M6 two-way RRF | 0.9375 | 1.0000 | 1.0000 |
| Current lexical | 1.0000 | 1.0000 | 1.0000 |
| Symbol | 0.2500 | 0.2500 | 0.2500 |
| Current hybrid | 1.0000 | 1.0000 | 1.0000 |

Hybrid preserves identifier performance and improves cross-file Recall@5 from 0.75 to 1.0 in
this tiny dataset. **Lexical-only also reaches 1.0, so this is not evidence of an additional
RRF benefit.** Conceptual queries already saturate the metrics under semantic retrieval; no
conceptual improvement is demonstrated. The fixture is synthetic and comment-rich, not a
representative workload. The milestone's broader relevance acceptance criterion remains open.

Evaluation separates host adapter preparation (`text_index_ms`) from repeated retrieval when
explicitly warmed. Current strategy timers include the complete retrieval call, including branch
work, evidence construction and deduplication, but exclude query embedding, index loading and
materialization. Search reports individual branch/fusion/dedup timers as well as total retrieval;
individual timers are not an exhaustive partition of the total. Upload/filter timers retain the
existing host-submission caveats. No GPU speedup or isolated serving-latency claim is made.

Postings and symbol maps are extra in-memory host caches, rebuilt from the snapshot, not persisted
in Arrow. They are not included in the existing Arrow buffer/vector-size accounting or IPC size.
No total/peak RSS measurement is added. Larger corpora, reviewed labels, and controlled optimized
benchmarks are still needed before choosing hybrid over lexical/semantic retrieval in production.

Tests cover identifier tokenization, qualified symbols, saved versus live source, source-rank
arithmetic, filtered provenance, model/snapshot guards, overlap suppression/refill/opt-out, host
CLI operation with unavailable models, shared query embeddings and FP32/FP16 behavior. A
precomputed-vector test deliberately misranks an identifier to verify recovery via complementary
signals; it is a mechanism test, not a learned-model quality claim.

```bash
cargo test --workspace
cargo test --workspace --no-default-features --features metal
cargo clippy --workspace --all-targets --features local-embeddings -- -D warnings
cargo check --workspace --all-targets --all-features
```
