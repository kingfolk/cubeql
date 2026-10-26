# Repository Code Index Builder

Milestone 4 adds the `cubeql-code` library and CLI. It converts a local repository into a
single Arrow batch that the existing CubeQL vector engine can load, filter and query.
Building an index does not initialize a CubeCL device.

## Scope and embedding caveat

The CLI defaults to **deterministic signed token hashing**, not a trained embedding model. It is a
reproducible offline fixture and lexical baseline: it verifies walking, chunking, incremental
reuse, persistence, and vector retrieval, but does **not** demonstrate semantic code-search
quality. Different tokens can collide, and synonyms are not understood. Natural-language search
requires a learned provider rather than the default hash fixture.
On a large corpus, collisions can dominate even exact-identifier queries; the runnable example
is a pipeline check, not a substitute for identifier search or a relevance benchmark.

Milestone 5 adds an optional learned MiniLM provider, batch encoding, full model metadata and
guarded text queries. See [embedding-providers.md](embedding-providers.md) for that workflow.
The low-level embedding callback remains supported, alongside the recommended provider API.
The builder emits FP32 vectors by default. Milestone 6 adds `index --precision fp16`, filtered
search/JSON, snapshot updates and labeled evaluation. See
[code-search-evaluation.md](code-search-evaluation.md) for that workflow.

## CLI

From the CubeQL checkout:

```bash
cargo run -p cubeql-code -- index ../cubecl --output /tmp/cubecl-code-v1.arrow

cargo run -p cubeql-code -- index ../cubecl \
  --previous /tmp/cubecl-code-v1.arrow \
  --output /tmp/cubecl-code-v2.arrow
```

Choose fresh output paths: existing files are never overwritten, including the previous
snapshot. Reading/embedding errors leave the previous snapshot intact. Serialization happens
before creating the new file; an I/O failure during the final write can leave a partial new file.
The command prints file/chunk counts, embedding calls, reuse, additions, changes and removals.

Options:

- `--dimensions`: token-hash vector width (default 128).
- `--fallback-lines`: non-overlapping fallback chunk size (default 80 lines).
- `--max-file-bytes`: maximum source file size (default 2 MiB).
- `--previous`: a compatible code-index IPC snapshot used as the embedding cache.
- `--embedding hash|minilm`, `--model-dir`, `--batch-size`: optional provider configuration;
  see the embedding-provider guide. MiniLM's vector width is fixed at 384.

## Walking and source extraction

The `ignore` walker respects `.gitignore`, `.ignore`, repository Git excludes, parent ignore
files, and configured global Git ignores. `.gitignore` also works outside Git repositories.
Hidden entries are skipped, as are `target`, `node_modules`, `.git`, and `.venv` directories.
Symbolic links are not followed. Only recognized source/document extensions are included:
Rust, Python, JavaScript/TypeScript, Go, C/C++, Java, Kotlin, Swift, Ruby, shell, SQL,
TOML/JSON/YAML, Markdown and text. Unknown extensions and extensionless files are skipped.
UTF-8 paths are required. Non-UTF-8/NUL-containing files and oversized files are skipped with
warnings; other traversal/read errors fail the build rather than silently hiding them.

Files are processed in sorted repository-relative path order. Rust uses `syn` without compiling
the source, resolving dependencies, expanding macros or executing a build script. Inactive
`cfg` branches are included too. A broken/in-progress Rust file falls back to line chunks with
a warning. Comment-only Rust also falls back to retain its text.

Rust chunks include functions, methods, structs/enums/unions, aliases, constants/statics,
module declarations, trait items and macro definitions/invocations. Inline modules, impl blocks
and traits get header-only chunks, followed by separate child chunks. This avoids duplicating
whole child bodies. Functions remain whole even if long; model-specific token limits and
splitting oversized syntax units are not implemented yet. Items nested inside function bodies
remain part of the enclosing function. Macro-generated functions are not individually extracted.

Qualified symbols include inline module and impl/trait context, e.g.
`storage::Index::query` or `storage::<Index as Search>::query`. These are display labels, not
compiler-resolved identifiers. Fallback chunks use labels such as `lines:81-160`.

## Arrow schema

| Column | Type | Meaning |
| --- | --- | --- |
| `path` | Utf8 | Repository-relative path |
| `language` | Utf8 | Language inferred from extension |
| `symbol` | Utf8 | Qualified Rust symbol or fallback line label |
| `kind` | Utf8 | `function`, `method`, `struct`, `module`, `lines`, etc. |
| `start_line`, `end_line` | UInt32 | Inclusive, one-based source range |
| `content_hash` | Utf8 | BLAKE3 of the exact chunk source bytes |
| `source` | Utf8 | Original chunk text, including Rust doc attributes where present |
| `embedding` | FixedSizeList<Float32> | Finite vector of the configured width |

Metadata records the code-format version, chunker version, fallback size, embedding identity,
dimensions and canonical repository root. Files use the existing CubeQL IPC envelope, with
`embedding` as the vector column and no additional resident normalization. The new core
`read_ipc_batch` / `write_ipc_batch` functions share this envelope with `VectorIndex` without
uploading anything. The code-index reader additionally validates its metadata, columns, source
hashes, line ranges and vector values before allowing reuse.

Line ranges identify where to find source in this snapshot, not necessarily current working-tree
content after subsequent edits. Row IDs are also snapshot-local; adding/removing chunks can
renumber them. Keep the `CodeIndex` metadata and `VectorIndex` batch from the same snapshot.

## Incremental indexing

1. Read and chunk all currently included files again. No mtime-only shortcuts are used.
2. Match previous chunks by `(path, source hash)`, consuming duplicate occurrences separately.
3. Reuse matching vectors even if line numbers or row positions shifted.
4. Invoke the callback only for unmatched chunks and validate dimensions/finite values.
5. Write a complete new Arrow snapshot containing only the current chunks.

The callback receives source text only. Its `embedding_id` must encode every model, weight,
version and preprocessing choice that can change the resulting vector. Changing the identity,
dimensions or fallback chunk size rejects incremental reuse rather than mixing incompatible
vectors. Omit `--previous` / pass `None` to rebuild with a different configuration.

Reports classify unmatched new chunks by `(path, kind, symbol)`: an old unmatched slot means
`changed`, otherwise `added`; old slots left over are `removed`. Consequently, renaming a Rust
symbol usually counts as removal plus addition. A file rename is also removal plus addition;
cross-path reuse is not attempted. Line-based fallback boundaries can shift after insertions
and invalidate more than one chunk. Empty chunks are omitted; unchanged repositories perform
zero embedding calls, not zero filesystem/parsing work. Every update rewrites the full IPC file.

## Library usage and querying

```rust
use cubeql::{Filter, Metric, VectorIndex};
use cubeql_code::{BuildOptions, build_repository, hash_embedding};

let options = BuildOptions::default();
let built = build_repository("../cubecl", &options, None, |source| {
    hash_embedding(source, options.dimensions)
})?;
built.index.save_ipc("/tmp/new-code-index.arrow")?;

let device = cubecl::Device::default();
let vectors = VectorIndex::new(&device, built.index.batch().clone(), "embedding")?;
let query = hash_embedding("TriviallyUnrollable", options.dimensions)?;
let hits = vectors.query_filtered(&query, 10, Metric::Cosine, &[
    Filter::StringEq { column: "language", value: "rust" },
    Filter::PathPrefix { column: "path", prefix: "crates/" },
])?;
for hit in hits {
    let chunk = &built.index.chunks()[hit.row];
    println!("{}:{}-{} {} ({})", chunk.path, chunk.start_line,
             chunk.end_line, chunk.symbol, hit.score);
}
Ok::<(), Box<dyn std::error::Error>>(())
```

The example above demonstrates the low-level, callback-only API. To use precomputed embeddings, replace the callback and set a corresponding
`embedding_id` and `dimensions`; embed queries using the same configuration. Core
`VectorIndex` intentionally validates query dimensions, not model identity. The provider-aware
`CodeSearchIndex` additionally validates the complete embedding contract. Metadata filtering,
exact metrics and device top-k remain unchanged. The walker, parser, hashing and callback all
run on the host; CubeCL is used for retrieval after the snapshot is loaded onto a device.

## Verification

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
# Apple native Metal, including code-index-to-query tests:
cargo test --workspace --no-default-features --features metal
```

Tests cover Unicode/CRLF and exact source locations, Rust container/method extraction,
ignore rules and symlinks, fallback boundaries, malformed/binary/oversized files, incremental
add/change/remove and line shifts, duplicate chunks, empty indexes, IPC roundtrips, CLI reuse,
overwrite protection, bad embeddings/configuration/manifests, and end-to-end filtered retrieval
with all three metrics checked against a CPU reference. They validate pipeline correctness,
not semantic relevance quality. Index construction is currently an in-memory, single-batch
operation; it is not a streaming or low-memory repository indexer.

A small runnable query example is provided for the built-in lexical fixture:

```bash
cargo run -p cubeql-code --example query -- /tmp/cubecl-code-v1.arrow "TriviallyUnrollable"
cargo run -p cubeql-code --no-default-features --features metal --example query -- \
  /tmp/cubecl-code-v1.arrow "TriviallyUnrollable"
```

It prints five hits with source ranges, symbols and cosine scores, and refuses indexes with a
different embedding identity. CPU and Metal can load the same IPC snapshot; the example's
query vectors and scoring configuration are identical on both backends.
