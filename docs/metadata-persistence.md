# Metadata Filtering and Arrow Persistence

Milestone 3 makes a resident vector index useful with code metadata and reusable across process
runs. The public API stays at the Rust function level; there is no query-language parser.

## Predicate Semantics

`prepare_filter` accepts a slice of predicates combined with AND. An empty slice selects every
row; an empty string-membership set selects no rows. Every predicate is validated even when an
earlier predicate has already eliminated all rows.

| Predicate | Supported columns | Matching rule |
| --- | --- | --- |
| `FloatEq` | Float16, Float32, Float64 | Exact numeric equality |
| `FloatRange` | Float16, Float32, Float64 | Inclusive min/max; either bound may be omitted |
| `StringEq` | Utf8, LargeUtf8, Utf8View, string dictionaries | Case-sensitive equality |
| `StringIn` | Same string types | Case-sensitive set membership |
| `PathPrefix` | Same string types | Literal prefix, without path normalization |

Float values are losslessly promoted to FP64 for predicate comparison. Equality has no tolerance:
for example, the FP16 representation of decimal 0.1 need not equal the FP64 literal 0.1.
Null metadata and NaN numeric values never match. NaN filter bounds and reversed ranges are
errors. Integer and decimal predicates, OR expressions, and filesystem path resolution are not
part of this version. Use a trailing slash when a path prefix should identify a directory.

String membership uses a host hash set. Dictionary-encoded columns compare each dictionary
value once and then inspect integer keys per row. Null dictionary keys or null dictionary
values do not match. All Arrow integer dictionary key types are supported.

The original `FloatFilter { column, min }` query option remains available. It continues to accept
only FP16/FP32 columns, rejects nulls as before, and uses the compact execution path internally.

## Prepared Selections

```rust
let filters = [
    cubeql::Filter::StringEq { column: "language", value: "rust" },
    cubeql::Filter::StringIn { column: "repository", values: &["cubecl", "burn"] },
    cubeql::Filter::PathPrefix { column: "path", prefix: "crates/" },
    cubeql::Filter::FloatRange { column: "quality", min: Some(0.8), max: None },
];
let selection = index.prepare_filter(&filters, cubeql::FilterMode::BeforeScoring)?;
let first = selection.query(&first_embedding, 10, cubeql::Metric::Cosine)?;
let second = selection.query(&second_embedding, 10, cubeql::Metric::Cosine)?;
```

Preparation evaluates Arrow metadata on the CPU, combines the predicates into a mask, and
uploads the selection once. `PreparedFilter` borrows its index, preventing use with a different
index or backend. Predicates themselves do not have to outlive preparation. `len()` reports the
number of selected rows before scoring and top-k.

Two device execution strategies share the same predicate semantics:

- `BeforeScoring` uploads a compact list of matching row IDs. The scoring kernel runs only over
  those candidates and maps each position back to its original Arrow row. Score buffers and
  top-k stages scale with the selected count.
- `DuringScoring` uploads a UInt32 mask over all rows. The kernel launches over all positions but
  checks the mask before computing vector arithmetic. Score buffers and top-k input remain dense.

Both strategies skip vector arithmetic for rejected rows. The difference is compaction of work
and intermediate buffers, not whether a rejected row's dot product is calculated. String/range
expressions are evaluated during host preparation, not interpreted on the GPU.

The same selection works with all three metrics, FP16/FP32 embeddings, slices, and empty batches.
Results retain original row IDs and deterministic tie ordering. No-match selections return an
empty result without launching scoring or top-k. `query_filtered` is a convenience operation
that prepares a compact selection and runs one query; repeated queries should reuse preparation.

## IPC Format

`save_ipc` / `load_ipc` provide path-based persistence. `write_ipc` / `read_ipc` support writable
byte sinks and readable, seekable sources such as `Vec<u8>` and `Cursor<Vec<u8>>`.

Version 1 is an Arrow IPC **file**, containing exactly one RecordBatch and three schema metadata
keys reserved by CubeQL:

| Key | Value |
| --- | --- |
| `cubeql.format_version` | `1` |
| `cubeql.vector_column` | Embedding column name |
| `cubeql.normalize` | `true` or `false` |

The stored vectors are the original Arrow values. On load, the recorded normalization option is
applied once during device upload, producing the same resident values without normalizing an
already rounded vector a second time. Column types, dictionary columns, nulls, schema metadata,
field metadata, row order, and original vector values are preserved. Format keys are removed
from the loaded batch so `batch()` exposes the original user schema.

Loading reads Arrow data directly from the file; it does not re-chunk code or regenerate
embeddings. It still validates vectors and uploads them to the requested device. Device buffers,
kernel caches, prepared selections, and a default metric are not serialized. The caller chooses
the metric per query and prepares any filters again after loading. CPU-created files can be
loaded on Metal and other supported backends.

Missing/unknown format versions, invalid options, zero or multiple batches, malformed IPC, and
invalid vector schemas return errors. For a generic Arrow IPC file without the CubeQL metadata,
read its RecordBatch with Arrow and construct a `VectorIndex` explicitly. `save_ipc` uses exclusive
file creation and never overwrites an existing file. It serializes into a host buffer first;
writing failures can leave a partial new file. This is a snapshot format, not a transactional
storage engine. Parquet import is deferred until the code-index builder needs external datasets.

## Verification and Benchmarks

Integration tests compare combined predicates and both execution modes against a CPU reference
for all metrics. They cover nullable dictionary keys and values, FP16/FP32/FP64 metadata, all
supported plain string layouts, inclusive bounds, Arrow slices, empty/no-match selections,
normalization roundtrips, user metadata, invalid files/options, and existing-file protection.
An additional CPU-to-Metal persistence test runs when both features are enabled on Apple hardware.

```bash
cargo test
cargo test --no-default-features --features cpu,metal
cargo bench --bench metadata_filters
cargo bench --no-default-features --features metal --bench metadata_filters
```

The filtering benchmark uses 65536 resident FP32 vectors of 128 dimensions, inner product, and
top-10. A numeric predicate retains approximately 100%, 10%, or 1% of rows. It measures:

- An unfiltered warm query.
- Preparation for each strategy, including host evaluation and completed selection upload.
- Repeated queries with a prepared selection, including query upload, scoring, top-k, and readback.

Index construction and initial embedding upload are excluded. Preparation is not included in
warm-query timings; add its separately measured cost when considering a single-use filter.
String and dictionary predicate preparation costs are not represented by this numeric benchmark.

### Initial Apple M1 Measurements

The following point estimates come from sequential CPU and native Metal runs, using 10 Criterion
samples per case. Times are full warm queries, not isolated scoring or top-k kernel durations.

| Rows retained | CubeCL CPU compact | CubeCL CPU mask | Metal compact | Metal mask |
| ---: | ---: | ---: | ---: | ---: |
| 100% | 7.19 ms | 6.59 ms | 3.00 ms | 2.64 ms |
| ~10% | 1.36 ms | 1.95 ms | 1.78 ms | 1.86 ms |
| ~1% | 1.09 ms | 2.26 ms | 0.976 ms | 1.61 ms |

The unfiltered warm query measured 7.56 ms on CubeCL CPU and 2.69 ms on Metal in the same run.
At approximately 1% retention, Metal compact queries were about 1.65x faster than mask queries.
At 100% retention, compaction provided no measured benefit; all rows still need to be scored.

Preparation, measured separately with completed upload, cost approximately 0.15–0.17 ms on CPU
and 0.18–0.21 ms on Metal. At 1% retention specifically, compact preparation cost 0.154 ms on CPU
and 0.178 ms on Metal; mask preparation cost 0.152 ms and 0.193 ms, respectively. Reusing a
selection amortizes this work across queries.

Some CPU cases showed substantial noise: the 1% compact-query interval was 0.81–1.53 ms, and the
mask-query interval was 1.84–3.09 ms. These measurements establish the cost/benefit for this small
numeric workload, rather than a universal backend or strategy ranking. A smaller candidate set
can favor CPU even when unfiltered scans favor Metal. No automatic strategy/backend router is
implemented; callers choose the mode explicitly.
