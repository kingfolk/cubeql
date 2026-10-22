# cubeql

A minimal Arrow-native retrieval engine powered by CubeCL.

The workspace also includes `cubeql-code`, a repository-to-Arrow index builder with Rust
syntax-aware chunks, line-based fallbacks and incremental embedding reuse:

```bash
cargo run -p cubeql-code -- index ../cubecl --output /tmp/cubecl-code-v1.arrow
cargo run -p cubeql-code -- index ../cubecl --previous /tmp/cubecl-code-v1.arrow \
  --output /tmp/cubecl-code-v2.arrow
```

The CLI currently uses deterministic token-hash vectors for offline pipeline validation, **not
a semantic embedding model**. The library accepts an embedding callback for real/precomputed
vectors. See [docs/code-index-builder.md](docs/code-index-builder.md) for the schema, incremental
rules, limitations and an end-to-end query example.

The current prototype intentionally implements one complete path only:

- input: Apache Arrow `RecordBatch`;
- vectors: `FixedSizeList<Float32>` or `FixedSizeList<Float16>`;
- query: exact inner product, cosine similarity, or squared L2 distance;
- metadata predicates: floating-point equality/ranges, string equality/set membership, path prefixes;
- reusable selections with AND semantics, compact scoring or in-kernel masks;
- versioned Arrow IPC persistence preserving metadata and normalization options;
- reusable device-resident indexes for repeated queries;
- execution: CubeCL (CPU by default, with CUDA/HIP/Metal/WGPU feature switches);
- top-k: exact device-side selection; only the final k row IDs and scores are read back.

For repeated queries, construct a `VectorIndex` once so the embedding column remains on the
selected device:

```rust
let device = cubecl::Device::default();
let index = cubeql::VectorIndex::new(&device, batch, "embedding")?;

let hits = index.query(
    &[1.0, 0.0, 0.0],
    cubeql::IndexQueryOptions {
        top_k: 10,
        filter: Some(cubeql::FloatFilter {
            column: "quality",
            min: 0.5,
        }),
    },
)?;
```

The original `query` function remains available as a convenience for one-shot queries. It builds
a temporary `VectorIndex` internally.

Select a metric explicitly when querying a resident index:

```rust
let hits = index.query_with_metric(
    &[1.0, 0.0, 0.0],
    cubeql::IndexQueryOptions::new(10),
    cubeql::Metric::Cosine,
)?;
```

`InnerProduct` and `Cosine` return the largest scores first. `SquaredL2` returns the smallest
distances first (scores are distances, not their negation). Equal scores are ordered by ascending
row ID, with signed zeros treated as equal. Cosine involving a zero vector is defined as zero.
Inputs must be finite; FP32 arithmetic can still overflow, in which case NaN scores are omitted
and infinite scores remain eligible.

Optional index normalization is available through
`VectorIndex::new_with_options(&device, batch, "embedding", IndexOptions { normalize: true })`.
It normalizes the resident vectors, preserving FP16/FP32 storage and the original Arrow batch.
Queries are used as supplied, so normalize them too if unit-length inner products are desired.

The portable top-k implementation uses repeated selection within tiles followed by merge stages.
It is intended for small k: selection costs approximately O(N * k), and requesting every row can
be expensive. Scoring and temporary buffers remain O(N); the host readback is at most 8 * k bytes.

Prepare a selection once for repeated metadata-filtered queries:

```rust
let filters = [
    cubeql::Filter::StringEq { column: "language", value: "rust" },
    cubeql::Filter::PathPrefix { column: "path", prefix: "crates/opt/" },
    cubeql::Filter::FloatRange { column: "quality", min: Some(0.8), max: None },
];
let selected = index.prepare_filter(&filters, cubeql::FilterMode::BeforeScoring)?;
let hits = selected.query(&query_vector, 10, cubeql::Metric::Cosine)?;
```

Metadata is evaluated on the host. Compact mode uploads only matching original row IDs; mask
mode uploads one UInt32 per row and checks it in the scoring kernel. Strings remain in Arrow
host memory. The selection is bound to its index and reused without re-evaluating predicates.
Null metadata never matches, and range endpoints are inclusive. The original `FloatFilter`
API retains its existing FP16/FP32 and null-rejection behavior.

Save and restore an index with:

```rust
index.save_ipc("code-index.arrow")?; // Fails if the file already exists.
let restored = cubeql::VectorIndex::load_ipc(&device, "code-index.arrow")?;
```

The file contains the original Arrow batch and index configuration, not device pointers. Loading
uploads vectors once to the chosen backend. See [docs/metadata-persistence.md](docs/metadata-persistence.md)
for supported types, format details, and the filtering benchmark.

Run the end-to-end FP32 and FP16 tests with:

```bash
cargo test
```

Compare index-loading, warm-query, and one-shot-query costs with:

```bash
cargo bench --bench resident_index
```

Compare direct CPU scoring/partial selection with CubeCL scoring/device selection across dataset
sizes (128 dimensions, top-10):

```bash
cargo bench --bench metrics_top_k
# Native Metal on Apple hardware:
cargo bench --no-default-features --features metal --bench metrics_top_k
```

These benchmarks include scoring, selection, and final readback, and exclude index loading.
See [docs/metrics-top-k.md](docs/metrics-top-k.md) for the selection algorithm, numeric rules,
and initial CPU/Metal measurements.

Compare compact filtering with in-kernel masks at different selectivities:

```bash
cargo bench --bench metadata_filters
cargo bench --no-default-features --features metal --bench metadata_filters
```

The default feature uses CubeCL's CPU runtime so tests work without a GPU. Select another runtime
when building an application, for example `--no-default-features --features cuda` or
`--no-default-features --features wgpu`.
