# cubeql

A minimal Arrow-native retrieval engine powered by CubeCL.

The current prototype intentionally implements one complete path only:

- input: Apache Arrow `RecordBatch`;
- vectors: `FixedSizeList<Float32>` or `FixedSizeList<Float16>`;
- query: exact inner product, cosine similarity, or squared L2 distance;
- optional predicate: `Float32`/`Float16` column `>= min`;
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

The default feature uses CubeCL's CPU runtime so tests work without a GPU. Select another runtime
when building an application, for example `--no-default-features --features cuda` or
`--no-default-features --features wgpu`.
