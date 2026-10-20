# cubeql

A minimal Arrow-native retrieval engine powered by CubeCL.

The current prototype intentionally implements one complete path only:

- input: Apache Arrow `RecordBatch`;
- vectors: `FixedSizeList<Float32>` or `FixedSizeList<Float16>`;
- query: exact inner product;
- optional predicate: `Float32`/`Float16` column `>= min`;
- reusable device-resident indexes for repeated queries;
- execution: CubeCL (CPU by default, with CUDA/HIP/Metal/WGPU feature switches);
- top-k: deterministic CPU selection after CubeCL device scoring.

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

Run the end-to-end FP32 and FP16 tests with:

```bash
cargo test
```

Compare index-loading, warm-query, and one-shot-query costs with:

```bash
cargo bench --bench resident_index
```

The default feature uses CubeCL's CPU runtime so tests work without a GPU. Select another runtime
when building an application, for example `--no-default-features --features cuda` or
`--no-default-features --features wgpu`.
