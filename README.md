# cubeql

A minimal Arrow-native retrieval engine powered by CubeCL.

The current prototype intentionally implements one complete path only:

- input: Apache Arrow `RecordBatch`;
- vectors: `FixedSizeList<Float32>` or `FixedSizeList<Float16>`;
- query: exact inner product;
- optional predicate: `Float32`/`Float16` column `>= min`;
- execution: CubeCL (CPU by default, with CUDA/HIP/Metal/WGPU feature switches);
- top-k: deterministic CPU selection after CubeCL device scoring.

```rust
let hits = cubeql::query(
    &cubecl::Device::default(),
    &batch,
    &[1.0, 0.0, 0.0],
    cubeql::QueryOptions {
        vector_column: "embedding",
        top_k: 10,
        filter: Some(cubeql::FloatFilter {
            column: "quality",
            min: 0.5,
        }),
    },
)?;
```

Run the end-to-end FP32 and FP16 tests with:

```bash
cargo test
```

The default feature uses CubeCL's CPU runtime so tests work without a GPU. Select another runtime
when building an application, for example `--no-default-features --features cuda` or
`--no-default-features --features wgpu`.
