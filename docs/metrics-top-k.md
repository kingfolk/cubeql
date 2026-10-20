# Metrics and Device-Side Top-K

Milestone 2 adds three exact scoring metrics, optional index normalization, and device-side
selection. Embeddings stay resident in `VectorIndex`; the host receives only selected results.

## API

```rust
let index = cubeql::VectorIndex::new_with_options(
    &device,
    batch,
    "embedding",
    cubeql::IndexOptions { normalize: false },
)?;

let hits = index.query_with_metric(
    &query_vector,
    cubeql::IndexQueryOptions::new(10),
    cubeql::Metric::Cosine,
)?;
```

Existing `VectorIndex::new`, `VectorIndex::query`, and the one-shot `cubeql::query` remain
available. Their defaults are unchanged: no normalization and inner-product scoring.

## Scoring and Numeric Rules

| Metric | Score | Order |
| --- | --- | --- |
| Inner product | sum(x[i] * q[i]) | Largest first |
| Cosine | dot(x, q) / (norm(x) * norm(q)) | Largest first |
| Squared L2 | sum((x[i] - q[i])^2) | Smallest first |

All metrics accumulate in FP32, including FP16 vector inputs. Metric selection is a CubeCL
compile-time argument: kernels specialize for the chosen metric instead of branching on it
for every row at runtime.

Cosine scales each operand by its largest absolute component before computing norms. This
avoids overflowing or underflowing the squared norms for large or tiny normal FP32 values.
Cosine involving a zero vector is defined as zero, and results are clamped to [-1, 1].

Index normalization is optional and runs once on the host. Norms are computed in FP64, and
normalized values are rounded back to the original FP16 or FP32 format before upload. Zero
vectors stay zero. The Arrow batch exposed by `batch()` retains the original vector values.
Queries are used as supplied; callers wanting unit-vector inner products must normalize their
queries too. Squared L2 on a normalized index measures distance to the normalized stored vectors.

Vector and query inputs must be finite. Inner-product or L2 arithmetic may still overflow in
FP32. Infinite scores remain eligible; NaN scores are excluded. Equal scores, including signed
zeros, are ordered by ascending row ID. Backend rounding can change near-ties; the row-ID rule
applies when computed scores are exactly equal.

## Selection Algorithm

The scoring kernel writes one FP32 score and one UInt32 row ID per input row. Filtered rows or
NaN scores receive the reserved invalid row ID, `u32::MAX`. This separates validity from the
score itself, allowing legitimate negative-infinity inner products to remain in the results.

The selection kernel assigns one tile to each execution unit. For each tile it scans candidates
repeatedly, emitting the next best (score, row ID) pair until k slots have been written. Exhausted
tiles emit invalid IDs. The host queues further selection stages over these candidates until
only one list of k results remains.

For example, with 4096 rows and k = 10:

```text
score 4096 rows on device
    -> 16 tiles of 256 candidates
    -> 160 tile winners
    -> one final selection stage
    -> 10 globally ordered results
    -> read 10 scores and 10 row IDs
```

This is exact: every discarded candidate has at least k candidates ahead of it in its own tile,
so it cannot belong to the global top-k. Original row IDs survive every stage, keeping tie
ordering consistent across tile boundaries.

Tile size is max(256, 2*k), capped at the number of rows. This ensures progress between stages,
including when k exceeds 128. The implementation favors small k: work is approximately O(N*k)
across selection stages, so requesting nearly all rows can be expensive. It does not yet use
warp reductions, shared-memory sorting, or a specialized large-k algorithm.

Temporary device storage is O(N). Host readback is at most 8 * min(k, N) bytes: four bytes per
score and four per row ID. For 4096 rows and top-10, the logical readback payload falls from
16 KiB of full scores to 80 bytes of selected results. Runtime synchronization and allocation
costs still apply, and Apple unified memory does not imply a physical PCIe transfer.

## Validation

The new tests compare all metrics with a direct FP64 CPU reference. They cover FP16/FP32,
normalization on/off, filtering, zero vectors, extreme finite cosine operands, large k, empty
result sets, score ties, and original row IDs across three merge stages. Existing integration
tests also exercise the new selection path through the unchanged query APIs.

Both the CPU and native Metal backends passed all 12 integration tests locally.

```bash
cargo test
cargo test --no-default-features --features metal
cargo bench --bench metrics_top_k
cargo bench --no-default-features --features metal --bench metrics_top_k
```

## Initial Performance Measurements

Measured locally on an Apple M1, using FP32 vectors, 128 dimensions, and top-10. These are warm
queries over resident vectors. Times include scoring, selection, and final result retrieval,
but exclude index construction, model inference, and embedding generation. The host baseline is
a direct single-threaded Rust scan with partial selection; it is not an optimized BLAS or vector
database implementation. The short Criterion run used 10 samples and showed some timing noise.

| Rows | Direct Rust CPU baseline | CubeCL native Metal |
| ---: | ---: | ---: |
| 128 | 7.57 us | 588 us |
| 4096 | 242 us | 1.41 ms |
| 65536 | 3.85 ms | 2.72 ms |

At the largest sampled size, Metal was approximately 1.4x faster than this CPU reference.
The sampled crossover lies between 4096 and 65536 rows; this does not establish an exact
threshold or predict behavior for other dimensions, k values, metrics, or hardware.

### CubeCL CPU Versus Native Metal

A follow-up run compared the CubeCL CPU and native Metal backends directly, executing them
sequentially with the same benchmark, data, and selection algorithm. This comparison does not
use the direct Rust CPU baseline. The workload remains FP32, 128 dimensions, inner product,
and top-10 on the same Apple M1, with 10 Criterion samples per case.

| Rows | CubeCL CPU | CubeCL native Metal |
| ---: | ---: | ---: |
| 128 | 472 us | 586 us |
| 4096 | 1.27 ms | 1.48 ms |
| 65536 | 6.31 ms | 2.45 ms |

Metal was approximately 2.57x faster than CubeCL CPU at 65536 rows in this run. CPU had lower
point estimates at the smaller sizes, although timing noise was substantial: at 4096 rows,
Criterion's estimated intervals were 1.06–1.60 ms for CPU and 1.40–1.63 ms for Metal, so the
small-input differences should not be treated as a firm performance guarantee.

These are full warm-query timings: query upload, inner-product scoring, device-side top-k,
and final result retrieval. They exclude index construction and embedding upload. They do not
isolate the top-k kernel, so the 2.57x ratio describes the full query rather than selection alone.

Reproduce the backend comparison with:

```bash
cargo bench --no-default-features --features cpu \
  --bench metrics_top_k -- cubecl_device_top_k

cargo bench --no-default-features --features metal \
  --bench metrics_top_k -- cubecl_device_top_k
```

Device-side selection reduces the result payload but adds kernel and allocation overhead.
Reducing that overhead, using parallel reductions within each tile, and choosing a direct CPU
path for small workloads remain optimization opportunities. A separate selection-only benchmark
is needed to attribute time specifically to top-k.
