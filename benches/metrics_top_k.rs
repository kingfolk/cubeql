use std::{hint::black_box, sync::Arc, time::Duration};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cubeql::{IndexQueryOptions, Metric, SearchHit, VectorIndex};

const DIMENSION: usize = 128;
const K: usize = 10;

// A direct host scan with partial selection, rather than a full O(N log N) sort.
fn cpu_top_k(values: &[f32], query: &[f32]) -> Vec<SearchHit> {
    let mut hits = values
        .chunks_exact(query.len())
        .enumerate()
        .map(|(row, vector)| SearchHit {
            row,
            score: vector.iter().zip(query).map(|(a, b)| a * b).sum(),
        })
        .collect::<Vec<_>>();
    let compare =
        |a: &SearchHit, b: &SearchHit| b.score.total_cmp(&a.score).then(a.row.cmp(&b.row));
    if hits.len() > K {
        hits.select_nth_unstable_by(K, compare);
        hits.truncate(K);
    }
    hits.sort_unstable_by(compare);
    hits
}

fn metrics_top_k(c: &mut Criterion) {
    let device = cubecl::Device::default();
    eprintln!("CubeCL benchmark backend: {}", device.client().name());
    let query = vec![0.25f32; DIMENSION];
    let mut group = c.benchmark_group(format!(
        "{}_inner_product_top10_d128",
        device.client().name()
    ));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    for rows in [128, 4096, 65_536] {
        let values = (0..rows * DIMENSION)
            .map(|i| ((i % 127) as f32 - 63.0) / 64.0)
            .collect::<Vec<_>>();
        let child: ArrayRef = Arc::new(Float32Array::from(values.clone()));
        let vectors: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIMENSION as i32,
                child,
                None,
            )
            .unwrap(),
        );
        let batch = RecordBatch::try_from_iter(vec![("embedding", vectors)]).unwrap();
        let index = VectorIndex::new(&device, batch, "embedding").unwrap();
        // Validate the benchmark workload before measuring it.
        let expected = cpu_top_k(&values, &query);
        let actual = index
            .query_with_metric(&query, IndexQueryOptions::new(K), Metric::InnerProduct)
            .unwrap();
        assert_eq!(actual, expected);
        group.bench_with_input(
            BenchmarkId::new("cpu_partial_select", rows),
            &rows,
            |b, _| {
                b.iter(|| black_box(cpu_top_k(black_box(&values), black_box(&query))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cubecl_device_top_k", rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    black_box(
                        index
                            .query_with_metric(
                                black_box(&query),
                                IndexQueryOptions::new(K),
                                Metric::InnerProduct,
                            )
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, metrics_top_k);
criterion_main!(benches);
