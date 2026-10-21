use std::{hint::black_box, sync::Arc, time::Duration};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cubeql::{Filter, FilterMode, IndexQueryOptions, Metric, VectorIndex};

fn metadata_filters(c: &mut Criterion) {
    const ROWS: usize = 65_536;
    const DIMENSION: usize = 128;
    let device = cubecl::Device::default();
    let client = device.client();
    let values: ArrayRef = Arc::new(Float32Array::from(
        (0..ROWS * DIMENSION)
            .map(|i| ((i % 127) as f32 - 63.0) / 64.0)
            .collect::<Vec<_>>(),
    ));
    let vectors: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            DIMENSION as i32,
            values,
            None,
        )
        .unwrap(),
    );
    let quality: ArrayRef = Arc::new(Float32Array::from(
        (0..ROWS).map(|row| (row % 100) as f32).collect::<Vec<_>>(),
    ));
    let batch =
        RecordBatch::try_from_iter(vec![("embedding", vectors), ("quality", quality)]).unwrap();
    let index = VectorIndex::new(&device, batch, "embedding").unwrap();
    let query = vec![0.25f32; DIMENSION];
    // Finish initial vector upload before timing filter preparation or warm queries.
    cubecl::future::block_on(client.sync()).unwrap();
    let mut group = c.benchmark_group(format!("{}_metadata_65536x128_top10", client.name()));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    group.bench_function("unfiltered_query", |b| {
        b.iter(|| {
            black_box(
                index
                    .query(black_box(&query), IndexQueryOptions::new(10))
                    .unwrap(),
            )
        })
    });
    for (label, max) in [("100pct", 99.0), ("10pct", 9.0), ("1pct", 0.0)] {
        let filters = [Filter::FloatRange {
            column: "quality",
            min: None,
            max: Some(max),
        }];
        let before = index
            .prepare_filter(&filters, FilterMode::BeforeScoring)
            .unwrap();
        let during = index
            .prepare_filter(&filters, FilterMode::DuringScoring)
            .unwrap();
        assert_eq!(before.len(), during.len());
        assert_eq!(
            before.query(&query, 10, Metric::InnerProduct).unwrap(),
            during.query(&query, 10, Metric::InnerProduct).unwrap()
        );
        for (name, mode, prepared) in [
            ("compact", FilterMode::BeforeScoring, &before),
            ("mask", FilterMode::DuringScoring, &during),
        ] {
            group.bench_function(BenchmarkId::new(format!("prepare_{name}"), label), |b| {
                b.iter(|| {
                    let filter = index.prepare_filter(black_box(&filters), mode).unwrap();
                    // Include completed selection upload, not merely async enqueue time.
                    cubecl::future::block_on(client.sync()).unwrap();
                    black_box(filter)
                })
            });
            group.bench_function(BenchmarkId::new(format!("warm_{name}"), label), |b| {
                b.iter(|| {
                    black_box(
                        prepared
                            .query(black_box(&query), 10, Metric::InnerProduct)
                            .unwrap(),
                    )
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, metadata_filters);
criterion_main!(benches);
