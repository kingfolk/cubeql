use std::{hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use criterion::{Criterion, criterion_group, criterion_main};
use cubeql::{IndexQueryOptions, QueryOptions, VectorIndex, query};

const ROWS: usize = 4_096;
const DIMENSION: usize = 128;

fn vector_batch() -> RecordBatch {
    let values = (0..ROWS * DIMENSION)
        .map(|index| ((index % 31) as f32 - 15.0) / 31.0)
        .collect::<Vec<_>>();
    let values: ArrayRef = Arc::new(Float32Array::from(values));
    let vectors: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            DIMENSION as i32,
            values,
            None,
        )
        .unwrap(),
    );
    RecordBatch::try_from_iter(vec![("embedding", vectors)]).unwrap()
}

fn resident_index(c: &mut Criterion) {
    let device = cubecl::Device::default();
    let batch = vector_batch();
    let query_vector = vec![1.0f32; DIMENSION];

    c.bench_function("index_load_4096x128_f32", |bencher| {
        bencher.iter(|| {
            black_box(VectorIndex::new(&device, black_box(batch.clone()), "embedding").unwrap())
        });
    });

    let index = VectorIndex::new(&device, batch.clone(), "embedding").unwrap();
    c.bench_function("warm_query_4096x128_f32", |bencher| {
        bencher.iter(|| {
            black_box(
                index
                    .query(black_box(&query_vector), IndexQueryOptions::new(10))
                    .unwrap(),
            )
        });
    });

    c.bench_function("one_shot_query_4096x128_f32", |bencher| {
        bencher.iter(|| {
            black_box(
                query(
                    &device,
                    black_box(&batch),
                    black_box(&query_vector),
                    QueryOptions::new("embedding", 10),
                )
                .unwrap(),
            )
        });
    });
}

criterion_group!(benches, resident_index);
criterion_main!(benches);
