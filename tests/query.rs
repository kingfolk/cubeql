use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arrow_array::{ArrayRef, FixedSizeListArray, Float16Array, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use cubeql::{FloatFilter, IndexQueryOptions, QueryError, QueryOptions, VectorIndex, query};
use half::f16;

fn f32_vectors(rows: &[f32], dimension: i32) -> ArrayRef {
    let values: ArrayRef = Arc::new(Float32Array::from(rows.to_vec()));
    Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dimension,
            values,
            None,
        )
        .unwrap(),
    )
}

fn f16_vectors(rows: &[f32], dimension: i32) -> ArrayRef {
    let values: ArrayRef = Arc::new(Float16Array::from(
        rows.iter().copied().map(f16::from_f32).collect::<Vec<_>>(),
    ));
    Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float16, false)),
            dimension,
            values,
            None,
        )
        .unwrap(),
    )
}

#[test]
fn queries_fp32_vectors_with_a_float_filter() {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "embedding",
            f32_vectors(
                &[
                    1.0, 0.0, 0.0, // filtered out
                    0.5, 0.5, 0.0, // second
                    2.0, 0.0, 0.0, // first
                    -1.0, 0.0, 0.0, // lower score
                ],
                3,
            ),
        ),
        (
            "quality",
            Arc::new(Float32Array::from(vec![0.2, 0.8, 0.9, 0.7])) as ArrayRef,
        ),
    ])
    .unwrap();

    let hits = query(
        &cubecl::Device::default(),
        &batch,
        &[1.0, 0.0, 0.0],
        QueryOptions {
            vector_column: "embedding",
            top_k: 2,
            filter: Some(FloatFilter {
                column: "quality",
                min: 0.5,
            }),
        },
    )
    .unwrap();

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].row, 2);
    assert_abs_diff_eq!(hits[0].score, 2.0, epsilon = 1e-5);
    assert_eq!(hits[1].row, 1);
    assert_abs_diff_eq!(hits[1].score, 0.5, epsilon = 1e-5);
}

#[test]
fn queries_fp16_vectors_without_a_filter() {
    let batch = RecordBatch::try_from_iter(vec![(
        "embedding",
        f16_vectors(
            &[
                0.0, 1.0, // second
                1.0, 0.0, // not returned
                0.5, 2.0, // first
            ],
            2,
        ),
    )])
    .unwrap();

    let hits = query(
        &cubecl::Device::default(),
        &batch,
        &[0.0, 1.0],
        QueryOptions::new("embedding", 2),
    )
    .unwrap();

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].row, 2);
    assert_abs_diff_eq!(hits[0].score, 2.0, epsilon = 2e-3);
    assert_eq!(hits[1].row, 0);
    assert_abs_diff_eq!(hits[1].score, 1.0, epsilon = 2e-3);
}

#[test]
fn resident_index_serves_multiple_queries() {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "embedding",
            f32_vectors(
                &[
                    1.0, 0.0, 0.0, // x
                    0.0, 1.0, 0.0, // y
                    1.0, 1.0, 0.0, // x + y
                    -1.0, 0.0, 0.0,
                ],
                3,
            ),
        ),
        (
            "quality",
            Arc::new(Float32Array::from(vec![0.2, 0.9, 0.8, 1.0])) as ArrayRef,
        ),
    ])
    .unwrap();
    let device = cubecl::Device::default();
    let index = VectorIndex::new(&device, batch.clone(), "embedding").unwrap();

    assert_eq!(index.vector_column(), "embedding");
    assert_eq!(index.dimension(), 3);
    assert_eq!(index.len(), 4);
    assert_eq!(index.batch().num_columns(), 2);
    assert!(matches!(
        index.query(&[1.0, 0.0], IndexQueryOptions::new(2)),
        Err(QueryError::DimensionMismatch {
            query: 2,
            vectors: 3
        })
    ));

    let x_hits = index
        .query(&[1.0, 0.0, 0.0], IndexQueryOptions::new(2))
        .unwrap();
    assert_eq!(x_hits[0].row, 0);
    assert_eq!(x_hits[1].row, 2);

    let y_hits = index
        .query(&[0.0, 1.0, 0.0], IndexQueryOptions::new(2))
        .unwrap();
    assert_eq!(y_hits[0].row, 1);
    assert_eq!(y_hits[1].row, 2);

    let filtered_hits = index
        .query(
            &[1.0, 0.0, 0.0],
            IndexQueryOptions {
                top_k: 2,
                filter: Some(FloatFilter {
                    column: "quality",
                    min: 0.8,
                }),
            },
        )
        .unwrap();
    assert_eq!(filtered_hits[0].row, 2);
    assert_eq!(filtered_hits[1].row, 1);

    let one_shot = query(
        &device,
        &batch,
        &[1.0, 0.0, 0.0],
        QueryOptions::new("embedding", 2),
    )
    .unwrap();
    assert_eq!(x_hits, one_shot);
}

#[test]
fn resident_index_handles_sliced_arrow_arrays() {
    let batch = RecordBatch::try_from_iter(vec![(
        "embedding",
        f32_vectors(
            &[
                100.0, 0.0, // excluded by the slice
                1.0, 0.0, // row 0 in the sliced batch
                0.0, 2.0, // row 1 in the sliced batch
                200.0, 0.0, // excluded by the slice
            ],
            2,
        ),
    )])
    .unwrap()
    .slice(1, 2);
    let index = VectorIndex::new(&cubecl::Device::default(), batch, "embedding").unwrap();

    let hits = index.query(&[1.0, 0.0], IndexQueryOptions::new(1)).unwrap();

    assert_eq!(hits[0].row, 0);
    assert_abs_diff_eq!(hits[0].score, 1.0, epsilon = 1e-5);
}

#[test]
fn resident_index_accepts_an_empty_batch() {
    let batch =
        RecordBatch::try_from_iter(vec![("embedding", f32_vectors(&[1.0, 0.0, 0.0, 1.0], 2))])
            .unwrap()
            .slice(0, 0);
    let index = VectorIndex::new(&cubecl::Device::default(), batch, "embedding").unwrap();

    assert!(index.is_empty());
    assert_eq!(index.dimension(), 2);
    assert!(
        index
            .query(&[1.0, 0.0], IndexQueryOptions::new(10))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn resident_index_validates_the_vector_schema_when_created() {
    let batch = RecordBatch::try_from_iter(vec![(
        "embedding",
        Arc::new(Float32Array::from(vec![1.0, 2.0])) as ArrayRef,
    )])
    .unwrap();

    assert!(matches!(
        VectorIndex::new(&cubecl::Device::default(), batch, "embedding"),
        Err(QueryError::UnsupportedVectorType(DataType::Float32))
    ));
}
