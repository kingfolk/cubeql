use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arrow_array::{ArrayRef, FixedSizeListArray, Float16Array, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use cubeql::{FloatFilter, QueryOptions, query};
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
