use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arrow_array::{ArrayRef, FixedSizeListArray, Float16Array, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use cubeql::{
    FloatFilter, IndexOptions, IndexQueryOptions, Metric, QueryError, SearchHit, VectorIndex,
};
use half::f16;

fn batch(values: &[f32], dimension: usize, fp16: bool) -> RecordBatch {
    let (values, ty): (ArrayRef, _) = if fp16 {
        (
            Arc::new(Float16Array::from(
                values
                    .iter()
                    .copied()
                    .map(f16::from_f32)
                    .collect::<Vec<_>>(),
            )),
            DataType::Float16,
        )
    } else {
        (
            Arc::new(Float32Array::from(values.to_vec())),
            DataType::Float32,
        )
    };
    let vectors: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", ty, false)),
            dimension as i32,
            values,
            None,
        )
        .unwrap(),
    );
    let quality: ArrayRef = Arc::new(Float32Array::from(
        (0..vectors.len())
            .map(|row| (row % 7) as f32)
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_from_iter(vec![("embedding", vectors), ("quality", quality)]).unwrap()
}

fn prepared(values: &[f32], dimension: usize, fp16: bool, normalize: bool) -> Vec<f32> {
    let round = |v| if fp16 { f16::from_f32(v).to_f32() } else { v };
    let mut values = values.iter().copied().map(round).collect::<Vec<_>>();
    if normalize {
        for vector in values.chunks_exact_mut(dimension) {
            let norm = vector
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            for value in vector {
                *value = if norm == 0.0 {
                    0.0
                } else {
                    round((f64::from(*value) / norm) as f32)
                };
            }
        }
    }
    values
}

fn reference(values: &[f32], query: &[f32], metric: Metric, min: f32, k: usize) -> Vec<SearchHit> {
    let mut hits = values
        .chunks_exact(query.len())
        .enumerate()
        .filter(|(row, _)| (row % 7) as f32 >= min)
        .map(|(row, vector)| {
            let dot: f64 = vector
                .iter()
                .zip(query)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let score = match metric {
                Metric::InnerProduct => dot,
                Metric::SquaredL2 => vector
                    .iter()
                    .zip(query)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum(),
                Metric::Cosine => {
                    let a = vector
                        .iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let b = query
                        .iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if a == 0.0 || b == 0.0 {
                        0.0
                    } else {
                        dot / (a * b)
                    }
                }
            } as f32;
            SearchHit { row, score }
        })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        let order = if metric == Metric::SquaredL2 {
            a.score.total_cmp(&b.score)
        } else {
            b.score.total_cmp(&a.score)
        };
        order.then(a.row.cmp(&b.row))
    });
    hits.truncate(k);
    hits
}

#[test]
fn metrics_and_multistage_top_k_match_cpu_reference() {
    let dimension = 7;
    let mut seed = 42u32;
    let values = (0..777 * dimension)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 12) % 2001) as f32 / 128.0 - 8.0
        })
        .collect::<Vec<_>>();
    let query = [0.5, -0.25, 0.125, 1.0, -0.75, 0.375, -0.5];
    for fp16 in [false, true] {
        for normalize in [false, true] {
            let index = VectorIndex::new_with_options(
                &cubecl::Device::default(),
                batch(&values, dimension, fp16),
                "embedding",
                IndexOptions { normalize },
            )
            .unwrap();
            assert_eq!(index.options().normalize, normalize);
            let stored = prepared(&values, dimension, fp16, normalize);
            for metric in [Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2] {
                for k in [0, 1, 10, 300, 1000] {
                    let hits = index
                        .query_with_metric(
                            &query,
                            IndexQueryOptions {
                                top_k: k,
                                filter: Some(FloatFilter {
                                    column: "quality",
                                    min: 2.0,
                                }),
                            },
                            metric,
                        )
                        .unwrap();
                    let expected = reference(&stored, &query, metric, 2.0, k);
                    assert_eq!(hits.len(), expected.len());
                    for (actual, expected) in hits.iter().zip(&expected) {
                        assert_eq!(
                            actual.row, expected.row,
                            "{metric:?}, fp16={fp16}, normalized={normalize}, k={k}"
                        );
                        assert_abs_diff_eq!(actual.score, expected.score, epsilon = 1e-4);
                    }
                }
            }
        }
    }
}

#[test]
fn cosine_zero_and_extreme_finite_vectors_are_defined() {
    let values = [0.0, 0.0, 3.0, 4.0, -3.0, -4.0, 3e30, 4e30, 3e-30, 4e-30];
    let index = VectorIndex::new(
        &cubecl::Device::default(),
        batch(&values, 2, false),
        "embedding",
    )
    .unwrap();
    for query in [[0.0, 0.0], [3e30, 4e30], [3e-30, 4e-30]] {
        let hits = index
            .query_with_metric(&query, IndexQueryOptions::new(10), Metric::Cosine)
            .unwrap();
        assert_eq!(hits.len(), 5);
        for hit in hits {
            let expected = if query[0] == 0.0 || hit.row == 0 {
                0.0
            } else if hit.row == 2 {
                -1.0
            } else {
                1.0
            };
            assert_abs_diff_eq!(hit.score, expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn ties_filters_and_exhausted_tiles_never_duplicate_rows() {
    for metric in [Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2] {
        let index = VectorIndex::new(
            &cubecl::Device::default(),
            batch(&vec![-1.0; 1030], 2, false),
            "embedding",
        )
        .unwrap();
        for min in [0.0, 5.0, 10.0] {
            let actual = index
                .query_with_metric(
                    &[1.0, 1.0],
                    IndexQueryOptions {
                        top_k: 600,
                        filter: Some(FloatFilter {
                            column: "quality",
                            min,
                        }),
                    },
                    metric,
                )
                .unwrap();
            let expected = reference(&vec![-1.0; 1030], &[1.0, 1.0], metric, min, 600);
            assert_eq!(
                actual.iter().map(|h| h.row).collect::<Vec<_>>(),
                expected.iter().map(|h| h.row).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn three_selection_stages_keep_global_row_ids_and_boundary_ties() {
    let mut values = vec![-1.0; 70_001];
    values[256] = 9.0;
    values[69_999] = 9.0;
    let index = VectorIndex::new(
        &cubecl::Device::default(),
        batch(&values, 1, false),
        "embedding",
    )
    .unwrap();
    assert_eq!(
        index.query(&[1.0], IndexQueryOptions::new(1)).unwrap(),
        vec![SearchHit {
            row: 256,
            score: 9.0
        }]
    );
}

#[test]
fn normalization_preserves_arrow_metadata_and_handles_zero_vectors() {
    for fp16 in [false, true] {
        let batch = batch(&[3.0, 4.0, 0.0, 0.0], 2, fp16);
        let original = batch.clone();
        let index = VectorIndex::new_with_options(
            &cubecl::Device::default(),
            batch,
            "embedding",
            IndexOptions { normalize: true },
        )
        .unwrap();
        assert_eq!(index.batch(), &original);
        let hits = index.query(&[1.0, 0.0], IndexQueryOptions::new(2)).unwrap();
        assert_abs_diff_eq!(hits[0].score, 0.6, epsilon = 3e-4);
        assert_eq!(hits[1].score, 0.0);
    }
}

#[test]
fn nonfinite_inputs_are_rejected_but_infinite_scores_are_ranked() {
    let device = cubecl::Device::default();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            VectorIndex::new(&device, batch(&[value], 1, false), "embedding"),
            Err(QueryError::NonFiniteVector)
        ));
    }
    let index = VectorIndex::new(
        &device,
        batch(&[f32::MAX, -f32::MAX], 1, false),
        "embedding",
    )
    .unwrap();
    assert!(matches!(
        index.query(&[f32::NAN], IndexQueryOptions::new(1)),
        Err(QueryError::NonFiniteVector)
    ));
    let hits = index.query(&[2.0], IndexQueryOptions::new(2)).unwrap();
    assert_eq!(
        hits,
        vec![
            SearchHit {
                row: 0,
                score: f32::INFINITY
            },
            SearchHit {
                row: 1,
                score: f32::NEG_INFINITY
            }
        ]
    );
}
