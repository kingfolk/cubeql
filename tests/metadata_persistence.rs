use std::{collections::HashMap, io::Cursor, sync::Arc};

use approx::assert_abs_diff_eq;
use arrow_array::{
    ArrayRef, DictionaryArray, FixedSizeListArray, Float16Array, Float32Array, Float64Array,
    Int8Array, LargeStringArray, RecordBatch, StringArray, StringViewArray, types::Int8Type,
};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use cubeql::{
    Filter, FilterMode, IndexOptions, IndexQueryOptions, Metric, PersistenceError, QueryError,
    SearchHit, VectorIndex,
};
use half::f16;

fn fixture(fp16: bool) -> RecordBatch {
    let count = 777;
    let values = (0..count)
        .flat_map(|row| [(row % 11) as f32 - 5.0, (row % 13) as f32 - 6.0, 1.0])
        .collect::<Vec<_>>();
    let (values, ty): (ArrayRef, _) = if fp16 {
        (
            Arc::new(Float16Array::from(
                values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
            )),
            DataType::Float16,
        )
    } else {
        (Arc::new(Float32Array::from(values)), DataType::Float32)
    };
    let vectors: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(Arc::new(Field::new("item", ty, false)), 3, values, None)
            .unwrap(),
    );
    let keys = Int8Array::from(
        (0..count)
            .map(|row| {
                if row % 37 == 0 {
                    None
                } else {
                    Some((row % 3) as i8)
                }
            })
            .collect::<Vec<_>>(),
    );
    let dictionary: ArrayRef = Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            keys,
            Arc::new(StringArray::from(vec![Some("rust"), None, Some("python")])),
        )
        .unwrap(),
    );
    let quality = (0..count)
        .map(|row| {
            if row % 23 == 0 {
                None
            } else if row % 29 == 0 {
                Some(f32::NAN)
            } else {
                Some((row % 10) as f32 / 10.0)
            }
        })
        .collect::<Vec<_>>();
    let mut batch = RecordBatch::try_from_iter(vec![
        ("embedding", vectors),
        ("quality", Arc::new(Float32Array::from(quality)) as ArrayRef),
        (
            "half",
            Arc::new(Float16Array::from(
                (0..count)
                    .map(|row| Some(f16::from_f32((row % 10) as f32)))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "double",
            Arc::new(Float64Array::from(
                (0..count).map(|row| row as f64 / 10.0).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        ("language", dictionary),
        (
            "repo",
            Arc::new(StringArray::from(
                (0..count)
                    .map(|row| {
                        if row % 2 == 0 {
                            Some("cubecl")
                        } else {
                            Some("burn")
                        }
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "path",
            Arc::new(LargeStringArray::from(
                (0..count)
                    .map(|row| {
                        if row % 4 == 0 {
                            "crates/opt/pass.rs"
                        } else {
                            "tests/pass.rs"
                        }
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "view",
            Arc::new(StringViewArray::from(vec![Some("rust"); count])) as ArrayRef,
        ),
    ])
    .unwrap();
    batch
        .schema_metadata_mut()
        .insert("embedding_model", "fixture-v1");
    let fields = batch
        .schema()
        .fields()
        .iter()
        .map(|field| {
            field.as_ref().clone().with_metadata(HashMap::from([(
                "description".to_string(),
                field.name().clone(),
            )]))
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        batch.schema().metadata().clone(),
    ));
    batch = RecordBatch::try_new(schema, batch.columns().to_vec()).unwrap();
    batch
}

fn reference(
    rows: impl Iterator<Item = usize>,
    query: &[f32],
    metric: Metric,
    k: usize,
) -> Vec<SearchHit> {
    let mut hits = rows
        .map(|row| {
            let vector = [(row % 11) as f64 - 5.0, (row % 13) as f64 - 6.0, 1.0];
            let dot: f64 = vector
                .iter()
                .zip(query)
                .map(|(a, b)| a * f64::from(*b))
                .sum();
            let score = match metric {
                Metric::InnerProduct => dot,
                Metric::SquaredL2 => vector
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a - f64::from(*b)).powi(2))
                    .sum(),
                Metric::Cosine => {
                    dot / (vector.iter().map(|v| v * v).sum::<f64>().sqrt()
                        * query
                            .iter()
                            .map(|v| f64::from(*v).powi(2))
                            .sum::<f64>()
                            .sqrt())
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
fn combined_filters_and_both_execution_modes_match_cpu_reference() {
    let device = cubecl::Device::default();
    for fp16 in [false, true] {
        let index = VectorIndex::new(&device, fixture(fp16), "embedding").unwrap();
        let filters = [
            Filter::FloatRange {
                column: "quality",
                min: Some(0.5),
                max: Some(0.9),
            },
            Filter::StringEq {
                column: "language",
                value: "rust",
            },
            Filter::StringIn {
                column: "repo",
                values: &["cubecl", "cubecl"],
            },
            Filter::PathPrefix {
                column: "path",
                prefix: "crates/opt/",
            },
        ];
        let expected_rows = (0..777)
            .filter(|row| {
                row % 23 != 0
                    && row % 29 != 0
                    && row % 10 >= 5
                    && row % 37 != 0
                    && row % 3 == 0
                    && row % 4 == 0
            })
            .collect::<Vec<_>>();
        for mode in [FilterMode::BeforeScoring, FilterMode::DuringScoring] {
            let prepared = index.prepare_filter(&filters, mode).unwrap();
            assert_eq!(prepared.len(), expected_rows.len());
            assert_eq!(prepared.mode(), mode);
            for query in [[1.0, 0.25, -0.5], [-0.5, 1.0, 0.25]] {
                for metric in [Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2] {
                    for k in [0, 1, 10, 1000] {
                        let hits = prepared.query(&query, k, metric).unwrap();
                        let expected = reference(expected_rows.iter().copied(), &query, metric, k);
                        assert_eq!(
                            hits.iter().map(|h| h.row).collect::<Vec<_>>(),
                            expected.iter().map(|h| h.row).collect::<Vec<_>>()
                        );
                        for (actual, expected) in hits.iter().zip(&expected) {
                            assert_abs_diff_eq!(actual.score, expected.score, epsilon = 1e-5);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn filter_types_bounds_nulls_and_slices_have_defined_semantics() {
    let batch = fixture(false).slice(10, 100);
    let index = VectorIndex::new(&cubecl::Device::default(), batch, "embedding").unwrap();
    let cases: Vec<(Vec<Filter<'_>>, Vec<usize>)> = vec![
        (vec![], (10..110).collect()),
        (
            vec![Filter::FloatEq {
                column: "quality",
                value: 0.5,
            }],
            (10..110)
                .filter(|r| r % 10 == 5 && r % 23 != 0 && r % 29 != 0)
                .collect(),
        ),
        (
            vec![Filter::FloatRange {
                column: "half",
                min: Some(2.0),
                max: Some(2.0),
            }],
            (10..110).filter(|r| r % 10 == 2).collect(),
        ),
        (
            vec![Filter::FloatRange {
                column: "double",
                min: None,
                max: Some(2.0),
            }],
            (10..=20).collect(),
        ),
        (
            vec![Filter::FloatRange {
                column: "quality",
                min: None,
                max: None,
            }],
            (10..110).filter(|r| r % 23 != 0 && r % 29 != 0).collect(),
        ),
        (
            vec![Filter::StringIn {
                column: "language",
                values: &["rust", "python"],
            }],
            (10..110).filter(|r| r % 37 != 0 && r % 3 != 1).collect(),
        ),
        (
            vec![Filter::StringEq {
                column: "view",
                value: "rust",
            }],
            (10..110).collect(),
        ),
        (
            vec![Filter::StringIn {
                column: "repo",
                values: &[],
            }],
            vec![],
        ),
        (
            vec![Filter::PathPrefix {
                column: "path",
                prefix: "missing/",
            }],
            vec![],
        ),
    ];
    for (filters, expected) in cases {
        for mode in [FilterMode::BeforeScoring, FilterMode::DuringScoring] {
            let prepared = index.prepare_filter(&filters, mode).unwrap();
            let mut rows = prepared
                .query(&[1.0, 0.0, 0.0], 1000, Metric::InnerProduct)
                .unwrap()
                .iter()
                .map(|h| h.row + 10)
                .collect::<Vec<_>>();
            rows.sort_unstable();
            assert_eq!(rows, expected);
        }
    }
    let empty = VectorIndex::new(
        &cubecl::Device::default(),
        fixture(true).slice(0, 0),
        "embedding",
    )
    .unwrap();
    let prepared = empty
        .prepare_filter(&[], FilterMode::BeforeScoring)
        .unwrap();
    assert!(prepared.is_empty());
    assert!(
        prepared
            .query(&[1.0, 0.0, 0.0], 10, Metric::Cosine)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn invalid_predicates_are_reported_even_after_an_empty_match() {
    let index = VectorIndex::new(&cubecl::Device::default(), fixture(false), "embedding").unwrap();
    for filter in [
        Filter::FloatEq {
            column: "quality",
            value: f64::NAN,
        },
        Filter::FloatRange {
            column: "quality",
            min: Some(2.0),
            max: Some(1.0),
        },
    ] {
        assert!(matches!(
            index.prepare_filter(&[filter], FilterMode::BeforeScoring),
            Err(QueryError::InvalidFilter(_))
        ));
    }
    assert!(matches!(
        index.prepare_filter(
            &[
                Filter::StringIn {
                    column: "repo",
                    values: &[]
                },
                Filter::StringEq {
                    column: "absent",
                    value: "x"
                }
            ],
            FilterMode::BeforeScoring
        ),
        Err(QueryError::MissingColumn(_))
    ));
    assert!(matches!(
        index.prepare_filter(
            &[Filter::FloatEq {
                column: "repo",
                value: 1.0
            }],
            FilterMode::BeforeScoring
        ),
        Err(QueryError::UnsupportedMetadataType(_))
    ));
}

#[test]
fn ipc_roundtrip_preserves_metadata_options_and_filtered_results() {
    let device = cubecl::Device::default();
    for fp16 in [false, true] {
        for normalize in [false, true] {
            let index = VectorIndex::new_with_options(
                &device,
                fixture(fp16).slice(10, 300),
                "embedding",
                IndexOptions { normalize },
            )
            .unwrap();
            let mut bytes = Vec::new();
            index.write_ipc(&mut bytes).unwrap();
            let loaded = VectorIndex::read_ipc(&device, Cursor::new(bytes)).unwrap();
            assert_eq!(loaded.batch(), index.batch());
            assert_eq!(loaded.options(), index.options());
            assert_eq!(loaded.vector_column(), index.vector_column());
            let filters = [
                Filter::StringEq {
                    column: "language",
                    value: "rust",
                },
                Filter::FloatRange {
                    column: "quality",
                    min: Some(0.5),
                    max: None,
                },
            ];
            for metric in [Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2] {
                assert_eq!(
                    loaded
                        .query_filtered(&[1.0, 0.25, 0.5], 10, metric, &filters)
                        .unwrap(),
                    index
                        .query_filtered(&[1.0, 0.25, 0.5], 10, metric, &filters)
                        .unwrap()
                );
            }
        }
    }
}

fn raw_ipc(batch: &RecordBatch, batches: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &batch.schema()).unwrap();
    for _ in 0..batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
    drop(writer);
    bytes
}

#[test]
fn ipc_rejects_bad_versions_options_batch_counts_and_corruption() {
    let device = cubecl::Device::default();
    let original = fixture(false);
    assert!(matches!(
        VectorIndex::read_ipc(&device, Cursor::new(raw_ipc(&original, 1))),
        Err(PersistenceError::Format(_))
    ));
    for (key, value) in [
        ("cubeql.format_version", "99"),
        ("cubeql.normalize", "invalid"),
    ] {
        let mut batch = original.clone();
        batch
            .schema_metadata_mut()
            .insert("cubeql.format_version", "1");
        batch
            .schema_metadata_mut()
            .insert("cubeql.vector_column", "embedding");
        batch
            .schema_metadata_mut()
            .insert("cubeql.normalize", "false");
        batch.schema_metadata_mut().insert(key, value);
        assert!(matches!(
            VectorIndex::read_ipc(&device, Cursor::new(raw_ipc(&batch, 1))),
            Err(PersistenceError::Format(_))
        ));
    }
    let mut batch = original;
    batch
        .schema_metadata_mut()
        .insert("cubeql.format_version", "1");
    batch
        .schema_metadata_mut()
        .insert("cubeql.vector_column", "embedding");
    batch
        .schema_metadata_mut()
        .insert("cubeql.normalize", "false");
    for count in [0, 2] {
        assert!(matches!(
            VectorIndex::read_ipc(&device, Cursor::new(raw_ipc(&batch, count))),
            Err(PersistenceError::Format(_))
        ));
    }
    assert!(VectorIndex::read_ipc(&device, Cursor::new(vec![0u8; 64])).is_err());
    let reserved = VectorIndex::new(&device, batch, "embedding").unwrap();
    assert!(matches!(
        reserved.write_ipc(Vec::new()),
        Err(PersistenceError::Format(_))
    ));
}

#[test]
fn ipc_empty_batch_and_file_helpers_work_without_overwriting() {
    let device = cubecl::Device::default();
    let index = VectorIndex::new(&device, fixture(false).slice(0, 0), "embedding").unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.arrow");
    index.save_ipc(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(
        matches!(index.save_ipc(&path), Err(PersistenceError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists)
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let loaded = VectorIndex::load_ipc(&device, path).unwrap();
    assert!(loaded.is_empty());
    assert_eq!(loaded.dimension(), 3);
    assert!(
        loaded
            .query(&[1.0, 0.0, 0.0], IndexQueryOptions::new(10))
            .unwrap()
            .is_empty()
    );
}

#[cfg(all(feature = "cpu", feature = "metal", target_vendor = "apple"))]
#[test]
fn ipc_created_on_cpu_can_be_loaded_on_metal() {
    let cpu = cubecl::Device::cpu().unwrap();
    let metal = cubecl::Device::metal(Default::default()).unwrap();
    let index = VectorIndex::new_with_options(
        &cpu,
        fixture(true),
        "embedding",
        IndexOptions { normalize: true },
    )
    .unwrap();
    let mut bytes = Vec::new();
    index.write_ipc(&mut bytes).unwrap();
    let loaded = VectorIndex::read_ipc(&metal, Cursor::new(bytes)).unwrap();
    let filters = [Filter::StringEq {
        column: "language",
        value: "rust",
    }];
    let cpu_hits = index
        .query_filtered(&[1.0, 0.25, 0.5], 10, Metric::InnerProduct, &filters)
        .unwrap();
    let metal_hits = loaded
        .query_filtered(&[1.0, 0.25, 0.5], 10, Metric::InnerProduct, &filters)
        .unwrap();
    assert_eq!(
        cpu_hits.iter().map(|h| h.row).collect::<Vec<_>>(),
        metal_hits.iter().map(|h| h.row).collect::<Vec<_>>()
    );
    for (cpu, metal) in cpu_hits.iter().zip(&metal_hits) {
        assert_abs_diff_eq!(cpu.score, metal.score, epsilon = 1e-5);
    }
}
