//! A deliberately small Arrow-native retrieval engine.
//!
//! The first implementation performs an exact inner-product scan with CubeCL. It accepts a
//! `FixedSizeList<Float32>` or `FixedSizeList<Float16>` Arrow column, optionally filters rows with
//! a floating-point column, and performs the final top-k selection on the CPU.

use arrow_array::{
    Array, FixedSizeListArray, Float16Array, Float32Array, RecordBatch,
    cast::AsArray,
    types::{Float16Type, Float32Type},
};
use arrow_schema::DataType;
use cubecl::{Device, calculate_cube_count_elemwise, prelude::*};
use half::f16;
use thiserror::Error;

const UNITS_PER_CUBE: u32 = 64;

/// A lower bound applied to one floating-point Arrow column before scoring a row.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatFilter<'a> {
    /// Name of a `Float32` or `Float16` column in the input batch.
    pub column: &'a str,
    /// Rows whose value is lower than this value are excluded.
    pub min: f32,
}

/// Options for an exact inner-product query.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryOptions<'a> {
    /// Name of the Arrow `FixedSizeList<Float32|Float16>` embedding column.
    pub vector_column: &'a str,
    /// Maximum number of hits returned.
    pub top_k: usize,
    /// Optional floating-point predicate.
    pub filter: Option<FloatFilter<'a>>,
}

impl<'a> QueryOptions<'a> {
    /// Create options without a filter.
    pub fn new(vector_column: &'a str, top_k: usize) -> Self {
        Self {
            vector_column,
            top_k,
            filter: None,
        }
    }
}

/// One matching row from the input [`RecordBatch`].
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    /// Zero-based row index in the input batch.
    pub row: usize,
    /// Inner product between the row vector and query vector.
    pub score: f32,
}

/// Errors returned before or while launching the retrieval kernel.
#[derive(Debug, Error)]
pub enum QueryError {
    #[error("column '{0}' does not exist")]
    MissingColumn(String),
    #[error("vector column must be FixedSizeList<Float32> or FixedSizeList<Float16>, got {0}")]
    UnsupportedVectorType(DataType),
    #[error("filter column must be Float32 or Float16, got {0}")]
    UnsupportedFilterType(DataType),
    #[error("vector dimension must be greater than zero")]
    EmptyVector,
    #[error("query dimension {query} does not match vector dimension {vectors}")]
    DimensionMismatch { query: usize, vectors: usize },
    #[error("nulls are not supported in the vector column yet")]
    NullVector,
    #[error("nulls are not supported in the filter column yet")]
    NullFilter,
    #[error("failed to read scores from the compute device: {0}")]
    ReadScores(String),
}

/// Execute an exact inner-product scan over an Arrow record batch.
///
/// The vector column must be a non-null `FixedSizeList<Float32>` or `FixedSizeList<Float16>`.
/// `query_vector` is always supplied as FP32. When a filter is present, only rows whose filter
/// value is greater than or equal to its minimum are scored.
pub fn query(
    device: &Device,
    batch: &RecordBatch,
    query_vector: &[f32],
    options: QueryOptions<'_>,
) -> Result<Vec<SearchHit>, QueryError> {
    if options.top_k == 0 || batch.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let vectors = batch
        .column_by_name(options.vector_column)
        .ok_or_else(|| QueryError::MissingColumn(options.vector_column.to_string()))?;
    let vectors = vectors
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| QueryError::UnsupportedVectorType(vectors.data_type().clone()))?;

    if vectors.null_count() != 0 || vectors.values().null_count() != 0 {
        return Err(QueryError::NullVector);
    }

    let dimension = usize::try_from(vectors.value_length()).expect("Arrow dimensions are positive");
    if dimension == 0 {
        return Err(QueryError::EmptyVector);
    }
    if query_vector.len() != dimension {
        return Err(QueryError::DimensionMismatch {
            query: query_vector.len(),
            vectors: dimension,
        });
    }

    let filter_values = read_filter(batch, options.filter.as_ref())?;
    let client = device.client();
    let query_handle = client.create_from_slice(f32::as_bytes(query_vector));
    let filter_handle = client.create_from_slice(f32::as_bytes(&filter_values));
    let initial_scores = vec![f32::NEG_INFINITY; batch.num_rows()];
    let scores_handle = client.create_from_slice(f32::as_bytes(&initial_scores));
    let cube_dim = CubeDim::new_1d(UNITS_PER_CUBE);
    let cube_count = calculate_cube_count_elemwise(&client, batch.num_rows(), cube_dim);
    let use_filter = u32::from(options.filter.is_some());
    let filter_min = options.filter.as_ref().map_or(0.0, |filter| filter.min);

    match vectors.value_type() {
        DataType::Float32 => {
            let values = vectors.values().as_primitive::<Float32Type>();
            let vectors_handle = client.create_from_slice(f32::as_bytes(values.values()));
            unsafe {
                exact_inner_product::launch_unchecked::<f32>(
                    &client,
                    cube_count,
                    cube_dim,
                    BufferArg::from_raw_parts(vectors_handle, values.len()),
                    BufferArg::from_raw_parts(query_handle, query_vector.len()),
                    BufferArg::from_raw_parts(filter_handle, filter_values.len()),
                    BufferArg::from_raw_parts(scores_handle.clone(), batch.num_rows()),
                    batch.num_rows(),
                    dimension,
                    use_filter,
                    filter_min,
                );
            }
        }
        DataType::Float16 => {
            let values = vectors.values().as_primitive::<Float16Type>();
            let vectors_handle = client.create_from_slice(f16::as_bytes(values.values()));
            unsafe {
                exact_inner_product::launch_unchecked::<f16>(
                    &client,
                    cube_count,
                    cube_dim,
                    BufferArg::from_raw_parts(vectors_handle, values.len()),
                    BufferArg::from_raw_parts(query_handle, query_vector.len()),
                    BufferArg::from_raw_parts(filter_handle, filter_values.len()),
                    BufferArg::from_raw_parts(scores_handle.clone(), batch.num_rows()),
                    batch.num_rows(),
                    dimension,
                    use_filter,
                    filter_min,
                );
            }
        }
        other => return Err(QueryError::UnsupportedVectorType(other)),
    }

    let score_bytes = client
        .read_one(scores_handle)
        .map_err(|error| QueryError::ReadScores(error.to_string()))?;
    let scores = f32::from_bytes(&score_bytes);
    let mut hits = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| *score != f32::NEG_INFINITY)
        .map(|(row, score)| SearchHit { row, score })
        .collect::<Vec<_>>();

    hits.sort_unstable_by(compare_hits);
    hits.truncate(options.top_k.min(hits.len()));
    Ok(hits)
}

fn read_filter(
    batch: &RecordBatch,
    filter: Option<&FloatFilter<'_>>,
) -> Result<Vec<f32>, QueryError> {
    let Some(filter) = filter else {
        return Ok(vec![0.0; batch.num_rows()]);
    };
    let values = batch
        .column_by_name(filter.column)
        .ok_or_else(|| QueryError::MissingColumn(filter.column.to_string()))?;
    if values.null_count() != 0 {
        return Err(QueryError::NullFilter);
    }

    match values.data_type() {
        DataType::Float32 => Ok(values
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("data type checked")
            .values()
            .to_vec()),
        DataType::Float16 => Ok(values
            .as_any()
            .downcast_ref::<Float16Array>()
            .expect("data type checked")
            .values()
            .iter()
            .map(|value| value.to_f32())
            .collect()),
        other => Err(QueryError::UnsupportedFilterType(other.clone())),
    }
}

#[cube(launch_unchecked)]
fn exact_inner_product<F: Float>(
    vectors: &[F],
    query: &[f32],
    filter_values: &[f32],
    scores: &mut [f32],
    row_count: usize,
    dimension: usize,
    use_filter: u32,
    filter_min: f32,
) {
    let row = ABSOLUTE_POS;
    if row < row_count {
        let passes_filter = use_filter == 0 || filter_values[row] >= filter_min;
        if passes_filter {
            let mut score = 0.0f32;
            for column in 0..dimension {
                let value = f32::cast_from(vectors[row * dimension + column]);
                score += value * query[column];
            }
            scores[row] = score;
        }
    }
}

// Keep deterministic ordering logic explicit and close to the public result type.
fn compare_hits(lhs: &SearchHit, rhs: &SearchHit) -> std::cmp::Ordering {
    rhs.score
        .total_cmp(&lhs.score)
        .then_with(|| lhs.row.cmp(&rhs.row))
}
