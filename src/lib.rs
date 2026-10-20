//! A deliberately small Arrow-native retrieval engine.
//!
//! The engine performs exact vector scans and top-k selection with CubeCL. It accepts a
//! `FixedSizeList<Float32>` or `FixedSizeList<Float16>` Arrow column, optionally filters rows with
//! a floating-point column. Only the selected row identifiers and scores return to the host.

use arrow_array::{
    Array, FixedSizeListArray, Float16Array, Float32Array, RecordBatch,
    cast::AsArray,
    types::{Float16Type, Float32Type},
};
use arrow_schema::DataType;
use cubecl::{Device, calculate_cube_count_elemwise, client::Client, prelude::*, server::Handle};
use half::f16;
use thiserror::Error;

const UNITS_PER_CUBE: u32 = 64;
const INVALID_ROW: u32 = u32::MAX;

/// Metric evaluated over the stored vectors (FP16 inputs accumulate in FP32).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Metric {
    /// Larger dot products rank first.
    #[default]
    InnerProduct,
    /// Larger cosine similarities rank first. A zero vector has similarity zero.
    Cosine,
    /// Smaller squared Euclidean distances rank first; returned scores are nonnegative distances.
    SquaredL2,
}

impl Metric {
    fn code(self) -> u32 {
        match self {
            Self::InnerProduct => 0,
            Self::Cosine => 1,
            Self::SquaredL2 => 2,
        }
    }
}

/// Options applied once when constructing an index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexOptions {
    /// L2-normalize each vector before upload, preserving its original scalar type.
    /// Zero vectors remain zero. The original Arrow batch is left unchanged.
    pub normalize: bool,
}

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

/// Options for querying an existing [`VectorIndex`].
#[derive(Clone, Debug, PartialEq)]
pub struct IndexQueryOptions<'a> {
    /// Maximum number of hits returned.
    pub top_k: usize,
    /// Optional floating-point predicate over the index's Arrow metadata.
    pub filter: Option<FloatFilter<'a>>,
}

impl IndexQueryOptions<'_> {
    /// Create options without a filter.
    pub fn new(top_k: usize) -> Self {
        Self {
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
    /// Metric value: descending inner product/cosine, or ascending squared L2 distance.
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
    #[error("vector values and query values must be finite")]
    NonFiniteVector,
    #[error("the index must contain fewer than u32::MAX rows")]
    TooManyRows,
    #[error("failed to read query results from the compute device: {0}")]
    ReadScores(String),
}

enum VectorBuffer {
    Float32 { handle: Option<Handle>, len: usize },
    Float16 { handle: Option<Handle>, len: usize },
}

/// An Arrow vector column uploaded once and retained on a CubeCL device.
///
/// `VectorIndex` owns the input [`RecordBatch`] so callers can inspect row
/// metadata through [`VectorIndex::batch`]. The vector values themselves are uploaded during
/// construction and their device handle is reused by every query.
pub struct VectorIndex {
    client: Client,
    batch: RecordBatch,
    vector_column: String,
    dimension: usize,
    vectors: VectorBuffer,
    no_filter_handle: Option<Handle>,
    options: IndexOptions,
}

impl VectorIndex {
    /// Validate and upload one Arrow vector column to `device`.
    ///
    /// The vector column must be a non-null `FixedSizeList<Float32>` or
    /// `FixedSizeList<Float16>`. Empty batches are valid and do not allocate a vector buffer.
    pub fn new(
        device: &Device,
        batch: RecordBatch,
        vector_column: impl Into<String>,
    ) -> Result<Self, QueryError> {
        Self::new_with_options(device, batch, vector_column, IndexOptions::default())
    }

    /// Construct an index with optional normalization of its resident vectors.
    ///
    /// Normalization uses FP64 norms on the host, then rounds back to the column's scalar type.
    /// Queries are not automatically normalized: cosine handles arbitrary query magnitudes, while
    /// inner product and squared L2 always evaluate the query supplied by the caller.
    pub fn new_with_options(
        device: &Device,
        batch: RecordBatch,
        vector_column: impl Into<String>,
        options: IndexOptions,
    ) -> Result<Self, QueryError> {
        let vector_column = vector_column.into();
        let vectors = vector_array(&batch, &vector_column)?;
        let dimension = vector_dimension(vectors)?;
        if batch.num_rows() >= INVALID_ROW as usize {
            return Err(QueryError::TooManyRows);
        }
        let client = device.client();

        let vectors = match vectors.value_type() {
            DataType::Float32 => {
                let values = vectors.values().as_primitive::<Float32Type>();
                let normalized = prepare_values(values.values(), dimension, options, |v| v, |v| v)?;
                let values_slice = normalized.as_deref().unwrap_or(values.values());
                let handle = (!values.is_empty())
                    .then(|| client.create_from_slice(f32::as_bytes(values_slice)));
                VectorBuffer::Float32 {
                    handle,
                    len: values.len(),
                }
            }
            DataType::Float16 => {
                let values = vectors.values().as_primitive::<Float16Type>();
                let normalized = prepare_values(
                    values.values(),
                    dimension,
                    options,
                    f16::to_f32,
                    f16::from_f32,
                )?;
                let values_slice = normalized.as_deref().unwrap_or(values.values());
                let handle = (!values.is_empty())
                    .then(|| client.create_from_slice(f16::as_bytes(values_slice)));
                VectorBuffer::Float16 {
                    handle,
                    len: values.len(),
                }
            }
            other => return Err(QueryError::UnsupportedVectorType(other)),
        };
        let no_filter_handle = (batch.num_rows() != 0)
            .then(|| client.create_from_slice(f32::as_bytes(&vec![0.0f32; batch.num_rows()])));

        Ok(Self {
            client,
            batch,
            vector_column,
            dimension,
            vectors,
            no_filter_handle,
            options,
        })
    }

    /// Query the resident vectors using exact inner product.
    ///
    /// The vector column remains resident. Only the FP32 query, optional filter values, and the
    /// final top-k results cross the device boundary.
    pub fn query(
        &self,
        query_vector: &[f32],
        options: IndexQueryOptions<'_>,
    ) -> Result<Vec<SearchHit>, QueryError> {
        self.query_with_metric(query_vector, options, Metric::InnerProduct)
    }

    /// Query using a selected metric and device-side exact top-k.
    ///
    /// Ties are ordered by ascending row ID. NaN scores caused by FP32 arithmetic overflow are
    /// excluded; infinite scores remain eligible. No full score array is read back to the host.
    pub fn query_with_metric(
        &self,
        query_vector: &[f32],
        options: IndexQueryOptions<'_>,
        metric: Metric,
    ) -> Result<Vec<SearchHit>, QueryError> {
        if query_vector.len() != self.dimension {
            return Err(QueryError::DimensionMismatch {
                query: query_vector.len(),
                vectors: self.dimension,
            });
        }
        if query_vector.iter().any(|v| !v.is_finite()) {
            return Err(QueryError::NonFiniteVector);
        }
        if options.top_k == 0 || self.batch.num_rows() == 0 {
            return Ok(Vec::new());
        }

        let query_handle = self.client.create_from_slice(f32::as_bytes(query_vector));
        let (filter_handle, filter_len, use_filter, filter_min) = match options.filter.as_ref() {
            Some(filter) => {
                let values = read_filter(&self.batch, filter)?;
                let handle = self.client.create_from_slice(f32::as_bytes(&values));
                (handle, values.len(), 1, filter.min)
            }
            None => (
                self.no_filter_handle
                    .as_ref()
                    .expect("non-empty index has a no-filter buffer")
                    .clone(),
                self.batch.num_rows(),
                0,
                0.0,
            ),
        };
        let scores_handle = self.client.empty(self.len() * size_of::<f32>());
        let rows_handle = self.client.empty(self.len() * size_of::<u32>());
        let cube_dim = CubeDim::new_1d(UNITS_PER_CUBE);
        let cube_count =
            calculate_cube_count_elemwise(&self.client, self.batch.num_rows(), cube_dim);
        match &self.vectors {
            VectorBuffer::Float32 { handle, len } => unsafe {
                exact_score::launch_unchecked::<f32>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    BufferArg::from_raw_parts(
                        handle
                            .as_ref()
                            .expect("non-empty index has a vector buffer")
                            .clone(),
                        *len,
                    ),
                    BufferArg::from_raw_parts(query_handle, query_vector.len()),
                    BufferArg::from_raw_parts(filter_handle, filter_len),
                    BufferArg::from_raw_parts(scores_handle.clone(), self.batch.num_rows()),
                    BufferArg::from_raw_parts(rows_handle.clone(), self.len()),
                    self.batch.num_rows(),
                    self.dimension,
                    use_filter,
                    filter_min,
                    metric.code(),
                );
            },
            VectorBuffer::Float16 { handle, len } => unsafe {
                exact_score::launch_unchecked::<f16>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    BufferArg::from_raw_parts(
                        handle
                            .as_ref()
                            .expect("non-empty index has a vector buffer")
                            .clone(),
                        *len,
                    ),
                    BufferArg::from_raw_parts(query_handle, query_vector.len()),
                    BufferArg::from_raw_parts(filter_handle, filter_len),
                    BufferArg::from_raw_parts(scores_handle.clone(), self.batch.num_rows()),
                    BufferArg::from_raw_parts(rows_handle.clone(), self.len()),
                    self.batch.num_rows(),
                    self.dimension,
                    use_filter,
                    filter_min,
                    metric.code(),
                );
            },
        }

        self.select_top_k(scores_handle, rows_handle, options.top_k, metric)
    }

    fn select_top_k(
        &self,
        mut scores: Handle,
        mut rows: Handle,
        top_k: usize,
        metric: Metric,
    ) -> Result<Vec<SearchHit>, QueryError> {
        let k = top_k.min(self.len());
        // At least two k-sized lists fit in a tile, ensuring progress even when k > 128.
        let tile_size = 256.max(k.saturating_mul(2)).min(self.len());
        let mut count = self.len();
        loop {
            let groups = count.div_ceil(tile_size);
            let output_len = groups * k;
            let output_scores = self.client.empty(output_len * size_of::<f32>());
            let output_rows = self.client.empty(output_len * size_of::<u32>());
            let cube_dim = CubeDim::new_1d(UNITS_PER_CUBE);
            // SAFETY: each unit owns one tile and its k output slots; all arrays have the
            // declared lengths. The next stage is queued after the preceding kernel.
            unsafe {
                tiled_top_k::launch_unchecked(
                    &self.client,
                    calculate_cube_count_elemwise(&self.client, groups, cube_dim),
                    cube_dim,
                    BufferArg::from_raw_parts(scores, count),
                    BufferArg::from_raw_parts(rows, count),
                    BufferArg::from_raw_parts(output_scores.clone(), output_len),
                    BufferArg::from_raw_parts(output_rows.clone(), output_len),
                    count,
                    groups,
                    tile_size,
                    k,
                    metric == Metric::SquaredL2,
                );
            }
            if groups == 1 {
                let buffers = self
                    .client
                    .read(vec![output_scores, output_rows])
                    .map_err(|error| QueryError::ReadScores(error.to_string()))?;
                let scores = f32::from_bytes(&buffers[0]);
                let rows = u32::from_bytes(&buffers[1]);
                return Ok(rows
                    .iter()
                    .zip(scores)
                    .filter(|(row, _)| **row != INVALID_ROW)
                    .map(|(&row, &score)| SearchHit {
                        row: row as usize,
                        score,
                    })
                    .collect());
            }
            scores = output_scores;
            rows = output_rows;
            count = output_len;
        }
    }

    /// Options used to prepare the resident vectors.
    pub fn options(&self) -> IndexOptions {
        self.options
    }

    /// The Arrow rows and metadata associated with this index.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Name of the resident vector column.
    pub fn vector_column(&self) -> &str {
        &self.vector_column
    }

    /// Number of scalar values in each indexed vector.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of indexed rows.
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    /// Whether the index contains no rows.
    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }
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
    if options.top_k == 0 {
        return Ok(Vec::new());
    }

    VectorIndex::new(device, batch.clone(), options.vector_column)?.query(
        query_vector,
        IndexQueryOptions {
            top_k: options.top_k,
            filter: options.filter,
        },
    )
}

fn vector_array<'a>(
    batch: &'a RecordBatch,
    vector_column: &str,
) -> Result<&'a FixedSizeListArray, QueryError> {
    let vectors = batch
        .column_by_name(vector_column)
        .ok_or_else(|| QueryError::MissingColumn(vector_column.to_string()))?;
    let vectors = vectors
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| QueryError::UnsupportedVectorType(vectors.data_type().clone()))?;

    if vectors.null_count() != 0 || vectors.values().null_count() != 0 {
        return Err(QueryError::NullVector);
    }

    Ok(vectors)
}

fn vector_dimension(vectors: &FixedSizeListArray) -> Result<usize, QueryError> {
    let dimension = usize::try_from(vectors.value_length()).expect("Arrow dimensions are positive");
    if dimension == 0 {
        return Err(QueryError::EmptyVector);
    }
    Ok(dimension)
}

fn prepare_values<T: Copy>(
    values: &[T],
    dimension: usize,
    options: IndexOptions,
    to_f32: impl Fn(T) -> f32,
    from_f32: impl Fn(f32) -> T,
) -> Result<Option<Vec<T>>, QueryError> {
    if values.iter().any(|&v| !to_f32(v).is_finite()) {
        return Err(QueryError::NonFiniteVector);
    }
    if !options.normalize {
        return Ok(None);
    }
    let mut normalized = Vec::with_capacity(values.len());
    for row in values.chunks_exact(dimension) {
        let norm = row
            .iter()
            .map(|&v| f64::from(to_f32(v)).powi(2))
            .sum::<f64>()
            .sqrt();
        normalized.extend(row.iter().map(|&v| {
            let value = f64::from(to_f32(v));
            from_f32(if norm == 0.0 {
                0.0
            } else {
                (value / norm) as f32
            })
        }));
    }
    Ok(Some(normalized))
}

fn read_filter(batch: &RecordBatch, filter: &FloatFilter<'_>) -> Result<Vec<f32>, QueryError> {
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
fn exact_score<F: Float>(
    vectors: &[F],
    query: &[f32],
    filter_values: &[f32],
    scores: &mut [f32],
    rows: &mut [u32],
    row_count: usize,
    dimension: usize,
    use_filter: u32,
    filter_min: f32,
    #[comptime] metric: u32,
) {
    let row = ABSOLUTE_POS;
    if row < row_count {
        rows[row] = u32::MAX;
        scores[row] = 0.0;
        let passes_filter = use_filter == 0 || filter_values[row] >= filter_min;
        if passes_filter {
            // Scale cosine operands before squaring to avoid overflowing norms for large
            // finite values or underflowing norms for tiny vectors.
            let mut vector_scale = 0.0f32;
            let mut query_scale = 0.0f32;
            if metric == 1 {
                for column in 0..dimension {
                    vector_scale = f32::max(
                        vector_scale,
                        f32::abs(f32::cast_from(vectors[row * dimension + column])),
                    );
                    query_scale = f32::max(query_scale, f32::abs(query[column]));
                }
            }
            let mut score = 0.0f32;
            let mut vector_norm = 0.0f32;
            let mut query_norm = 0.0f32;
            for column in 0..dimension {
                let mut value = f32::cast_from(vectors[row * dimension + column]);
                let mut query_value = query[column];
                if metric == 1 {
                    if vector_scale > 0.0 {
                        value /= vector_scale;
                    }
                    if query_scale > 0.0 {
                        query_value /= query_scale;
                    }
                    vector_norm += value * value;
                    query_norm += query_value * query_value;
                }
                if metric == 2 {
                    let difference = value - query_value;
                    score += difference * difference;
                } else {
                    score += value * query_value;
                }
            }
            if metric == 1 {
                if vector_norm > 0.0 && query_norm > 0.0 {
                    score /= f32::sqrt(vector_norm) * f32::sqrt(query_norm);
                    score = score.clamp(-1.0, 1.0);
                } else {
                    score = 0.0;
                }
            }
            scores[row] = score;
            if !score.is_nan() {
                rows[row] = u32::cast_from(row);
            }
        }
    }
}

/// One unit scans a tile k times, selecting the next best (score, row) pair on each pass.
/// Retaining k winners per tile is exact: a discarded row already has k better rows in its tile.
/// Merge stages repeat the operation until only k globally ranked results remain.
#[cube(launch_unchecked)]
fn tiled_top_k(
    scores: &[f32],
    rows: &[u32],
    output_scores: &mut [f32],
    output_rows: &mut [u32],
    count: usize,
    groups: usize,
    tile_size: usize,
    k: usize,
    #[comptime] ascending: bool,
) {
    let group = ABSOLUTE_POS;
    if group < groups {
        let start = group * tile_size;
        let end = start + usize::min(tile_size, count - start);
        let mut previous_score = 0.0f32;
        let mut previous_row = 4_294_967_295u32;
        let mut exhausted = false;
        for rank in 0..k {
            let mut best_score = 0.0f32;
            let mut best_row = 4_294_967_295u32;
            for position in start..end {
                let row = rows[position];
                let score = scores[position];
                let mut after_previous = score < previous_score;
                let mut better = score > best_score;
                if ascending {
                    after_previous = score > previous_score;
                    better = score < best_score;
                }
                // Only consider candidates strictly after the previously emitted result.
                // Ties use ascending original row IDs, so equal scores require a larger row ID.
                // This avoids selecting an earlier winner again without modifying the inputs.
                after_previous = after_previous || (score == previous_score && row > previous_row);
                // Among eligible candidates, keep the best for this rank: a better score or,
                // on a tie, a smaller row ID. An invalid best_row accepts the first candidate,
                // even if its score is worse than the initial best_score of zero.
                better = best_row == u32::MAX || better || (score == best_score && row < best_row);
                // Rank zero has no previous result, so it skips the after_previous boundary.
                if !exhausted && row != u32::MAX && (rank == 0 || after_previous) && better {
                    best_score = score;
                    best_row = row;
                }
            }
            output_scores[group * k + rank] = best_score;
            output_rows[group * k + rank] = best_row;
            previous_score = best_score;
            previous_row = best_row;
            exhausted = best_row == u32::MAX;
        }
    }
}
