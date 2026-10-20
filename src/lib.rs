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
use cubecl::{Device, calculate_cube_count_elemwise, client::Client, prelude::*, server::Handle};
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

enum VectorBuffer {
    Float32 { handle: Option<Handle>, len: usize },
    Float16 { handle: Option<Handle>, len: usize },
}

/// An Arrow vector column uploaded once and retained on a CubeCL device.
///
/// `VectorIndex` owns a cheap clone of the input [`RecordBatch`] so callers can inspect row
/// metadata through [`VectorIndex::batch`]. The vector values themselves are uploaded during
/// construction and their device handle is reused by every query.
pub struct VectorIndex {
    client: Client,
    batch: RecordBatch,
    vector_column: String,
    dimension: usize,
    vectors: VectorBuffer,
    no_filter_handle: Option<Handle>,
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
        let vector_column = vector_column.into();
        let vectors = vector_array(&batch, &vector_column)?;
        let dimension = vector_dimension(vectors)?;
        let client = device.client();

        let vectors = match vectors.value_type() {
            DataType::Float32 => {
                let values = vectors.values().as_primitive::<Float32Type>();
                let handle = (!values.is_empty())
                    .then(|| client.create_from_slice(f32::as_bytes(values.values())));
                VectorBuffer::Float32 {
                    handle,
                    len: values.len(),
                }
            }
            DataType::Float16 => {
                let values = vectors.values().as_primitive::<Float16Type>();
                let handle = (!values.is_empty())
                    .then(|| client.create_from_slice(f16::as_bytes(values.values())));
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
        })
    }

    /// Query the resident vectors using exact inner product.
    ///
    /// Only per-query inputs and outputs are transferred: the FP32 query vector, optional filter
    /// values, and the score buffer. The vector column remains resident on the device.
    pub fn query(
        &self,
        query_vector: &[f32],
        options: IndexQueryOptions<'_>,
    ) -> Result<Vec<SearchHit>, QueryError> {
        if query_vector.len() != self.dimension {
            return Err(QueryError::DimensionMismatch {
                query: query_vector.len(),
                vectors: self.dimension,
            });
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
        let initial_scores = vec![f32::NEG_INFINITY; self.batch.num_rows()];
        let scores_handle = self
            .client
            .create_from_slice(f32::as_bytes(&initial_scores));
        let cube_dim = CubeDim::new_1d(UNITS_PER_CUBE);
        let cube_count =
            calculate_cube_count_elemwise(&self.client, self.batch.num_rows(), cube_dim);
        match &self.vectors {
            VectorBuffer::Float32 { handle, len } => unsafe {
                exact_inner_product::launch_unchecked::<f32>(
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
                    self.batch.num_rows(),
                    self.dimension,
                    use_filter,
                    filter_min,
                );
            },
            VectorBuffer::Float16 { handle, len } => unsafe {
                exact_inner_product::launch_unchecked::<f16>(
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
                    self.batch.num_rows(),
                    self.dimension,
                    use_filter,
                    filter_min,
                );
            },
        }

        let score_bytes = self
            .client
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
