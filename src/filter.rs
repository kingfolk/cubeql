use std::collections::HashSet;

use arrow_array::{
    Array, DictionaryArray, Float16Array, Float32Array, Float64Array, LargeStringArray,
    RecordBatch, StringArray, StringViewArray, types::*,
};
use arrow_schema::DataType;

use crate::QueryError;

/// A metadata predicate. Every predicate in a query must match (logical AND).
/// Nulls never match. Numeric predicates support Float16, Float32 and Float64 columns;
/// string predicates support Utf8, LargeUtf8, Utf8View and dictionaries of these types.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter<'a> {
    /// Exact numeric equality after lossless conversion of the stored float to FP64.
    FloatEq { column: &'a str, value: f64 },
    /// Inclusive bounds. `None` leaves that side unbounded. NaN column values never match.
    FloatRange {
        column: &'a str,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Case-sensitive string equality.
    StringEq { column: &'a str, value: &'a str },
    /// Membership in a set of case-sensitive strings. An empty set matches no rows.
    StringIn {
        column: &'a str,
        values: &'a [&'a str],
    },
    /// Literal string prefix, without filesystem normalization. Use a trailing `/` for directories.
    PathPrefix { column: &'a str, prefix: &'a str },
}

/// Where a prepared metadata selection takes effect in device execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterMode {
    /// Score a compact list of matching row IDs. Intermediate buffers scale with matching rows.
    #[default]
    BeforeScoring,
    /// Launch over all rows and check a prepared mask before computing each vector score.
    DuringScoring,
}

impl Filter<'_> {
    fn column(&self) -> &str {
        match self {
            Self::FloatEq { column, .. }
            | Self::FloatRange { column, .. }
            | Self::StringEq { column, .. }
            | Self::StringIn { column, .. }
            | Self::PathPrefix { column, .. } => column,
        }
    }
}

pub(crate) fn evaluate(
    batch: &RecordBatch,
    filters: &[Filter<'_>],
) -> Result<Vec<u32>, QueryError> {
    let mut mask = vec![1; batch.num_rows()];
    // Validate every predicate even if an earlier one matches no rows.
    for filter in filters {
        let column = filter.column();
        let array = batch
            .column_by_name(column)
            .ok_or_else(|| QueryError::MissingColumn(column.to_string()))?;
        let matches = match filter {
            Filter::FloatEq { value, .. } => {
                if value.is_nan() {
                    return Err(QueryError::InvalidFilter(column.to_string()));
                }
                float_matches(array.as_ref(), |v| v == *value)?
            }
            Filter::FloatRange { min, max, .. } => {
                if min.is_some_and(f64::is_nan)
                    || max.is_some_and(f64::is_nan)
                    || matches!((min, max), (Some(min), Some(max)) if min > max)
                {
                    return Err(QueryError::InvalidFilter(column.to_string()));
                }
                float_matches(array.as_ref(), |v| {
                    !v.is_nan() && min.is_none_or(|min| v >= min) && max.is_none_or(|max| v <= max)
                })?
            }
            Filter::StringEq { value, .. } => string_matches(array.as_ref(), &|v| v == *value)?,
            Filter::StringIn { values, .. } => {
                let accepted: HashSet<&str> = values.iter().copied().collect();
                string_matches(array.as_ref(), &|v| accepted.contains(v))?
            }
            Filter::PathPrefix { prefix, .. } => {
                string_matches(array.as_ref(), &|v| v.starts_with(prefix))?
            }
        };
        for (keep, matched) in mask.iter_mut().zip(matches) {
            *keep &= u32::from(matched);
        }
    }
    Ok(mask)
}

fn float_matches(
    array: &dyn Array,
    matches: impl Fn(f64) -> bool,
) -> Result<Vec<bool>, QueryError> {
    Ok(match array.data_type() {
        DataType::Float16 => array
            .as_any()
            .downcast_ref::<Float16Array>()
            .unwrap()
            .iter()
            .map(|v| v.is_some_and(|v| matches(f64::from(v.to_f32()))))
            .collect(),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .iter()
            .map(|v| v.is_some_and(|v| matches(f64::from(v))))
            .collect(),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .map(|v| v.is_some_and(&matches))
            .collect(),
        other => return Err(QueryError::UnsupportedMetadataType(other.clone())),
    })
}

fn string_matches(
    array: &dyn Array,
    matches: &impl Fn(&str) -> bool,
) -> Result<Vec<bool>, QueryError> {
    Ok(match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.is_some_and(matches))
            .collect(),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap()
            .iter()
            .map(|v| v.is_some_and(matches))
            .collect(),
        DataType::Utf8View => array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .iter()
            .map(|v| v.is_some_and(matches))
            .collect(),
        DataType::Dictionary(key, _) => {
            macro_rules! dictionary {
                ($ty:ty) => {{
                    let dictionary = array
                        .as_any()
                        .downcast_ref::<DictionaryArray<$ty>>()
                        .unwrap();
                    // Compare each distinct string once, then inspect integer keys per row.
                    let accepted = string_matches(dictionary.values().as_ref(), matches)?;
                    dictionary
                        .keys_iter()
                        .map(|key| key.is_some_and(|key| accepted[key]))
                        .collect()
                }};
            }
            match key.as_ref() {
                DataType::Int8 => dictionary!(Int8Type),
                DataType::Int16 => dictionary!(Int16Type),
                DataType::Int32 => dictionary!(Int32Type),
                DataType::Int64 => dictionary!(Int64Type),
                DataType::UInt8 => dictionary!(UInt8Type),
                DataType::UInt16 => dictionary!(UInt16Type),
                DataType::UInt32 => dictionary!(UInt32Type),
                DataType::UInt64 => dictionary!(UInt64Type),
                _ => {
                    return Err(QueryError::UnsupportedMetadataType(
                        array.data_type().clone(),
                    ));
                }
            }
        }
        other => return Err(QueryError::UnsupportedMetadataType(other.clone())),
    })
}
