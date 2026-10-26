/// Compute the product of two floating point numbers.
pub fn multiply(a: f32, b: f32) -> f32 { a * b }

/// Clamp a percentage to the inclusive range zero through one hundred.
pub fn clamp_percentage(value: f32) -> f32 { value.clamp(0.0, 100.0) }

/// Count the number of Unicode characters in a string.
pub fn character_count(text: &str) -> usize { text.chars().count() }

/// Convert a string to lowercase for display purposes.
pub fn lowercase(text: &str) -> String { text.to_lowercase() }

/// Convert milliseconds into seconds.
pub fn seconds(milliseconds: u64) -> f64 { milliseconds as f64 / 1000.0 }

/// Calculate the arithmetic mean, returning zero for no samples.
pub fn average(samples: &[f32]) -> f32 {
    if samples.is_empty() { 0.0 } else { samples.iter().sum::<f32>() / samples.len() as f32 }
}
