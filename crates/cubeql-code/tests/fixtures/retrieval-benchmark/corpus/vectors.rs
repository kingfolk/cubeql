/// Normalize an embedding to unit length. Leave a zero vector unchanged to avoid division by zero.
pub fn normalize_vector(values: &mut [f32]) {
    let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values { *value /= norm; }
    }
}

/// Measure directional similarity using the dot product divided by both vector lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let aa: f32 = a.iter().map(|x| x * x).sum();
    let bb: f32 = b.iter().map(|x| x * x).sum();
    let denominator = (aa * bb).sqrt();
    if denominator == 0.0 { 0.0 } else { dot / denominator }
}

/// Measure squared Euclidean distance between two vectors.
pub fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Multiply a vector by a scalar gain.
pub fn scale_vector(values: &mut [f32], gain: f32) {
    for value in values { *value *= gain; }
}

/// Return the largest element of a vector, or none for an empty vector.
pub fn maximum_value(values: &[f32]) -> Option<f32> {
    values.iter().copied().max_by(f32::total_cmp)
}

/// Add corresponding components of two vectors.
pub fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
