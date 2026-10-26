/// Order search hits by descending similarity score, with smaller row IDs winning ties.
pub fn rank_hits(hits: &mut [(usize, f32)]) {
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

/// Keep source paths under a literal directory prefix.
pub fn filter_paths<'a>(paths: &'a [String], prefix: &str) -> Vec<&'a str> {
    paths.iter().filter(|p| p.starts_with(prefix)).map(String::as_str).collect()
}

/// Restrict metadata rows to one programming language.
pub fn filter_language<'a>(languages: &'a [String], language: &str) -> Vec<usize> {
    languages.iter().enumerate().filter(|(_, value)| *value == language).map(|(row, _)| row).collect()
}

/// Discard repeated row IDs while preserving their first appearance.
pub fn deduplicate_rows(rows: &[usize]) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    rows.iter().copied().filter(|row| seen.insert(*row)).collect()
}

/// Select one page from an already ordered hit list.
pub fn page_hits(hits: &[usize], offset: usize, limit: usize) -> Vec<usize> {
    hits.iter().skip(offset).take(limit).copied().collect()
}

/// Produce a human-readable source location with a one-based line number.
pub fn source_location(path: &str, line: u32) -> String {
    format!("{path}:{line}")
}
