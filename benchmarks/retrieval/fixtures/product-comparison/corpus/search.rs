pub fn rank_hits(hits: &mut [(usize, f32)]) {
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}
