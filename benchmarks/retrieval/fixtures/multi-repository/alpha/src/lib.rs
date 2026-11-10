pub fn load_config(path: &std::path::Path) -> std::io::Result<String> {
    std::fs::read_to_string(path)
}

pub fn rank_hits(hits: &mut [(usize, f32)]) {
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

pub fn server_port(value: Option<u16>) -> u16 {
    value.unwrap_or(8080)
}
