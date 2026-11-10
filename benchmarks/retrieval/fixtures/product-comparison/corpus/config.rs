pub fn load_config(path: &std::path::Path) -> std::io::Result<String> {
    std::fs::read_to_string(path)
}
