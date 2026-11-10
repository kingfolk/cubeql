pub fn write_config(path: &std::path::Path, value: &str) -> std::io::Result<()> {
    std::fs::write(path, value)
}
