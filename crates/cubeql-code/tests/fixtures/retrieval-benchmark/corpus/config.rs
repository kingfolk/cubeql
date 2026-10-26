/// Read application configuration text from a file on disk.
pub fn load_config(path: &std::path::Path) -> std::io::Result<String> {
    std::fs::read_to_string(path)
}

/// Save application configuration text to a file on disk.
pub fn save_config(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    std::fs::write(path, text)
}

/// Select the server port, using a default when no override was provided.
pub fn server_port(override_port: Option<u16>) -> u16 {
    override_port.unwrap_or(8080)
}

/// Check whether the application is configured to emit verbose logs.
pub fn verbose_logging(level: &str) -> bool {
    level == "debug" || level == "trace"
}

/// Read a named environment variable, or return a supplied fallback string.
pub fn environment_or(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.to_owned())
}

/// Determine the user cache directory beneath an application root.
pub fn cache_directory(root: &std::path::Path) -> std::path::PathBuf {
    root.join("cache")
}
