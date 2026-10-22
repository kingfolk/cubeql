/// Read configuration settings from a file on disk.
fn load_config() -> String {
    std::fs::read_to_string("config.toml").unwrap()
}

/// Save configuration settings to a file on disk.
fn save_config(text: &str) {
    std::fs::write("config.toml", text).unwrap();
}

/// Multiply two integers and return their product.
fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
