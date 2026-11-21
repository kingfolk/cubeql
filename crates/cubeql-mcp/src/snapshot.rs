//! Bounded, side-effect-free preflight for a replacement code snapshot.
//!
//! This module reads one regular file into a bounded byte buffer, hashes those exact bytes,
//! and parses the same bytes as CubeQL IPC. It does not create a device or embedding provider.

use std::{
    fs::{File, OpenOptions},
    io::{Cursor, Read},
    path::Path,
};

use cubeql_code::CodeIndex;
use cubeql_embeddings::EmbeddingConfig;

use crate::contract::ToolFailure;

/// Maximum configured snapshot size accepted by the MCP reload path (512 MiB).
pub const MAX_RELOAD_BYTES: u64 = 512 * 1024 * 1024;

/// Fully parsed candidate and its content-derived identity.
pub struct SnapshotCandidate {
    pub snapshot_id: String,
    pub code: CodeIndex,
}

/// Read and validate a candidate snapshot without changing any server state.
///
/// The hash and parse both use the same bounded byte buffer, so the returned identity always
/// names the parsed contents. A provider-aware configuration is required only when the active
/// server has an embedding configuration; host-only mode accepts legacy provider-less snapshots.
pub fn load_candidate(
    path: &Path,
    expected_embedding: Option<&EmbeddingConfig>,
    max_bytes: u64,
) -> Result<SnapshotCandidate, ToolFailure> {
    if max_bytes == 0 || max_bytes == u64::MAX {
        return Err(reload_failed("snapshot byte limit must be in 1..u64::MAX"));
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|error| reload_failed(format!("cannot open snapshot: {error}")))?;
    read_candidate(file, path, expected_embedding, max_bytes)
}

fn read_candidate(
    mut file: File,
    path: &Path,
    expected_embedding: Option<&EmbeddingConfig>,
    max_bytes: u64,
) -> Result<SnapshotCandidate, ToolFailure> {
    let metadata = file
        .metadata()
        .map_err(|error| reload_failed(format!("cannot inspect snapshot: {error}")))?;
    if !metadata.is_file() {
        return Err(reload_failed("snapshot path must name a regular file"));
    }
    if metadata.len() > max_bytes {
        return Err(reload_failed(format!(
            "snapshot is {} bytes; configured limit is {max_bytes}",
            metadata.len()
        )));
    }

    let read_limit = max_bytes + 1;
    let mut bytes = Vec::with_capacity(metadata.len().min(read_limit) as usize);
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            reload_failed(format!("cannot read snapshot {}: {error}", path.display()))
        })?;
    if bytes.len() as u64 > max_bytes {
        return Err(reload_failed(format!(
            "snapshot grew beyond the configured {max_bytes}-byte limit while being read"
        )));
    }

    let snapshot_id = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let code = CodeIndex::read_ipc(Cursor::new(bytes))
        .map_err(|error| reload_failed(format!("invalid code snapshot: {error}")))?;
    if let Some(expected) = expected_embedding {
        let actual = code.embedding_config().ok_or_else(|| {
            embedding_mismatch("snapshot has no provider-aware embedding configuration")
        })?;
        expected.ensure_compatible(actual).map_err(|_| {
            embedding_mismatch(
                "snapshot embedding configuration is incompatible with the active provider",
            )
        })?;
    }

    Ok(SnapshotCandidate { snapshot_id, code })
}

fn reload_failed(message: impl Into<String>) -> ToolFailure {
    ToolFailure::new("reload_failed", message)
}

fn embedding_mismatch(message: impl Into<String>) -> ToolFailure {
    ToolFailure::new("embedding_mismatch", message)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use cubeql_code::{
        BuildOptions, build_repository, build_repository_with_provider, hash_embedding,
    };
    use cubeql_embeddings::{EmbeddingConfig, EmbeddingProvider, HashProvider};

    use super::*;

    fn provider_snapshot(source: &str) -> (Vec<u8>, EmbeddingConfig) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("lib.rs"), source).unwrap();
        let mut provider = HashProvider::new(16).unwrap();
        let config = provider.config().clone();
        let options = BuildOptions::for_embedding(&config).unwrap();
        let code = build_repository_with_provider(dir.path(), &options, None, &mut provider, 2)
            .unwrap()
            .index;
        let mut bytes = Vec::new();
        code.write_ipc(&mut bytes).unwrap();
        (bytes, config)
    }

    fn write_temp(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.arrow");
        fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    fn failure(result: Result<SnapshotCandidate, ToolFailure>) -> ToolFailure {
        match result {
            Ok(_) => panic!("expected snapshot preflight to fail"),
            Err(error) => error,
        }
    }

    #[test]
    fn identity_is_hash_of_exact_parsed_bytes_and_changes_with_snapshot_bytes() {
        let (first_bytes, config) = provider_snapshot("pub fn alpha() {}\n");
        let (dir, path) = write_temp(&first_bytes);
        let first = load_candidate(&path, Some(&config), MAX_RELOAD_BYTES).unwrap();
        assert_eq!(
            first.snapshot_id,
            format!("blake3:{}", blake3::hash(&first_bytes).to_hex())
        );
        assert_eq!(first.code.chunks().len(), 1);
        assert_eq!(
            load_candidate(&path, Some(&config), MAX_RELOAD_BYTES)
                .unwrap()
                .snapshot_id,
            first.snapshot_id
        );

        let (second_bytes, _) = provider_snapshot("pub fn beta() {}\n");
        fs::write(&path, &second_bytes).unwrap();
        let second = load_candidate(&path, Some(&config), MAX_RELOAD_BYTES).unwrap();
        assert_ne!(first.snapshot_id, second.snapshot_id);
        assert_ne!(first_bytes, second_bytes);
        drop(dir);
    }

    #[test]
    fn requires_exact_embedding_contract_when_expected() {
        let (bytes, config) = provider_snapshot("pub fn alpha() {}\n");
        let (_dir, path) = write_temp(&bytes);
        let mutators: [fn(&mut EmbeddingConfig); 3] = [
            |c: &mut EmbeddingConfig| c.dimensions += 1,
            |c: &mut EmbeddingConfig| c.document_prefix.push_str("changed"),
            |c: &mut EmbeddingConfig| c.implementation.push_str("-changed"),
        ];
        for mutate in mutators {
            let mut expected = config.clone();
            mutate(&mut expected);
            assert_eq!(
                failure(load_candidate(&path, Some(&expected), MAX_RELOAD_BYTES)).code,
                "embedding_mismatch"
            );
        }
        assert!(load_candidate(&path, Some(&config), MAX_RELOAD_BYTES).is_ok());
    }

    #[test]
    fn host_only_accepts_providerless_legacy_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("lib.rs"), "pub fn alpha() {}\n").unwrap();
        let options = BuildOptions {
            dimensions: 16,
            embedding_id: "legacy".into(),
            fallback_lines: 8,
            ..BuildOptions::default()
        };
        let code = build_repository(dir.path(), &options, None, |text| hash_embedding(text, 16))
            .unwrap()
            .index;
        let mut bytes = Vec::new();
        code.write_ipc(&mut bytes).unwrap();
        let (_tmp, path) = write_temp(&bytes);
        let loaded = load_candidate(&path, None, MAX_RELOAD_BYTES).unwrap();
        assert!(loaded.code.embedding_config().is_none());
        let expected = HashProvider::new(16).unwrap();
        assert_eq!(
            failure(load_candidate(
                &path,
                Some(expected.config()),
                MAX_RELOAD_BYTES
            ))
            .code,
            "embedding_mismatch"
        );
        assert_eq!(
            loaded.snapshot_id,
            format!("blake3:{}", blake3::hash(&bytes).to_hex())
        );
    }

    #[test]
    fn errors_for_missing_corrupt_oversize_zero_max_and_directory() {
        let absent = std::env::temp_dir().join(format!("cubeql-missing-{}", std::process::id()));
        let missing_failure = failure(load_candidate(&absent, None, MAX_RELOAD_BYTES));
        assert_eq!(missing_failure.code, "reload_failed");

        let (_dir, path) = write_temp(b"not an Arrow IPC snapshot");
        assert_eq!(
            failure(load_candidate(&path, None, MAX_RELOAD_BYTES)).code,
            "reload_failed"
        );
        assert_eq!(
            failure(load_candidate(&path, None, 4)).code,
            "reload_failed"
        );
        assert_eq!(
            failure(load_candidate(&path, None, 0)).code,
            "reload_failed"
        );
        assert_eq!(
            failure(load_candidate(&path, None, u64::MAX)).code,
            "reload_failed"
        );
        assert_eq!(
            failure(load_candidate(_dir.path(), None, MAX_RELOAD_BYTES)).code,
            "reload_failed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn fifo_fails_without_blocking() {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("snapshot.fifo");
        let c_path = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let failure = failure(load_candidate(&fifo, None, MAX_RELOAD_BYTES));
        assert_eq!(failure.code, "reload_failed");
    }

    #[test]
    fn failed_preflight_does_not_change_input_bytes() {
        let (bytes, config) = provider_snapshot("pub fn alpha() {}\n");
        let (_dir, path) = write_temp(&bytes);
        let before = fs::read(&path).unwrap();
        let mut mismatch = config;
        mismatch.query_prefix.push_str("wrong");
        assert_eq!(
            failure(load_candidate(&path, Some(&mismatch), MAX_RELOAD_BYTES)).code,
            "embedding_mismatch"
        );
        assert_eq!(fs::read(path).unwrap(), before);
    }
}
