//! Explicitly authorized, directory-relative workspace checks. Never derive ambient authority
//! from snapshot metadata; the operator must supply --workspace-root at startup.
use std::{
    io::{self, Read},
    path::Path,
};

use cap_std::fs::{Dir, OpenOptions};

use crate::contract::Freshness;

pub(crate) const MAX_CHECK_BYTES: u64 = 16 * 1024 * 1024;

pub(crate) struct Check {
    pub status: Freshness,
    pub reason: &'static str,
    pub actual_hash: Option<String>,
    pub bytes: u64,
}
impl Check {
    fn unknown(reason: &'static str, bytes: u64) -> Self {
        Self {
            status: Freshness::Unverifiable,
            reason,
            actual_hash: None,
            bytes,
        }
    }
}

pub(crate) fn check_file(
    root: Option<&Dir>,
    path: &str,
    expected: Option<&str>,
    limit: u64,
) -> Check {
    let Some(root) = root else {
        return Check::unknown("workspace_not_configured", 0);
    };
    let Some(expected) = expected else {
        return Check::unknown("legacy_snapshot", 0);
    };
    // Treat snapshot metadata as untrusted too. Reject ambiguous cross-platform paths before
    // asking cap-std to resolve anything; cap-std additionally bounds symlink traversal.
    if !safe_relative_path(path) {
        return Check::unknown("unsafe_path", 0);
    }
    let limit = limit.min(MAX_CHECK_BYTES);
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use cap_std::fs::OpenOptionsExt;
        // Opening a substituted FIFO must not block the sole retrieval worker before we can
        // inspect its file type. Regular-file reads are unaffected by O_NONBLOCK.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = match root.open_with(path, &options) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Check {
                status: Freshness::Missing,
                reason: "file_not_found",
                actual_hash: None,
                bytes: 0,
            };
        }
        Err(_) => return Check::unknown("inaccessible_or_outside_workspace", 0),
    };
    let before = match file.metadata() {
        Ok(meta) if meta.is_file() => meta,
        Ok(_) => return Check::unknown("not_regular_file", 0),
        Err(_) => return Check::unknown("metadata_unavailable", 0),
    };
    if before.len() > limit {
        return Check::unknown("size_limit", 0);
    }
    let mut reader = file.take(limit + 1);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Check::unknown("read_failed", bytes),
        };
        if count == 0 {
            break;
        }
        bytes += count as u64;
        if bytes > limit {
            return Check::unknown("size_limit", bytes);
        }
        hasher.update(&buffer[..count]);
    }
    let after = match reader.get_ref().metadata() {
        Ok(meta) => meta,
        Err(_) => return Check::unknown("metadata_unavailable", bytes),
    };
    if before.len() != after.len()
        || bytes != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Check::unknown("changed_during_check", bytes);
    }
    let actual = hasher.finalize().to_hex().to_string();
    let unchanged = actual == expected;
    Check {
        status: if unchanged {
            Freshness::Unchanged
        } else {
            Freshness::Modified
        },
        reason: if unchanged {
            "file_hash_match"
        } else {
            "file_hash_mismatch"
        },
        actual_hash: Some(actual),
        bytes,
    }
}

fn safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !Path::new(path).is_absolute()
        && !path.contains(['\\', ':', '\0'])
        && path.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguous_snapshot_paths_are_never_opened() {
        for path in [
            "",
            "/etc/passwd",
            "../secret.rs",
            "src/../secret.rs",
            "C:/secret.rs",
            "src\\file.rs",
            "./file.rs",
            "src//file.rs",
        ] {
            assert!(!safe_relative_path(path), "{path}");
        }
        assert!(safe_relative_path("src/配置.rs"));
    }

    #[test]
    fn absence_of_evidence_never_means_unchanged() {
        let root = tempfile::tempdir().unwrap();
        let dir = Dir::open_ambient_dir(root.path(), cap_std::ambient_authority()).unwrap();
        let hash = blake3::hash(b"saved").to_hex().to_string();
        assert_eq!(
            check_file(None, "lib.rs", Some(&hash), 100).reason,
            "workspace_not_configured"
        );
        assert_eq!(
            check_file(Some(&dir), "lib.rs", None, 100).reason,
            "legacy_snapshot"
        );
        assert_eq!(
            check_file(Some(&dir), "../lib.rs", Some(&hash), 100).reason,
            "unsafe_path"
        );
        std::fs::write(root.path().join("lib.rs"), "larger than the limit").unwrap();
        assert_eq!(
            check_file(Some(&dir), "lib.rs", Some(&hash), 4).reason,
            "size_limit"
        );
        std::fs::create_dir(root.path().join("directory.rs")).unwrap();
        assert_eq!(
            check_file(Some(&dir), "directory.rs", Some(&hash), 100).reason,
            "not_regular_file"
        );
    }
}
