use std::{fs, io::Cursor};

use cubeql::{IndexOptions, write_ipc_batch};
use cubeql_code::{BuildOptions, CodeIndex, VectorPrecision, build_repository, hash_embedding};

const FILE_HASHES: &str = "cubeql.code.file_hashes";

fn build(root: &std::path::Path, previous: Option<&CodeIndex>) -> cubeql_code::BuildResult {
    let options = BuildOptions::default();
    build_repository(root, &options, previous, |text| {
        hash_embedding(text, options.dimensions)
    })
    .unwrap()
}

fn from_metadata(
    code: &CodeIndex,
    manifest: Option<&str>,
) -> Result<CodeIndex, cubeql_code::CodeError> {
    let mut batch = code.batch().clone();
    match manifest {
        Some(value) => {
            batch.schema_metadata_mut().insert(FILE_HASHES, value);
        }
        None => {
            batch.schema_metadata_mut().remove(FILE_HASHES);
        }
    }
    let mut bytes = Vec::new();
    write_ipc_batch(&batch, "embedding", IndexOptions::default(), &mut bytes).unwrap();
    CodeIndex::read_ipc(Cursor::new(bytes))
}

#[test]
fn file_hashes_cover_original_bytes_and_survive_ipc_and_precision_conversion() {
    let root = tempfile::tempdir().unwrap();
    let source =
        "\u{feff}// unchunked prefix 配置🦀\r\npub fn target() {}\r\n// trailing comment\r\n";
    fs::write(root.path().join("lib.rs"), source).unwrap();
    fs::write(root.path().join("empty.rs"), "").unwrap();
    fs::write(root.path().join("binary.rs"), [0, 1, 2]).unwrap();
    let built = build(root.path(), None);
    let hash = blake3::hash(source.as_bytes()).to_hex().to_string();
    assert_eq!(built.index.file_hash("lib.rs"), Some(hash.as_str()));
    assert_eq!(
        built.index.file_hash("empty.rs"),
        Some(blake3::hash(b"").to_hex().as_str())
    );
    assert_eq!(built.index.file_hash("binary.rs"), None);
    assert_ne!(built.index.chunks()[0].content_hash, hash);
    for precision in [VectorPrecision::Fp16, VectorPrecision::Fp32] {
        let code = built.index.clone().with_precision(precision).unwrap();
        let mut bytes = Vec::new();
        code.write_ipc(&mut bytes).unwrap();
        let loaded = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
        assert_eq!(loaded.file_hash("lib.rs"), Some(hash.as_str()));
    }
}

#[test]
fn comment_only_changes_refresh_file_evidence_without_reembedding_unchanged_chunks() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("lib.rs"), "pub fn target() {}\n").unwrap();
    let before = build(root.path(), None);
    let changed = "// inserted outside the function\npub fn target() {}\n";
    fs::write(root.path().join("lib.rs"), changed).unwrap();
    let after = build(root.path(), Some(&before.index));
    assert_eq!(after.report.embedded, 0);
    assert_eq!(after.report.reused, 1);
    assert_eq!(
        before.index.chunks()[0].content_hash,
        after.index.chunks()[0].content_hash
    );
    assert_ne!(
        before.index.file_hash("lib.rs"),
        after.index.file_hash("lib.rs")
    );
    assert_eq!(
        after.index.file_hash("lib.rs"),
        Some(blake3::hash(changed.as_bytes()).to_hex().as_str())
    );
    assert_eq!(after.index.chunks()[0].start_line, 2);
}

#[test]
fn old_snapshots_remain_loadable_and_updates_add_missing_file_evidence() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("lib.rs"), "pub fn target() {}\n").unwrap();
    let code = build(root.path(), None).index;
    let legacy = from_metadata(&code, None).unwrap();
    assert_eq!(legacy.file_hash("lib.rs"), None);
    assert_eq!(legacy.chunks(), code.chunks());
    let updated = build(root.path(), Some(&legacy));
    assert_eq!(updated.report.embedded, 0);
    assert_eq!(updated.index.file_hash("lib.rs"), code.file_hash("lib.rs"));
}

#[test]
fn malformed_or_incomplete_present_manifests_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("lib.rs"), "pub fn target() {}\n").unwrap();
    let code = build(root.path(), None).index;
    for value in [
        "null",
        "[]",
        "{",
        "{}",
        r#"{"lib.rs":"not-a-hash"}"#,
        r#"{"lib.rs":123}"#,
    ] {
        assert!(from_metadata(&code, Some(value)).is_err(), "{value}");
    }
    let upper = format!(r#"{{"lib.rs":"{}"}}"#, "A".repeat(64));
    assert!(from_metadata(&code, Some(&upper)).is_err());
}
