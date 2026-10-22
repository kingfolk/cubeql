use std::{fs, io::Cursor, path::Path, process::Command};

use arrow_array::{Array, FixedSizeListArray, Float32Array};
use cubeql::{Filter, IndexOptions, Metric, VectorIndex, write_ipc_batch};
use cubeql_code::{
    BuildOptions, BuildResult, CodeError, CodeIndex, build_repository, hash_embedding,
};

fn write(root: &Path, path: &str, content: &str) {
    let path = root.join(path);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn build(root: &Path, previous: Option<&CodeIndex>) -> BuildResult {
    let options = BuildOptions::default();
    build_repository(root, &options, previous, |text| {
        hash_embedding(text, options.dimensions)
    })
    .unwrap()
}

#[test]
fn rust_syntax_chunks_preserve_exact_source_and_locations() {
    let dir = tempfile::tempdir().unwrap();
    let source = "// 中文 🦀\r\n\
        /// Documentation\r\n\
        #[inline]\r\n\
        pub fn café() -> &'static str { \"你好\" }\r\n\
        pub mod nested {\r\n\
            pub struct Store { value: u32 }\r\n\
            impl Store {\r\n\
                pub fn score(&self) -> u32 { self.value }\r\n\
            }\r\n\
            pub trait Search {\r\n\
                fn search(&self);\r\n\
            }\r\n\
            impl Search for Store {\r\n\
                fn search(&self) {}\r\n\
            }\r\n\
        }\r\n\
        mod external;\r\n\
        macro_rules! example { () => {} }\r\n";
    write(dir.path(), "src/lib.rs", source);
    let result = build(dir.path(), None);
    assert!(result.report.warnings.is_empty());
    let chunks = result.index.chunks();
    let get = |symbol: &str| chunks.iter().find(|c| c.symbol == symbol).unwrap();
    let function = get("café");
    assert_eq!(function.start_line, 2);
    assert_eq!(function.end_line, 4);
    assert!(function.source.starts_with("/// Documentation"));
    assert!(function.source.ends_with("{ \"你好\" }"));
    assert_eq!(get("nested").source, "pub mod nested {");
    assert_eq!(get("nested::Store").kind, "struct");
    assert_eq!(get("nested::Store::score").start_line, 8);
    assert_eq!(get("nested::Store::score").kind, "method");
    assert_eq!(get("nested::Search").source, "pub trait Search {");
    assert_eq!(get("nested::Search::search").kind, "trait_method");
    assert_eq!(get("nested::<Store as Search>::search").kind, "method");
    assert_eq!(get("external").source, "mod external;");
    assert_eq!(get("example").kind, "macro");
    for c in chunks {
        assert!(source.contains(&c.source));
        let lines = source.split_inclusive('\n').collect::<Vec<_>>();
        assert!(
            lines[c.start_line as usize - 1..c.end_line as usize]
                .concat()
                .contains(&c.source)
        );
        assert_eq!(
            blake3::hash(c.source.as_bytes()).to_hex().as_str(),
            c.content_hash
        );
    }
}

#[test]
fn bom_and_shebang_offsets_still_point_to_original_source() {
    for prefix in [
        "\u{feff}",
        "#!/usr/bin/env rust-script\n",
        "\u{feff}#!/usr/bin/env rust-script\n",
    ] {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "main.rs",
            &format!("{prefix}pub fn hello() {{ println!(\"🦀\"); }}\n"),
        );
        let result = build(dir.path(), None);
        let chunk = &result.index.chunks()[0];
        assert_eq!(chunk.source, "pub fn hello() { println!(\"🦀\"); }");
        assert_eq!(chunk.start_line, if prefix.contains('\n') { 2 } else { 1 });
        assert_eq!(chunk.start_line, chunk.end_line);
    }
}

#[test]
fn ignore_rules_fallback_and_input_limits() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), ".gitignore", "ignored.rs\nignored_dir/\n");
    write(dir.path(), ".ignore", "extra.py\n");
    for path in [
        "ignored.rs",
        "ignored_dir/x.rs",
        "extra.py",
        "target/x.rs",
        "node_modules/x.js",
        ".hidden.rs",
    ] {
        write(dir.path(), path, "fn ignored() {}\n");
    }
    write(dir.path(), "nested/.gitignore", "local.rs\n!keep.rs\n");
    write(dir.path(), "nested/local.rs", "fn ignored() {}\n");
    write(dir.path(), "nested/keep.rs", "fn keep() {}\n");
    write(dir.path(), "fallback.py", "one\ntwo\nthree\nfour\nfive");
    write(dir.path(), "broken.rs", "fn unfinished(\nsecond\nthird");
    write(dir.path(), "comments.rs", "// comment only\n");
    write(dir.path(), "large.rs", &"x".repeat(101));
    fs::write(dir.path().join("binary.py"), [0xff, 0]).unwrap();
    write(dir.path(), "data.bin", "not source");
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.path().join("fallback.py"), dir.path().join("link.py")).unwrap();
    let options = BuildOptions {
        fallback_lines: 2,
        max_file_bytes: 100,
        ..Default::default()
    };
    let result =
        build_repository(dir.path(), &options, None, |text| hash_embedding(text, 128)).unwrap();
    assert_eq!(result.report.files, 4);
    assert_eq!(result.index.chunks().len(), 7);
    let python = result
        .index
        .chunks()
        .iter()
        .filter(|c| c.language == "python")
        .collect::<Vec<_>>();
    assert_eq!(python.len(), 3);
    assert_eq!((python[1].start_line, python[1].end_line), (3, 4));
    assert_eq!(python[1].source, "three\nfour\n");
    assert_eq!(python[2].source, "five");
    assert!(
        result
            .index
            .chunks()
            .iter()
            .filter(|c| c.path == "broken.rs")
            .all(|c| c.kind == "lines")
    );
    assert_eq!(result.report.warnings.len(), 4); // binary, large, broken Rust, comment-only Rust
}

#[test]
fn incremental_reuses_moved_chunks_and_detects_add_change_remove() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "src/lib.rs",
        "fn keep() { keep(); }\nfn change() { old(); }\nfn gone() {}\n",
    );
    write(dir.path(), "src/deleted.rs", "fn deleted() {}\n");
    let first = build(dir.path(), None);
    assert_eq!(first.report.added, 4);
    assert_eq!(first.report.embedded, 4);
    let mut bytes = Vec::new();
    first.index.write_ipc(&mut bytes).unwrap();
    let previous = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    let unchanged = build_repository(
        dir.path(),
        &BuildOptions::default(),
        Some(&previous),
        |_| panic!("unchanged repository must not invoke the embedding callback"),
    )
    .unwrap();
    assert_eq!(unchanged.report.embedded, 0);
    assert_eq!(unchanged.report.reused, 4);
    assert_eq!(unchanged.report.removed, 0);
    assert_eq!(unchanged.index.batch(), previous.batch());

    write(
        dir.path(),
        "src/lib.rs",
        "// new comment\n\nfn keep() { keep(); }\nfn change() { new(); }\n",
    );
    fs::remove_file(dir.path().join("src/deleted.rs")).unwrap();
    write(dir.path(), "src/new.rs", "fn added() {}\n");
    let mut calls = 0;
    let updated = build_repository(
        dir.path(),
        &BuildOptions::default(),
        Some(&previous),
        |text| {
            calls += 1;
            hash_embedding(text, 128)
        },
    )
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!((updated.report.reused, updated.report.embedded), (1, 2));
    assert_eq!(
        (
            updated.report.added,
            updated.report.changed,
            updated.report.removed
        ),
        (1, 1, 2)
    );
    let keep = updated
        .index
        .chunks()
        .iter()
        .find(|c| c.symbol == "keep")
        .unwrap();
    assert_eq!(keep.start_line, 3);
    assert!(
        !updated
            .index
            .chunks()
            .iter()
            .any(|c| c.symbol == "gone" || c.symbol == "deleted")
    );
}

#[test]
fn editing_one_method_reuses_container_and_other_methods_with_custom_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "lib.rs",
        "pub struct Index;\nimpl Index {\nfn keep() {}\nfn change() { old(); }\n}\n",
    );
    let options = BuildOptions {
        embedding_id: "precomputed-fixture-v1".into(),
        dimensions: 2,
        ..Default::default()
    };
    let first = build_repository(dir.path(), &options, None, |_| Ok(vec![1.0, 2.0])).unwrap();
    assert_eq!(first.report.chunks, 4);
    write(
        dir.path(),
        "lib.rs",
        "pub struct Index;\nimpl Index {\nfn keep() {}\nfn change() { new(); }\n}\n",
    );
    let second = build_repository(dir.path(), &options, Some(&first.index), |source| {
        assert_eq!(source, "fn change() { new(); }");
        Ok(vec![3.0, 4.0])
    })
    .unwrap();
    assert_eq!(
        (
            second.report.reused,
            second.report.embedded,
            second.report.changed
        ),
        (3, 1, 1)
    );
    assert_eq!((second.report.added, second.report.removed), (0, 0));
    assert_eq!(second.index.embedding_id(), "precomputed-fixture-v1");
    assert_eq!(second.index.dimensions(), 2);
}

#[test]
fn duplicate_sources_are_consumed_once_and_empty_updates_work() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "lib.rs", "fn same() {}\nfn same() {}\n");
    let first = build(dir.path(), None); // Parsing, not type checking: duplicate names are allowed.
    write(dir.path(), "lib.rs", "fn same() {}\n");
    let second = build(dir.path(), Some(&first.index));
    assert_eq!((second.report.reused, second.report.removed), (1, 1));
    write(dir.path(), "lib.rs", "");
    let third = build(dir.path(), Some(&second.index));
    assert_eq!((third.report.chunks, third.report.removed), (0, 1));
    let mut bytes = Vec::new();
    third.index.write_ipc(&mut bytes).unwrap();
    let loaded = CodeIndex::read_ipc(Cursor::new(bytes)).unwrap();
    assert!(loaded.chunks().is_empty());
    assert_eq!(loaded.dimensions(), 128);
    let index = VectorIndex::new(
        &cubecl::Device::default(),
        loaded.batch().clone(),
        "embedding",
    )
    .unwrap();
    assert!(
        index
            .query_with_metric(
                &[0.0; 128],
                cubeql::IndexQueryOptions::new(10),
                Metric::Cosine
            )
            .unwrap()
            .is_empty()
    );
}

#[test]
fn ipc_roundtrip_is_queryable_with_locations_and_matches_cpu_reference() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "src/lib.rs",
        "fn vector_score() { cosine_similarity(); }\nfn parse_query() { parse(); }\n",
    );
    write(
        dir.path(),
        "tests/check.rs",
        "fn check() { assert_equal(); }\n",
    );
    let built = build(dir.path(), None);
    let path = dir.path().join("index.arrow");
    built.index.save_ipc(&path).unwrap();
    assert!(built.index.save_ipc(&path).is_err());
    let loaded = CodeIndex::load_ipc(&path).unwrap();
    assert_eq!(loaded.batch(), built.index.batch());
    let index = VectorIndex::load_ipc(&cubecl::Device::default(), &path).unwrap();
    let query = hash_embedding("fn vector_score() { cosine_similarity(); }", 128).unwrap();
    let hit = index
        .query_filtered(
            &query,
            1,
            Metric::Cosine,
            &[
                Filter::StringEq {
                    column: "language",
                    value: "rust",
                },
                Filter::PathPrefix {
                    column: "path",
                    prefix: "src/",
                },
            ],
        )
        .unwrap()
        .remove(0);
    let chunk = &loaded.chunks()[hit.row];
    assert_eq!(
        (
            &*chunk.path,
            &*chunk.symbol,
            chunk.start_line,
            chunk.end_line
        ),
        ("src/lib.rs", "vector_score", 1, 1)
    );
    assert!((hit.score - 1.0).abs() < 1e-5);
    let vectors = loaded
        .batch()
        .column_by_name("embedding")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    for metric in [Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2] {
        let hits = index
            .query_with_metric(&query, cubeql::IndexQueryOptions::new(10), metric)
            .unwrap();
        let mut reference = (0..loaded.chunks().len())
            .map(|row| {
                let values = vectors.value(row);
                let values = values.as_any().downcast_ref::<Float32Array>().unwrap();
                let score = match metric {
                    Metric::InnerProduct | Metric::Cosine => values
                        .values()
                        .iter()
                        .zip(&query)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>(),
                    Metric::SquaredL2 => values
                        .values()
                        .iter()
                        .zip(&query)
                        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                        .sum::<f64>(),
                };
                (row, score)
            })
            .collect::<Vec<_>>();
        reference.sort_by(|a, b| {
            let order = a.1.total_cmp(&b.1);
            (if metric == Metric::SquaredL2 {
                order
            } else {
                order.reverse()
            })
            .then(a.0.cmp(&b.0))
        });
        for (hit, (row, score)) in hits.iter().zip(reference) {
            assert_eq!(hit.row, row);
            assert!((hit.score as f64 - score).abs() < 1e-5);
        }
    }
}

#[test]
fn invalid_embeddings_configuration_and_manifests_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "lib.rs", "fn example() {}\n");
    let options = BuildOptions::default();
    for vector in [
        vec![0.0; 127],
        vec![f32::NAN; 128],
        vec![f32::INFINITY; 128],
    ] {
        assert!(build_repository(dir.path(), &options, None, |_| Ok(vector.clone())).is_err());
    }
    assert!(matches!(
        build_repository(dir.path(), &options, None, |_| Err(CodeError::Embedding(
            "offline".into()
        ))),
        Err(CodeError::Embedding(_))
    ));
    let first = build(dir.path(), None);
    for bad in [
        BuildOptions {
            dimensions: 0,
            ..options.clone()
        },
        BuildOptions {
            fallback_lines: 0,
            ..options.clone()
        },
        BuildOptions {
            max_file_bytes: 0,
            ..options.clone()
        },
        BuildOptions {
            embedding_id: "".into(),
            ..options.clone()
        },
        BuildOptions {
            embedding_id: "different-model".into(),
            ..options.clone()
        },
        BuildOptions {
            dimensions: 64,
            ..options.clone()
        },
        BuildOptions {
            fallback_lines: 40,
            ..options.clone()
        },
    ] {
        assert!(
            build_repository(dir.path(), &bad, Some(&first.index), |_| panic!(
                "invalid config must fail before embedding"
            ))
            .is_err()
        );
    }
    for key in [
        "cubeql.code.version",
        "cubeql.code.chunker",
        "cubeql.code.dimensions",
        "cubeql.code.fallback_lines",
        "cubeql.code.embedding_id",
    ] {
        let mut batch = first.index.batch().clone();
        batch.schema_metadata_mut().insert(key, "");
        let mut bytes = Vec::new();
        write_ipc_batch(&batch, "embedding", IndexOptions::default(), &mut bytes).unwrap();
        assert!(CodeIndex::read_ipc(Cursor::new(bytes)).is_err());
    }
    let mut bytes = Vec::new();
    write_ipc_batch(
        first.index.batch(),
        "embedding",
        IndexOptions { normalize: true },
        &mut bytes,
    )
    .unwrap();
    assert!(CodeIndex::read_ipc(Cursor::new(bytes)).is_err());
    assert!(CodeIndex::read_ipc(Cursor::new(b"invalid IPC")).is_err());
}

#[test]
fn hash_fixture_is_deterministic_and_cli_reuses_embeddings() {
    assert_eq!(
        hash_embedding("HELLO world", 128).unwrap(),
        hash_embedding("hello WORLD", 128).unwrap()
    );
    assert_eq!(hash_embedding("", 4).unwrap(), vec![0.0; 4]);
    assert!(hash_embedding("text", 0).is_err());
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "lib.rs", "fn example() {}\n");
    let first = dir.path().join("first.arrow");
    let second = dir.path().join("second.arrow");
    let bin = env!("CARGO_BIN_EXE_cubeql-code");
    let run = Command::new(bin)
        .arg("index")
        .arg(dir.path())
        .arg("--output")
        .arg(&first)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("embedded=1 reused=0"));
    let run = Command::new(bin)
        .arg("index")
        .arg(dir.path())
        .arg("--output")
        .arg(&second)
        .arg("--previous")
        .arg(&first)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("embedded=0 reused=1"));
    let original = fs::read(&first).unwrap();
    let run = Command::new(bin)
        .arg("index")
        .arg(dir.path())
        .arg("--output")
        .arg(&first)
        .output()
        .unwrap();
    assert!(!run.status.success());
    assert_eq!(fs::read(&first).unwrap(), original);
    assert_eq!(
        CodeIndex::load_ipc(first).unwrap().batch(),
        CodeIndex::load_ipc(second).unwrap().batch()
    );
}
