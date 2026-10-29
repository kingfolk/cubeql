//! Exercise the real binary over MCP newline-delimited JSON-RPC, not private handler methods.
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    time::Duration,
};

use cubeql_code::{BuildOptions, CodeIndex, VectorPrecision, build_repository_with_provider};
use cubeql_embeddings::{EmbeddingConfig, EmbeddingError, EmbeddingProvider, HashProvider};
use serde_json::{Value, json};

const TARGET: &str =
    "/// Read application settings. 配置🦀\npub fn load_config() { needle_api(); }";
const OTHER: &str = "pub fn unrelated() { something_else(); }";

fn fixture(precision: VectorPrecision, unsupported: bool) -> (tempfile::TempDir, CodeIndex) {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("lib.rs"), format!("{TARGET}\n{OTHER}\n")).unwrap();
    // Same cheap vectors, but an intentionally unsupported identity proves host-only sessions
    // never attempt to instantiate a model merely because a snapshot has provider metadata.
    struct Provider {
        inner: HashProvider,
        config: EmbeddingConfig,
    }
    impl EmbeddingProvider for Provider {
        fn config(&self) -> &EmbeddingConfig {
            &self.config
        }
        fn embed_documents(&mut self, text: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.inner.embed_documents(text)
        }
        fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            self.inner.embed_query(text)
        }
    }
    let inner = HashProvider::new(32).unwrap();
    let mut config = inner.config().clone();
    if unsupported {
        config.model = "not-installed-test-model".into();
    }
    let mut provider = Provider { inner, config };
    let built = build_repository_with_provider(
        dir.path(),
        &BuildOptions::for_embedding(provider.config()).unwrap(),
        None,
        &mut provider,
        2,
    )
    .unwrap();
    let code = built.index.with_precision(precision).unwrap();
    code.save_ipc(dir.path().join("code.arrow")).unwrap();
    (dir, code)
}

struct Client {
    child: Child,
    input: Option<ChildStdin>,
    output: Receiver<Result<Value, String>>,
    stderr: Option<std::thread::JoinHandle<String>>,
    next_id: u64,
}
impl Client {
    fn start(index: &Path, text_only: bool) -> Self {
        Self::start_with_model(index, text_only, None)
    }
    fn start_with_model(index: &Path, text_only: bool, model_dir: Option<&Path>) -> Self {
        Self::start_with_workspace(index, text_only, model_dir, None)
    }
    fn start_with_workspace(
        index: &Path,
        text_only: bool,
        model_dir: Option<&Path>,
        workspace_root: Option<&Path>,
    ) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cubeql-mcp"));
        command.arg(index);
        if text_only {
            command.arg("--text-only");
        }
        if let Some(directory) = model_dir {
            command.arg("--model-dir").arg(directory);
        }
        if let Some(directory) = workspace_root {
            command.arg("--workspace-root").arg(directory);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let stderr = Some(std::thread::spawn(move || {
            let mut text = String::new();
            stderr.read_to_string(&mut text).unwrap();
            text
        }));
        let (send, output) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let message = line.map_err(|e| e.to_string()).and_then(|line| {
                    serde_json::from_str(&line)
                        .map_err(|e| format!("non-protocol stdout: {line}: {e}"))
                });
                if send.send(message).is_err() {
                    break;
                }
            }
        });
        let mut client = Self {
            child,
            input,
            output,
            stderr,
            next_id: 0,
        };
        let init = client.request(
            "initialize",
            json!({
                "protocolVersion":"2025-11-25", "capabilities":{},
                "clientInfo":{"name":"cubeql-protocol-test","version":"1"}
            }),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-11-25");
        assert!(init["result"]["capabilities"]["tools"].is_object());
        assert!(
            init["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("snapshot")
        );
        client.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
        client
    }
    fn send(&mut self, message: Value) {
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{message}").unwrap();
        input.flush().unwrap();
    }
    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        self.send(json!({"jsonrpc":"2.0","id":self.next_id,"method":method,"params":params}));
        loop {
            let message = match self.output.recv_timeout(Duration::from_secs(45)) {
                Ok(message) => message.unwrap(),
                Err(error) => {
                    // Capture startup/runtime diagnostics before Drop reaps the subprocess.
                    // In particular, do not hide a wrong-feature binary behind "Disconnected".
                    let _ = self.child.kill();
                    let status = self.child.wait();
                    let diagnostics = self.stderr.take().unwrap().join().unwrap();
                    panic!(
                        "MCP response failed: {error}; status={status:?}; stderr:\n{diagnostics}"
                    );
                }
            };
            if message.get("id") == Some(&json!(self.next_id)) {
                return message;
            }
            assert!(
                message.get("method").is_some(),
                "unexpected response: {message}"
            );
        }
    }
    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let response = self.request("tools/call", json!({"name":name,"arguments":arguments}));
        assert!(response.get("error").is_none(), "{response}");
        response["result"].clone()
    }
    fn success(&mut self, name: &str, arguments: Value) -> Value {
        let response = self.call(name, arguments);
        assert_eq!(response["isError"], false, "{response}");
        let structured = response["structuredContent"].clone();
        // Older clients can consume the same JSON in the text-content fallback.
        let text: Value =
            serde_json::from_str(response["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, structured);
        structured
    }
    fn error(&mut self, name: &str, arguments: Value, code: &str) {
        let response = self.call(name, arguments);
        assert_eq!(response["isError"], true, "{response}");
        assert_eq!(
            response["structuredContent"]["error"]["code"], code,
            "{response}"
        );
    }
    fn close(mut self) -> String {
        // EOF should shut down cleanly, without a separate nonstandard shutdown RPC.
        self.input.take();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "server did not exit on EOF"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        self.stderr.take().unwrap().join().unwrap()
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn host_tools_negotiate_search_and_page_immutable_source_without_a_provider() {
    let (dir, code) = fixture(VectorPrecision::Fp32, true);
    let index = dir.path().join("code.arrow");
    let expected_id = format!(
        "blake3:{}",
        blake3::hash(&fs::read(&index).unwrap()).to_hex()
    );
    let mut client = Client::start(&index, true);
    let tools = client.request("tools/list", json!({}));
    let tools = tools["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 3);
    for tool in tools {
        assert_eq!(tool["annotations"]["readOnlyHint"], true);
        assert_eq!(tool["annotations"]["openWorldHint"], false);
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
    }
    let found = client.success(
        "search_code",
        json!({"query":"needle API","strategy":"lexical","language":"rust"}),
    );
    assert_eq!(found["format_version"], 1);
    assert_eq!(found["snapshot_id"], expected_id);
    assert_eq!(found["backend"], "Host");
    assert_eq!(found["embedding_ms"], 0.0);
    assert_eq!(found["hits"].as_array().unwrap().len(), 1);
    let hit = &found["hits"][0];
    assert_eq!(hit["symbol"], "load_config");
    assert_eq!(hit["provenance"][0]["source"], "lexical");
    assert!(hit.get("source").is_none());
    let row = hit["chunk_id"].as_u64().unwrap() as usize;
    assert_eq!(code.chunks()[row].source, TARGET);
    // Neither changed live source nor removal of the startup IPC file changes this session.
    fs::write(dir.path().join("lib.rs"), "fn changed_live_checkout() {}\n").unwrap();
    fs::remove_file(&index).unwrap();
    let mut offset = 0usize;
    let mut reconstructed = String::new();
    loop {
        let page = client.success(
            "read_code",
            json!({"snapshot_id":expected_id,"chunk_id":row,"offset":offset,"max_bytes":4}),
        );
        assert_eq!(page["source"], "snapshot");
        assert_eq!(page["content_hash"], code.chunks()[row].content_hash);
        assert_eq!(page["start_line"], 1);
        assert_eq!(page["end_line"], 2);
        let text = page["text"].as_str().unwrap();
        assert!(text.len() <= 4);
        reconstructed.push_str(text);
        match page["next_offset"].as_u64() {
            Some(next) => {
                assert!(next as usize > offset);
                offset = next as usize;
            }
            None => break,
        }
    }
    assert_eq!(reconstructed, TARGET);
    let again = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    assert_eq!(again["hits"][0]["chunk_id"], row);
    assert_eq!(again["snapshot_id"], expected_id);
    let empty = client.success(
        "search_code",
        json!({"query":"needle","strategy":"lexical","path_prefix":"missing/"}),
    );
    assert!(empty["hits"].as_array().unwrap().is_empty());
    let stderr = client.close();
    let events = stderr
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect::<Vec<_>>();
    assert_eq!(events.iter().filter(|e| e["event"] == "ready").count(), 1);
    assert!(
        events
            .iter()
            .any(|e| e["method"] == "search_code" && e["returned_chunk_ids"] == json!([row]))
    );
    assert!(!stderr.contains("needle API")); // traces exclude raw queries and source.
}

#[test]
fn bad_requests_are_explicit_and_do_not_break_the_session() {
    let (dir, _) = fixture(VectorPrecision::Fp32, false);
    let mut client = Client::start(&dir.path().join("code.arrow"), true);
    let search = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    let id = search["snapshot_id"].clone();
    let row = search["hits"][0]["chunk_id"].clone();
    client.error(
        "search_code",
        json!({"query":"load_config"}),
        "unsupported_strategy",
    );
    for args in [
        json!({"query":" ","strategy":"lexical"}),
        json!({"query":"x".repeat(8193),"strategy":"lexical"}),
        json!({"query":"x","strategy":"lexical","top_k":101}),
        json!({"query":"x","strategy":"lexical","top_k":0}),
        json!({"query":"x","strategy":"lexical","top_k":5,"candidate_k":4}),
        json!({"query":"x","strategy":"lexical","candidate_k":1001}),
    ] {
        client.error("search_code", args, "invalid_request");
    }
    client.error(
        "read_code",
        json!({"snapshot_id":"another","chunk_id":row}),
        "snapshot_mismatch",
    );
    client.error(
        "read_code",
        json!({"snapshot_id":id,"chunk_id":99999}),
        "invalid_request",
    );
    client.error(
        "read_code",
        json!({"snapshot_id":id,"chunk_id":row,"max_bytes":65537}),
        "invalid_request",
    );
    client.error(
        "read_code",
        json!({"snapshot_id":id,"chunk_id":row,"offset":TARGET.len()+1}),
        "invalid_request",
    );
    let inside_unicode = TARGET.find('配').unwrap() + 1;
    client.error(
        "read_code",
        json!({"snapshot_id":id,"chunk_id":row,"offset":inside_unicode}),
        "invalid_request",
    );
    let malformed = client.request("tools/call", json!({"name":"search_code","arguments":{"query":"x","strategy":"lexical","file":"/etc/passwd"}}));
    // The SDK rejects unknown fields before invoking our handler and reports a tool error.
    assert_eq!(malformed["result"]["isError"], true, "{malformed}");
    assert!(
        malformed["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown field")
    );
    let unknown = client.request("tools/call", json!({"name":"write_code","arguments":{}}));
    assert!(unknown["error"].is_object());
    let read = client.success("read_code", json!({"snapshot_id":id,"chunk_id":row}));
    assert_eq!(read["text"], TARGET);
    client.close();
}

#[test]
fn full_server_reuses_resident_fp32_and_fp16_indexes_for_all_strategies() {
    for precision in [VectorPrecision::Fp32, VectorPrecision::Fp16] {
        let (dir, _) = fixture(precision, false);
        let index = dir.path().join("code.arrow");
        let mut client = Client::start(&index, false);
        fs::remove_file(&index).unwrap();
        let mut id = None;
        for strategy in ["semantic", "hybrid", "lexical", "symbol", "hybrid"] {
            let result = client.success("search_code", json!({"query":"load_config","strategy":strategy,"candidate_k":10,"language":"rust","path_prefix":"lib.rs"}));
            assert!(!result["hits"].as_array().unwrap().is_empty());
            assert_eq!(result["strategy"], strategy);
            if let Some(id) = &id {
                assert_eq!(id, &result["snapshot_id"]);
            }
            id = Some(result["snapshot_id"].clone());
            if strategy == "hybrid" {
                assert_eq!(result["hits"][0]["symbol"], "load_config");
                let evidence = result["hits"][0]["provenance"].as_array().unwrap();
                assert_eq!(evidence.len(), 3);
                assert_eq!(result["metric"], "rrf");
            }
            if matches!(strategy, "lexical" | "symbol") {
                assert_eq!(result["backend"], "Host");
            }
        }
        let stderr = client.close();
        assert_eq!(
            stderr
                .lines()
                .filter(|line| line.contains("\"event\":\"ready\""))
                .count(),
            1
        );
    }
}

#[test]
fn replacing_a_snapshot_changes_identity_and_rejects_old_chunk_handles() {
    let (dir, _) = fixture(VectorPrecision::Fp32, false);
    let path = dir.path().join("code.arrow");
    let mut first = Client::start(&path, true);
    let old = first.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    first.close();
    let (other, _) = fixture(VectorPrecision::Fp16, false);
    let mut second = Client::start(&other.path().join("code.arrow"), true);
    second.error(
        "read_code",
        json!({"snapshot_id":old["snapshot_id"],"chunk_id":old["hits"][0]["chunk_id"]}),
        "snapshot_mismatch",
    );
    second.close();
}

#[test]
fn startup_errors_do_not_emit_protocol_or_modify_inputs() {
    let (dir, _) = fixture(VectorPrecision::Fp32, true);
    let path = dir.path().join("code.arrow");
    let original = fs::read(&path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cubeql-mcp"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("unsupported embedding provider")
    );
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[test]
fn check_code_detects_whole_file_edits_and_missing_files_without_changing_snapshot_reads() {
    let (dir, code) = fixture(VectorPrecision::Fp32, false);
    let mut client =
        Client::start_with_workspace(&dir.path().join("code.arrow"), true, None, Some(dir.path()));
    let found = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    let handle =
        json!({"snapshot_id":found["snapshot_id"], "chunk_id":found["hits"][0]["chunk_id"]});
    let checked = client.success("check_code", handle.clone());
    assert_eq!(checked["status"], "unchanged");
    assert_eq!(checked["source"], "workspace_check");
    assert_eq!(
        checked["expected_file_hash"],
        code.file_hash("lib.rs").unwrap()
    );
    assert_eq!(checked["expected_file_hash"], checked["actual_file_hash"]);
    // This changes neither saved function body, but shifts every source location.
    fs::write(
        dir.path().join("lib.rs"),
        format!("// new header\n{TARGET}\n{OTHER}\n"),
    )
    .unwrap();
    let checked = client.success("check_code", handle.clone());
    assert_eq!(checked["status"], "modified");
    assert_eq!(checked["reason"], "file_hash_mismatch");
    // Restoring the original bytes is unchanged, irrespective of write time/inode history.
    fs::write(dir.path().join("lib.rs"), format!("{TARGET}\n{OTHER}\n")).unwrap();
    assert_eq!(
        client.success("check_code", handle.clone())["status"],
        "unchanged"
    );
    // Editing only the other function must still invalidate full-file evidence for this one.
    fs::write(
        dir.path().join("lib.rs"),
        format!("{TARGET}\nfn other_changed() {{}}\n"),
    )
    .unwrap();
    assert_eq!(
        client.success("check_code", handle.clone())["status"],
        "modified"
    );
    fs::remove_file(dir.path().join("lib.rs")).unwrap();
    assert_eq!(
        client.success("check_code", handle.clone())["status"],
        "missing"
    );
    assert_eq!(client.success("read_code", handle.clone())["text"], TARGET);
    client.error(
        "check_code",
        json!({"snapshot_id":"wrong","chunk_id":0}),
        "snapshot_mismatch",
    );
    client.error(
        "check_code",
        json!({"snapshot_id":found["snapshot_id"],"chunk_id":99999}),
        "invalid_request",
    );
    let malformed = client.request(
        "tools/call",
        json!({"name":"check_code","arguments":{
            "snapshot_id":found["snapshot_id"], "chunk_id":0, "path":"/etc/passwd"
        }}),
    );
    assert_eq!(malformed["result"]["isError"], true);
    let stderr = client.close();
    let events = stderr
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect::<Vec<_>>();
    assert!(
        events
            .iter()
            .any(|e| e["method"] == "check_code" && e["freshness"] == "modified")
    );
}

#[test]
fn workspace_authority_is_explicit_and_old_snapshots_never_claim_to_be_fresh() {
    let (dir, code) = fixture(VectorPrecision::Fp32, false);
    let index = dir.path().join("code.arrow");
    let mut client = Client::start(&index, true);
    let found = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    let checked = client.success(
        "check_code",
        json!({"snapshot_id":found["snapshot_id"],"chunk_id":found["hits"][0]["chunk_id"]}),
    );
    assert_eq!(checked["status"], "unverifiable");
    assert_eq!(checked["reason"], "workspace_not_configured");
    client.close();

    // Relocated workspaces are allowed only through an explicit operator-provided directory.
    let relocated = tempfile::tempdir().unwrap();
    fs::write(
        relocated.path().join("lib.rs"),
        format!("{TARGET}\n{OTHER}\n"),
    )
    .unwrap();
    let mut client = Client::start_with_workspace(&index, true, None, Some(relocated.path()));
    let found = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    assert_eq!(
        client.success(
            "check_code",
            json!({"snapshot_id":found["snapshot_id"],"chunk_id":found["hits"][0]["chunk_id"]})
        )["status"],
        "unchanged"
    );
    client.close();

    let mut batch = code.batch().clone();
    batch
        .schema_metadata_mut()
        .remove("cubeql.code.file_hashes");
    let mut bytes = Vec::new();
    cubeql::write_ipc_batch(
        &batch,
        "embedding",
        cubeql::IndexOptions::default(),
        &mut bytes,
    )
    .unwrap();
    let legacy_path = dir.path().join("legacy.arrow");
    fs::write(&legacy_path, bytes).unwrap();
    let mut client = Client::start_with_workspace(&legacy_path, true, None, Some(dir.path()));
    let found = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    let checked = client.success(
        "check_code",
        json!({"snapshot_id":found["snapshot_id"],"chunk_id":found["hits"][0]["chunk_id"]}),
    );
    assert_eq!(checked["status"], "unverifiable");
    assert_eq!(checked["reason"], "legacy_snapshot");
    assert!(checked["expected_file_hash"].is_null());
    client.close();
}

#[cfg(unix)]
#[test]
fn workspace_checks_reject_external_symlinks_and_never_block_on_a_substituted_fifo() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let (dir, _) = fixture(VectorPrecision::Fp32, false);
    let mut client =
        Client::start_with_workspace(&dir.path().join("code.arrow"), true, None, Some(dir.path()));
    let found = client.success(
        "search_code",
        json!({"query":"load_config","strategy":"symbol"}),
    );
    let handle =
        json!({"snapshot_id":found["snapshot_id"],"chunk_id":found["hits"][0]["chunk_id"]});
    let outside = tempfile::tempdir().unwrap();
    fs::write(
        outside.path().join("secret.rs"),
        format!("{TARGET}\n{OTHER}\n"),
    )
    .unwrap();
    let path = dir.path().join("lib.rs");
    fs::remove_file(&path).unwrap();
    symlink(outside.path().join("secret.rs"), &path).unwrap();
    let checked = client.success("check_code", handle.clone());
    assert_eq!(checked["status"], "unverifiable");
    assert_eq!(checked["reason"], "inaccessible_or_outside_workspace");
    assert_eq!(checked["checked_bytes"], 0);
    assert!(checked["actual_file_hash"].is_null());
    fs::remove_file(&path).unwrap();
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: a valid NUL-terminated path into this test's temporary directory, no aliases or
    // pointers retained by libc. This creates only a fixture, never production source paths.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let checked = client.success("check_code", handle);
    assert_eq!(checked["status"], "unverifiable");
    assert_eq!(checked["reason"], "not_regular_file");
    client.close();
}

#[cfg(feature = "local-embeddings")]
#[test]
#[ignore = "requires an existing pinned MiniLM model directory in CUBEQL_MINILM_DIR; never downloads"]
fn local_minilm_search_and_read_work_over_stdio() {
    let directory = std::env::var_os("CUBEQL_MINILM_DIR").expect("set CUBEQL_MINILM_DIR");
    let directory = Path::new(&directory);
    let mut provider = cubeql_embeddings::MiniLm::from_dir(directory).unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cubeql-code/tests/fixtures/retrieval-benchmark/corpus");
    let built = build_repository_with_provider(
        &source,
        &BuildOptions::for_embedding(provider.config()).unwrap(),
        None,
        &mut provider,
        8,
    )
    .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("minilm.arrow");
    built.index.save_ipc(&path).unwrap();
    let mut client = Client::start_with_model(&path, false, Some(directory));
    for strategy in ["semantic", "hybrid", "semantic"] {
        let found = client.success(
            "search_code",
            json!({
                "query":"Where do we read application settings from a file?",
                "strategy":strategy, "top_k":5, "candidate_k":24
            }),
        );
        let target = found["hits"]
            .as_array()
            .unwrap()
            .iter()
            .find(|hit| hit["symbol"] == "load_config")
            .expect("configuration reader is in top five");
        let read = client.success(
            "read_code",
            json!({"snapshot_id":found["snapshot_id"],"chunk_id":target["chunk_id"]}),
        );
        assert!(read["text"].as_str().unwrap().contains("fn load_config"));
    }
    let stderr = client.close();
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.contains("\"event\":\"ready\""))
            .count(),
        1
    );
}
