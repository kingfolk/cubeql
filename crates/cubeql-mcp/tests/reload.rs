//! Exercise the opt-in snapshot reload protocol through the real MCP stdio binary.
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    time::Duration,
};

use cubeql_code::{BuildOptions, CodeIndex, build_repository_with_provider};
use cubeql_embeddings::{EmbeddingConfig, EmbeddingError, EmbeddingProvider, HashProvider};
use serde_json::{Value, json};

const ORIGINAL: &str = "pub fn original_symbol() { original_marker(); }\n";
const UPDATED: &str = "pub fn replacement_symbol() { replacement_marker(); }\n";

struct FixtureProvider {
    inner: HashProvider,
    config: EmbeddingConfig,
}

impl EmbeddingProvider for FixtureProvider {
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

fn snapshot(root: &Path, source: &str, unsupported: bool) -> CodeIndex {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("lib.rs"), source).unwrap();
    let inner = HashProvider::new(32).unwrap();
    let mut config = inner.config().clone();
    if unsupported {
        config.model = "reload-test-unsupported-model".into();
    }
    let mut provider = FixtureProvider { inner, config };
    let result = build_repository_with_provider(
        root,
        &BuildOptions::for_embedding(provider.config()).unwrap(),
        None,
        &mut provider,
        2,
    )
    .unwrap();
    result.index
}

fn write_snapshot(root: &Path, source: &str, unsupported: bool) -> std::path::PathBuf {
    let index = snapshot(root, source, unsupported);
    let path = root.join("snapshot.arrow");
    index.save_ipc(&path).unwrap();
    path
}

fn snapshot_id(path: &Path) -> String {
    format!("blake3:{}", blake3::hash(&fs::read(path).unwrap()).to_hex())
}

struct Client {
    child: Child,
    input: Option<ChildStdin>,
    output: Receiver<Result<Value, String>>,
    stderr: Option<std::thread::JoinHandle<String>>,
    next_id: u64,
}

impl Client {
    fn start(index: &Path, reload_index: &Path, text_only: bool) -> Self {
        Self::start_with_policy(index, reload_index, text_only, None, None)
    }

    fn start_with_policy(
        index: &Path,
        reload_index: &Path,
        text_only: bool,
        strategy: Option<&str>,
        workspace_root: Option<&Path>,
    ) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cubeql-mcp"));
        command.arg(index).arg("--reload-index").arg(reload_index);
        if text_only {
            command.arg("--text-only");
        }
        if let Some(strategy) = strategy {
            command.arg("--search-strategy").arg(strategy);
        }
        if let Some(workspace_root) = workspace_root {
            command.arg("--workspace-root").arg(workspace_root);
        }
        Self::from_command(command)
    }

    fn start_default(index: &Path) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cubeql-mcp"));
        command.arg(index);
        Self::from_command(command)
    }

    fn from_command(mut command: Command) -> Self {
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
                let message = line.map_err(|error| error.to_string()).and_then(|line| {
                    serde_json::from_str(&line)
                        .map_err(|error| format!("non-protocol stdout: {line}: {error}"))
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
                "clientInfo":{"name":"cubeql-reload-test","version":"1"}
            }),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-11-25");
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
        let id = self.next_id;
        self.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        loop {
            let message = match self.output.recv_timeout(Duration::from_secs(20)) {
                Ok(message) => message.unwrap(),
                Err(error) => {
                    let _ = self.child.kill();
                    let status = self.child.wait();
                    let diagnostics = self.stderr.take().unwrap().join().unwrap();
                    panic!(
                        "MCP response failed: {error}; status={status:?}; stderr:\n{diagnostics}"
                    );
                }
            };
            if message.get("id") == Some(&json!(id)) {
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
        let result = self.call(name, arguments);
        assert_eq!(result["isError"], false, "{result}");
        let structured = result["structuredContent"].clone();
        let text: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, structured);
        structured
    }

    fn error(&mut self, name: &str, arguments: Value, code: &str) {
        let result = self.call(name, arguments);
        assert_eq!(result["isError"], true, "{result}");
        assert_eq!(
            result["structuredContent"]["error"]["code"], code,
            "{result}"
        );
    }

    fn tools(&mut self) -> Vec<Value> {
        self.request("tools/list", json!({}))["result"]["tools"]
            .as_array()
            .unwrap()
            .clone()
    }

    fn close(mut self) {
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
        self.stderr.take().unwrap().join().unwrap();
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn search(client: &mut Client, symbol: &str) -> Value {
    client.success("search_code", json!({"query":symbol,"strategy":"symbol"}))
}

#[test]
fn reload_is_opt_in_and_has_an_exact_non_open_input_contract() {
    let dir = tempfile::tempdir().unwrap();
    let startup = write_snapshot(&dir.path().join("startup"), ORIGINAL, false);
    let mut child = Client::start_default(&startup);
    let tools = child.tools();
    assert_eq!(tools.len(), 3);
    assert!(tools.iter().all(|tool| tool["name"] != "reload_snapshot"));
    let unknown = child.request(
        "tools/call",
        json!({"name":"reload_snapshot","arguments":{"expected_snapshot_id":format!("blake3:{}", "0".repeat(64))}}),
    );
    assert!(unknown["error"].is_object());
    child.close();

    let target = dir.path().join("future.arrow");
    let mut client = Client::start(&startup, &target, true);
    let tools = client.tools();
    assert_eq!(tools.len(), 4);
    let reload = tools
        .iter()
        .find(|tool| tool["name"] == "reload_snapshot")
        .unwrap();
    assert_eq!(reload["annotations"]["readOnlyHint"], false);
    assert_eq!(reload["annotations"]["destructiveHint"], false);
    assert_eq!(reload["annotations"]["idempotentHint"], false);
    assert_eq!(reload["inputSchema"]["additionalProperties"], false);
    assert_eq!(
        reload["inputSchema"]["required"],
        json!(["expected_snapshot_id"])
    );
    assert_eq!(
        reload["inputSchema"]["properties"]["expected_snapshot_id"]["type"],
        "string"
    );
    assert_eq!(
        reload["inputSchema"]["properties"]
            .as_object()
            .unwrap()
            .len(),
        1
    );
    let current = search(&mut client, "original_symbol");
    let id = current["snapshot_id"].as_str().unwrap().to_owned();
    let malformed = client.request(
        "tools/call",
        json!({"name":"reload_snapshot","arguments":{"expected_snapshot_id":id,"path":"/etc/passwd"}}),
    );
    assert_eq!(malformed["result"]["isError"], true, "{malformed}");
    assert!(
        malformed["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown field")
    );
    client.close();
}

#[test]
fn reload_is_explicit_compare_and_swap_and_keeps_old_state_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let startup_path = write_snapshot(&dir.path().join("startup"), ORIGINAL, false);
    let startup_id = snapshot_id(&startup_path);
    let target = dir.path().join("reload-target.arrow"); // Deliberately absent at startup.
    let mut client = Client::start(&startup_path, &target, false);
    let initial = search(&mut client, "original_symbol");
    assert_eq!(initial["snapshot_id"], startup_id);
    let old_handle = json!({"snapshot_id":startup_id,"chunk_id":initial["hits"][0]["chunk_id"]});

    client.error(
        "reload_snapshot",
        json!({"expected_snapshot_id":startup_id}),
        "reload_failed",
    ); // Missing target is harmless and does not change the active snapshot.
    assert_eq!(
        search(&mut client, "original_symbol")["snapshot_id"],
        startup_id
    );

    let updated_path = write_snapshot(&dir.path().join("updated"), UPDATED, false);
    fs::copy(&updated_path, &target).unwrap();
    let updated_id = snapshot_id(&target);
    // A valid but stale CAS id is rejected before the target snapshot can be loaded.
    fs::write(
        &target,
        b"corrupt but must not be read for a stale compare-and-swap",
    )
    .unwrap();
    client.error(
        "reload_snapshot",
        json!({"expected_snapshot_id":format!("blake3:{}", "0".repeat(64))}),
        "snapshot_mismatch",
    );
    assert_eq!(
        search(&mut client, "original_symbol")["snapshot_id"],
        startup_id
    );
    fs::copy(&updated_path, &target).unwrap();
    assert_eq!(snapshot_id(&target), updated_id);
    assert_eq!(
        search(&mut client, "original_symbol")["snapshot_id"],
        startup_id
    );
    client.error(
        "reload_snapshot",
        json!({"expected_snapshot_id":"invalid"}),
        "invalid_request",
    );

    let switched = client.success(
        "reload_snapshot",
        json!({"expected_snapshot_id":startup_id}),
    );
    assert_eq!(switched["format_version"], 1);
    assert_eq!(switched["source"], "snapshot_reload");
    assert_eq!(switched["previous_snapshot_id"], startup_id);
    assert_eq!(switched["snapshot_id"], updated_id);
    assert_eq!(switched["changed"], true);
    assert_eq!(switched["provider_reused"], true);
    for key in [
        "chunks",
        "preflight_ms",
        "prepare_ms",
        "text_index_ms",
        "reload_ms",
    ] {
        assert!(
            switched[key].is_number(),
            "missing numeric {key}: {switched}"
        );
    }
    client.error("read_code", old_handle.clone(), "snapshot_mismatch");
    client.error("check_code", old_handle, "snapshot_mismatch");
    let replacement = search(&mut client, "replacement_symbol");
    assert_eq!(replacement["snapshot_id"], updated_id);
    let new_handle = json!({
        "snapshot_id":updated_id,
        "chunk_id":replacement["hits"][0]["chunk_id"]
    });
    let read = client.success("read_code", new_handle.clone());
    assert_eq!(read["text"], UPDATED.trim_end());

    let noop = client.success(
        "reload_snapshot",
        json!({"expected_snapshot_id":updated_id}),
    );
    assert_eq!(noop["changed"], false);
    assert_eq!(noop["snapshot_id"], updated_id);
    assert_eq!(
        client.success("read_code", new_handle.clone())["text"],
        UPDATED.trim_end()
    );

    fs::write(&target, b"bad IPC snapshot").unwrap();
    client.error(
        "reload_snapshot",
        json!({"expected_snapshot_id":updated_id}),
        "reload_failed",
    );
    assert_eq!(
        search(&mut client, "replacement_symbol")["snapshot_id"],
        updated_id
    );
    fs::File::create(&target)
        .unwrap()
        .set_len(cubeql_mcp::snapshot::MAX_RELOAD_BYTES + 1)
        .unwrap();
    client.error(
        "reload_snapshot",
        json!({"expected_snapshot_id":updated_id}),
        "reload_failed",
    );
    assert_eq!(
        client.success("read_code", new_handle)["text"],
        UPDATED.trim_end()
    );
    // Repeated generations release borrowed adapters and bind text/device metadata together.
    let mut active = updated_id.clone();
    for (path, symbol, expected) in [
        (&startup_path, "original_symbol", &startup_id),
        (&updated_path, "replacement_symbol", &updated_id),
        (&startup_path, "original_symbol", &startup_id),
    ] {
        fs::copy(path, &target).unwrap();
        let value = client.success("reload_snapshot", json!({"expected_snapshot_id":active}));
        assert_eq!(value["snapshot_id"], *expected);
        assert_eq!(search(&mut client, symbol)["snapshot_id"], *expected);
        active = expected.clone();
    }
    client.close();
}

#[test]
fn reload_checks_embedding_compatibility_unless_text_only_is_selected() {
    let dir = tempfile::tempdir().unwrap();
    let startup_path = write_snapshot(&dir.path().join("startup"), ORIGINAL, false);
    let startup_id = snapshot_id(&startup_path);
    let incompatible = write_snapshot(&dir.path().join("incompatible"), UPDATED, true);
    let target = dir.path().join("target.arrow");
    fs::copy(&incompatible, &target).unwrap();
    let incompatible_id = snapshot_id(&target);

    let mut indexed = Client::start(&startup_path, &target, false);
    indexed.error(
        "reload_snapshot",
        json!({"expected_snapshot_id":startup_id}),
        "embedding_mismatch",
    );
    assert_eq!(
        search(&mut indexed, "original_symbol")["snapshot_id"],
        startup_id
    );
    indexed.close();

    let mut text_only =
        Client::start_with_policy(&startup_path, &target, true, Some("lexical"), None);
    let result = text_only.success(
        "reload_snapshot",
        json!({"expected_snapshot_id":startup_id}),
    );
    assert_eq!(result["snapshot_id"], incompatible_id);
    assert_eq!(result["changed"], true);
    let found = text_only.success(
        "search_code",
        json!({"query":"replacement_symbol","strategy":"lexical"}),
    );
    assert_eq!(found["snapshot_id"], incompatible_id);
    assert_eq!(found["backend"], "Host");
    text_only.error(
        "search_code",
        json!({"query":"replacement_symbol","strategy":"symbol"}),
        "strategy_restricted",
    );
    let checked = text_only.success(
        "check_code",
        json!({
            "snapshot_id":incompatible_id,
            "chunk_id":found["hits"][0]["chunk_id"]
        }),
    );
    assert_eq!(checked["status"], "unverifiable");
    assert_eq!(checked["reason"], "workspace_not_configured");
    text_only.close();
}
