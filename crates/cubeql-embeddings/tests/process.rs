#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, time::Instant};

use cubeql_embeddings::{
    EmbeddingConfig, EmbeddingProvider, EncoderProcessConfig, Normalization, ProcessProvider,
};
use serde_json::json;
use tempfile::TempDir;

fn model_config(dimensions: usize) -> EmbeddingConfig {
    EmbeddingConfig {
        model: "fixture".into(),
        revision: "r1".into(),
        dimensions,
        normalization: Normalization::None,
        implementation: "fixture-v1".into(),
        pooling: "mean".into(),
        max_tokens: None,
        document_prefix: "doc: ".into(),
        query_prefix: "query: ".into(),
        artifacts: Default::default(),
    }
}

fn fixture(dir: &TempDir, script: &str, startup_ms: u64, request_ms: u64) -> EncoderProcessConfig {
    let path = dir.path().join("child.sh");
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    let cfg_path = dir.path().join("encoder.json");
    fs::write(
        &cfg_path,
        serde_json::to_vec(&json!({
            "format_version": 1, "command": path, "args": [],
            "startup_timeout_ms": startup_ms, "request_timeout_ms": request_ms,
        }))
        .unwrap(),
    )
    .unwrap();
    EncoderProcessConfig::load(cfg_path).unwrap()
}

fn ready(c: &EmbeddingConfig) -> String {
    format!(
        "{{\"protocol_version\":1,\"event\":\"ready\",\"config\":{},\"model_load_ms\":0}}",
        serde_json::to_string(c).unwrap()
    )
}

fn result(id: u64, vectors: &str, ms: &str) -> String {
    format!("{{\"protocol_version\":1,\"id\":{id},\"vectors\":{vectors},\"encoding_ms\":{ms}}}")
}

#[test]
fn loads_once_and_handles_multiple_roles_requests() {
    let dir = TempDir::new().unwrap();
    let c = model_config(2);
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n' '{}'\ni=0\nwhile IFS= read -r line; do i=$((i+1)); case $i in 1) echo '{}';; 2) echo '{}';; 3) echo '{}';; esac; done\n",
        ready(&c),
        result(1, "[[1,0],[0,1]]", "1"),
        result(2, "[[1,0]]", "0"),
        result(3, "[[0,1]]", "2")
    );
    let cfg = fixture(&dir, &script, 5000, 5000);
    let mut p = ProcessProvider::spawn(&cfg, Some(&c)).unwrap();
    assert_eq!(p.execution_kind(), "resident-process-encoding");
    let docs = p.embed_documents(&["a", "b"]).unwrap();
    assert_eq!(docs, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(p.embed_query("q").unwrap(), vec![1.0, 0.0]);
    assert_eq!(p.embed_documents(&["c"]).unwrap(), vec![vec![0.0, 1.0]]);
}

#[test]
fn splits_document_batches_preserving_rows_and_protocol_roles() {
    let dir = TempDir::new().unwrap();
    let c = model_config(2);
    let log = dir.path().join("requests.jsonl");
    let first = serde_json::to_string(&vec![vec![1.0, 0.0]; 32]).unwrap();
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n' '{}'\ni=0\nwhile IFS= read -r line; do printf '%s\\n' \"$line\" >> '{}'; i=$((i+1)); case $i in 1) echo '{}';; 2) echo '{}';; 3) echo '{}';; esac; done\n",
        ready(&c),
        log.display(),
        result(1, &first, "0"),
        result(2, "[[0,1]]", "0"),
        result(3, "[[1,0]]", "0")
    );
    let cfg = fixture(&dir, &script, 5000, 5000);
    let mut p = ProcessProvider::spawn(&cfg, Some(&c)).unwrap();
    let documents: Vec<_> = (0..33).map(|i| format!("document-{i}")).collect();
    let refs: Vec<_> = documents.iter().map(String::as_str).collect();
    let vectors = p.embed_documents(&refs).unwrap();
    assert_eq!(vectors.len(), 33);
    assert!(vectors[..32].iter().all(|v| *v == vec![1.0, 0.0]));
    assert_eq!(vectors[32], vec![0.0, 1.0]);
    assert!(p.embed_query("   ").is_err()); // Local validation must not consume an ID.
    assert_eq!(p.embed_query("new query").unwrap(), vec![1.0, 0.0]);
    drop(p);
    let requests: Vec<serde_json::Value> = fs::read_to_string(log)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["texts"], json!(documents[..32]));
    assert_eq!(requests[1]["texts"], json!(["document-32"]));
    assert_eq!(requests[0]["role"], "document");
    assert_eq!(requests[1]["id"], 2);
    assert_eq!(requests[2]["role"], "query");
    assert_eq!(requests[2]["id"], 3);
}

#[test]
fn validates_config_and_expected_contract() {
    let dir = TempDir::new().unwrap();
    let c = model_config(2);
    let cfg = fixture(
        &dir,
        &format!("#!/bin/sh\nprintf '%s\\n' '{}'\n", ready(&c)),
        5000,
        5000,
    );
    let wrong = model_config(3);
    assert!(ProcessProvider::spawn(&cfg, Some(&wrong)).is_err());
    assert!(ProcessProvider::spawn(&cfg, None).is_ok());

    let malformed = dir.path().join("bad.json");
    fs::write(
        &malformed,
        r#"{"format_version":1,"command":"relative","surprise":true}"#,
    )
    .unwrap();
    assert!(EncoderProcessConfig::load(malformed).is_err());
    let mut cfg_json = serde_json::to_value(&cfg).unwrap();
    cfg_json["startup_timeout_ms"] = json!(0);
    let invalid: EncoderProcessConfig = serde_json::from_value(cfg_json).unwrap();
    assert!(ProcessProvider::spawn(&invalid, None).is_err());
}

#[test]
fn rejects_wrong_id_dimensions_and_nan_and_poisons() {
    for (response, expected) in [
        (result(99, "[[1,0]]", "0"), "id"),
        (result(1, "[[1]]", "0"), "dimension"),
        (result(1, "[[NaN,0]]", "0"), "nan"),
    ] {
        let dir = TempDir::new().unwrap();
        let c = model_config(2);
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' '{}'\nIFS= read -r line\necho '{}'\n",
            ready(&c),
            response
        );
        let cfg = fixture(&dir, &script, 5000, 5000);
        let mut p = ProcessProvider::spawn(&cfg, Some(&c)).unwrap();
        let error = p.embed_query("q").unwrap_err().to_string().to_lowercase();
        if expected == "nan" {
            assert!(error.contains("json") || error.contains("non-finite"));
        } else if expected == "dimension" {
            assert!(error.contains("expected 2"));
        } else {
            assert!(error.contains("identity"));
        }
        assert!(p.embed_query("again").is_err());
    }
}

#[test]
fn startup_and_request_timeouts_kill_child_without_hanging() {
    let dir = TempDir::new().unwrap();
    let cfg = fixture(&dir, "#!/bin/sh\nsleep 30\n", 40, 40);
    let start = Instant::now();
    assert!(ProcessProvider::spawn(&cfg, None).is_err());
    assert!(start.elapsed().as_secs() < 2);

    let c = model_config(2);
    let cfg = fixture(
        &dir,
        &format!(
            "#!/bin/sh\nprintf '%s\\n' '{}'\nIFS= read -r line\nsleep 30\n",
            ready(&c)
        ),
        5000,
        40,
    );
    let mut p = ProcessProvider::spawn(&cfg, None).unwrap();
    let start = Instant::now();
    assert!(p.embed_query("q").is_err());
    assert!(start.elapsed().as_secs() < 2);
}

#[test]
fn child_exit_is_reported_and_cleaned_up() {
    let dir = TempDir::new().unwrap();
    let cfg = fixture(&dir, "#!/bin/sh\nexit 7\n", 5000, 5000);
    assert!(ProcessProvider::spawn(&cfg, None).is_err());

    let c = model_config(2);
    let cfg = fixture(
        &dir,
        &format!(
            "#!/bin/sh\nprintf '%s\\n' '{}'\nIFS= read -r line\nexit 7\n",
            ready(&c)
        ),
        5000,
        5000,
    );
    let mut p = ProcessProvider::spawn(&cfg, None).unwrap();
    assert!(p.embed_query("q").is_err());
}

#[test]
fn oversized_input_is_rejected_before_send_and_provider_can_continue() {
    let dir = TempDir::new().unwrap();
    let c = model_config(2);
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n' '{}'\nIFS= read -r line\necho '{}'\n",
        ready(&c),
        result(1, "[[1,0]]", "0")
    );
    let cfg = fixture(&dir, &script, 5000, 5000);
    let mut p = ProcessProvider::spawn(&cfg, None).unwrap();
    let enormous = "x".repeat(1024 * 1024);
    let error = p.embed_query(&enormous).unwrap_err().to_string();
    assert!(error.contains("exceeds 1 MiB"));
    assert_eq!(p.embed_query("small").unwrap(), vec![1.0, 0.0]);
}

#[test]
fn blocked_stdin_times_out_and_error_includes_bounded_stderr_tail() {
    let dir = TempDir::new().unwrap();
    let c = model_config(2);
    let pid_path = dir.path().join("child.pid");
    let script = format!(
        "#!/bin/sh\necho $$ > '{}'\nprintf '%s\\n' '{}'\necho STDERR_MARKER >&2\nsleep 30\n",
        pid_path.display(),
        ready(&c)
    );
    let cfg = fixture(&dir, &script, 5000, 60);
    let mut p = ProcessProvider::spawn(&cfg, None).unwrap();
    let payload = "x".repeat(512 * 1024);
    let start = Instant::now();
    let error = p.embed_query(&payload).unwrap_err().to_string();
    assert!(start.elapsed().as_secs() < 2);
    assert!(error.contains("timed out"));
    assert!(error.contains("STDERR_MARKER"));
    let child_pid: i32 = fs::read_to_string(pid_path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(unsafe { libc::kill(child_pid, 0) }, -1);
}

#[test]
fn dropping_provider_kills_and_reaps_owned_child() {
    let dir = TempDir::new().unwrap();
    let c = model_config(2);
    let pid_path = dir.path().join("child.pid");
    let script = format!(
        "#!/bin/sh\necho $$ > '{}'\nprintf '%s\\n' '{}'\nwhile :; do sleep 1; done\n",
        pid_path.display(),
        ready(&c)
    );
    let cfg = fixture(&dir, &script, 5000, 5000);
    let p = ProcessProvider::spawn(&cfg, None).unwrap();
    let child_pid: i32 = fs::read_to_string(pid_path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    drop(p);
    // Drop joins the actor, which waits for the direct child before returning.
    assert_eq!(unsafe { libc::kill(child_pid, 0) }, -1);
}
