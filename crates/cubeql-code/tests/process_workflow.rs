#![cfg(unix)]
use cubeql_embeddings::{EmbeddingProvider, HashProvider};
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cubeql-code"))
        .args(args)
        .output()
        .unwrap()
}
fn ok(args: &[&str]) -> Value {
    let out = cli(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
fn profile(dir: &Path) -> std::path::PathBuf {
    let mut config = HashProvider::new(2).unwrap().config().clone();
    config.model = "resident-fixture".into();
    fs::write(
        dir.join("embedding.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::write(dir.join("worker.py"),r#"import json,sys
from pathlib import Path
root=Path(sys.argv[1]); config=json.loads((root/'embedding.json').read_text())
with (root/'loads').open('a') as f:f.write('loaded\n')
print(json.dumps({'protocol_version':1,'event':'ready','config':config,'model_load_ms':0.1}),flush=True)
for line in sys.stdin:
 q=json.loads(line)
 with (root/'requests').open('a') as f:f.write(q['role']+'\n')
 vectors=[[1.,0.] if ('read' in t or 'load' in t) else [0.,1.] for t in q['texts']]
 print(json.dumps({'protocol_version':1,'id':q['id'],'vectors':vectors,'encoding_ms':0.1}),flush=True)
"#).unwrap();
    let p = dir.join("process.json");
    fs::write(&p,serde_json::to_vec(&json!({"format_version":1,"command":"/usr/bin/python3","args":[dir.join("worker.py"),dir],"startup_timeout_ms":5000,"request_timeout_ms":1000})).unwrap()).unwrap();
    p
}
#[test]
fn process_index_search_update_and_conflicts() {
    let d = tempfile::tempdir().unwrap();
    let repo = d.path().join("repo");
    fs::create_dir(&repo).unwrap();
    fs::write(
        repo.join("lib.rs"),
        "pub fn read_settings() -> usize { 1 }\npub fn save_settings() -> usize { 2 }\n",
    )
    .unwrap();
    let cfg = profile(d.path());
    let index = d.path().join("index.arrow");
    let built = ok(&[
        "index",
        repo.to_str().unwrap(),
        "--embedding",
        "process",
        "--encoder-config",
        cfg.to_str().unwrap(),
        "--output",
        index.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert_eq!(built["report"]["embedded"], 2);
    let found = ok(&[
        "search",
        index.to_str().unwrap(),
        "read settings",
        "--encoder-config",
        cfg.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert_eq!(found["hits"][0]["symbol"], "read_settings");
    let found = ok(&[
        "search",
        index.to_str().unwrap(),
        "save settings",
        "--encoder-config",
        cfg.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert_eq!(found["hits"][0]["symbol"], "save_settings");
    let next = d.path().join("next.arrow");
    let update = ok(&[
        "update",
        index.to_str().unwrap(),
        "--encoder-config",
        cfg.to_str().unwrap(),
        "--output",
        next.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert_eq!(update["report"]["embedded"], 0);
    assert_eq!(update["report"]["reused"], 2);
    let conflict = cli(&[
        "search",
        index.to_str().unwrap(),
        "read",
        "--encoder-config",
        cfg.to_str().unwrap(),
        "--model-dir",
        d.path().to_str().unwrap(),
    ]);
    assert!(!conflict.status.success());
    let wrong = d.path().join("wrong.arrow");
    let out = cli(&[
        "index",
        repo.to_str().unwrap(),
        "--encoder-config",
        cfg.to_str().unwrap(),
        "--output",
        wrong.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(!wrong.exists());
    let mut changed: Value =
        serde_json::from_slice(&fs::read(d.path().join("embedding.json")).unwrap()).unwrap();
    changed["query_prefix"] = json!("changed: ");
    fs::write(
        d.path().join("embedding.json"),
        serde_json::to_vec(&changed).unwrap(),
    )
    .unwrap();
    let requests_before_mismatch = fs::read(d.path().join("requests")).unwrap();
    let out = cli(&[
        "search",
        index.to_str().unwrap(),
        "read",
        "--encoder-config",
        cfg.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Incompatible"));
    assert_eq!(
        fs::read(d.path().join("requests")).unwrap(),
        requests_before_mismatch
    );
}
