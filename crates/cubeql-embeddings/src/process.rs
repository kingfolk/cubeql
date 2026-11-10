use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{EmbeddingConfig, EmbeddingError, EmbeddingProvider};

const MAX_FRAME: usize = 1024 * 1024;
const MAX_BATCH: usize = 32;
const STDERR_CAP: usize = 64 * 1024;
const MAX_CONFIG_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderProcessConfig {
    pub format_version: u32,
    pub command: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_ms: u64,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_ms: u64,
}

fn default_startup_timeout() -> u64 {
    120_000
}
fn default_request_timeout() -> u64 {
    30_000
}

impl EncoderProcessConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take((MAX_CONFIG_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_CONFIG_BYTES {
            return Err(EmbeddingError::Invalid(
                "encoder process config exceeds 1 MiB".into(),
            ));
        }
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), EmbeddingError> {
        if self.format_version != 1 {
            return Err(EmbeddingError::Invalid(
                "unsupported encoder process format_version".into(),
            ));
        }
        if !self.command.is_absolute() {
            return Err(EmbeddingError::Invalid(
                "encoder command must be an absolute path".into(),
            ));
        }
        if !(1..=600_000).contains(&self.startup_timeout_ms)
            || !(1..=600_000).contains(&self.request_timeout_ms)
        {
            return Err(EmbeddingError::Invalid(
                "encoder timeouts must be in 1..=600000 ms".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct Request<'a> {
    protocol_version: u32,
    id: u64,
    role: &'static str,
    texts: &'a [&'a str],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    protocol_version: u32,
    event: String,
    config: EmbeddingConfig,
    model_load_ms: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultFrame {
    protocol_version: u32,
    id: u64,
    vectors: Vec<Vec<f32>>,
    encoding_ms: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ErrorFrame {
    protocol_version: u32,
    id: u64,
    error: String,
}

enum CommandMsg {
    Encode {
        id: u64,
        role: &'static str,
        texts: Vec<String>,
        reply: SyncSender<Result<Vec<Vec<f32>>, String>>,
    },
    Stop,
}

/// A serialized client for a long-lived encoder implementing the JSONL protocol.
pub struct ProcessProvider {
    config: EmbeddingConfig,
    tx: Option<SyncSender<CommandMsg>>,
    actor: Option<JoinHandle<()>>,
    child_pid: u32,
    poisoned: bool,
    next_id: u64,
    timeout: Duration,
    killed: Arc<AtomicBool>,
    stderr: Arc<Mutex<Vec<u8>>>,
}

impl ProcessProvider {
    pub fn spawn(
        config: &EncoderProcessConfig,
        expected: Option<&EmbeddingConfig>,
    ) -> Result<Self, EmbeddingError> {
        #[cfg(not(unix))]
        {
            let _ = (config, expected);
            return Err(EmbeddingError::Backend(
                "resident encoder processes currently require POSIX process groups".into(),
            ));
        }
        config.validate()?;
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: setpgid is async-signal-safe and only establishes ownership of this child group.
            unsafe {
                command.pre_exec(|| {
                    if libc::setpgid(0, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        let child = command.spawn().map_err(|e| {
            EmbeddingError::Backend(format!("failed to start encoder process: {e}"))
        })?;
        let pid = child.id();
        let killed = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::sync_channel(1);
        let (start_tx, start_rx) = mpsc::sync_channel(1);
        let actor_killed = killed.clone();
        let stderr = Arc::new(Mutex::new(Vec::with_capacity(STDERR_CAP)));
        let actor_stderr = stderr.clone();
        let actor = match thread::Builder::new()
            .name("cubeql-encoder-process".into())
            .spawn(move || {
                actor(child, rx, start_tx, actor_killed, actor_stderr);
            }) {
            Ok(actor) => actor,
            Err(e) => {
                kill_pid(pid, &killed);
                reap_pid(pid);
                return Err(EmbeddingError::Backend(format!(
                    "failed to create encoder actor: {e}"
                )));
            }
        };

        let ready = match start_rx.recv_timeout(Duration::from_millis(config.startup_timeout_ms)) {
            Ok(Ok(ready)) => ready,
            Ok(Err(e)) => {
                let detail = stop_actor(tx, actor, pid, &killed, &stderr);
                return Err(EmbeddingError::Backend(format!(
                    "encoder startup failed: {}",
                    with_stderr(&e, &detail)
                )));
            }
            Err(_) => {
                let detail = stop_actor(tx, actor, pid, &killed, &stderr);
                return Err(EmbeddingError::Backend(with_stderr(
                    "encoder startup timed out",
                    &detail,
                )));
            }
        };
        if let Err(error) = ready.config.validate() {
            let _ = stop_actor(tx, actor, pid, &killed, &stderr);
            return Err(error);
        }
        if ready.protocol_version != 1 || ready.event != "ready" || !valid_time(ready.model_load_ms)
        {
            let _ = stop_actor(tx, actor, pid, &killed, &stderr);
            return Err(EmbeddingError::Backend(
                "invalid encoder ready frame".into(),
            ));
        }
        if let Some(expected) = expected
            && let Err(e) = ready.config.ensure_compatible(expected)
        {
            let _ = stop_actor(tx, actor, pid, &killed, &stderr);
            return Err(e);
        }
        Ok(Self {
            config: ready.config,
            tx: Some(tx),
            actor: Some(actor),
            child_pid: pid,
            poisoned: false,
            next_id: 1,
            timeout: Duration::from_millis(config.request_timeout_ms),
            killed,
            stderr,
        })
    }

    fn encode(
        &mut self,
        role: &'static str,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if self.poisoned {
            return Err(EmbeddingError::Backend(
                "encoder process is poisoned".into(),
            ));
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() > MAX_BATCH {
            return Err(EmbeddingError::Invalid(
                "encoder batches are limited to 32 texts".into(),
            ));
        }
        if request_frame_len(self.next_id, role, texts).is_none_or(|size| size > MAX_FRAME) {
            return Err(EmbeddingError::Invalid(
                "encoder request exceeds 1 MiB frame limit".into(),
            ));
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| EmbeddingError::Backend("encoder request ID exhausted".into()))?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let msg = CommandMsg::Encode {
            id,
            role,
            texts: texts.iter().map(|s| (*s).to_owned()).collect(),
            reply: reply_tx,
        };
        let Some(tx) = self.tx.as_ref() else {
            return self.fail("encoder actor stopped");
        };
        match tx.try_send(msg) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return self.fail("encoder actor did not accept request"),
            Err(TrySendError::Disconnected(_)) => return self.fail("encoder actor stopped"),
        }
        match reply_rx.recv_timeout(self.timeout) {
            Ok(Ok(vectors)) => {
                if let Err(error) = self.config.validate_vectors(&vectors, texts.len()) {
                    return self.fail(&error.to_string());
                }
                Ok(vectors)
            }
            Ok(Err(message)) => self.fail(&message),
            Err(_) => self.fail("encoder request timed out"),
        }
    }

    fn fail<T>(&mut self, message: &str) -> Result<T, EmbeddingError> {
        self.poisoned = true;
        let stderr = self.shutdown();
        Err(EmbeddingError::Backend(with_stderr(message, &stderr)))
    }

    fn shutdown(&mut self) -> String {
        match (self.tx.take(), self.actor.take()) {
            (Some(tx), Some(actor)) => {
                stop_actor(tx, actor, self.child_pid, &self.killed, &self.stderr)
            }
            _ => {
                kill_pid(self.child_pid, &self.killed);
                stderr_text(&self.stderr)
            }
        }
    }
}

impl EmbeddingProvider for ProcessProvider {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    fn execution_kind(&self) -> &'static str {
        "resident-process-encoding"
    }

    fn embed_documents(&mut self, documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut vectors = Vec::with_capacity(documents.len());
        for chunk in documents.chunks(MAX_BATCH) {
            vectors.extend(self.encode("document", chunk)?);
        }
        Ok(vectors)
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        if query.trim().is_empty() {
            return Err(EmbeddingError::Invalid(
                "query text must not be blank".into(),
            ));
        }
        self.encode("query", &[query])?
            .pop()
            .ok_or_else(|| EmbeddingError::Backend("encoder returned no query vector".into()))
    }
}

impl Drop for ProcessProvider {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn actor(
    mut child: Child,
    rx: Receiver<CommandMsg>,
    start: SyncSender<Result<Ready, String>>,
    killed: Arc<AtomicBool>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
) {
    let mut stderr_pump = None;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    if let (Some(stdin), Some(stdout)) = (stdin, stdout) {
        if let Some(stderr) = child.stderr.take() {
            let tail = stderr_tail.clone();
            stderr_pump = thread::Builder::new()
                .name("cubeql-encoder-stderr".into())
                .spawn(move || drain_stderr(stderr, tail))
                .ok();
        }
        let mut reader = BufReader::new(stdout);
        let mut writer = stdin;
        let startup = read_frame(&mut reader).and_then(|bytes| {
            serde_json::from_slice::<Ready>(&bytes).map_err(|e| format!("invalid ready JSON: {e}"))
        });
        match startup {
            Ok(ready) => {
                if start.send(Ok(ready)).is_err() {
                    kill_child(&mut child, &killed);
                }
            }
            Err(e) => {
                let _ = start.send(Err(e));
                kill_child(&mut child, &killed);
            }
        }
        if !killed.load(Ordering::SeqCst) {
            while let Ok(msg) = rx.recv() {
                match msg {
                    CommandMsg::Stop => break,
                    CommandMsg::Encode {
                        id,
                        role,
                        texts,
                        reply,
                    } => {
                        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                        let outcome = transact(&mut writer, &mut reader, id, role, &refs);
                        let fatal = outcome.is_err();
                        let _ = reply.send(outcome);
                        if fatal {
                            break;
                        }
                    }
                }
            }
        }
    }
    kill_child(&mut child, &killed);
    if let Some(pump) = stderr_pump {
        let _ = pump.join();
    }
}

fn transact(
    writer: &mut ChildStdin,
    reader: &mut BufReader<std::process::ChildStdout>,
    id: u64,
    role: &'static str,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>, String> {
    let request = serde_json::to_vec(&Request {
        protocol_version: 1,
        id,
        role,
        texts,
    })
    .map_err(|e| e.to_string())?;
    if request.len() + 1 > MAX_FRAME {
        return Err("encoder request exceeds 1 MiB frame limit".into());
    }
    writer
        .write_all(&request)
        .and_then(|_| writer.write_all(b"\n"))
        .and_then(|_| writer.flush())
        .map_err(|e| format!("encoder request write failed: {e}"))?;
    let bytes = read_frame(reader)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid response JSON: {e}"))?;
    if value.get("error").is_some() {
        let frame: ErrorFrame = serde_json::from_value(value)
            .map_err(|e| format!("invalid encoder error frame: {e}"))?;
        if frame.protocol_version != 1 || frame.id != id || frame.error.len() > 4096 {
            return Err("invalid encoder error frame identity or message".into());
        }
        return Err(format!("encoder reported fatal error: {}", frame.error));
    }
    let frame: ResultFrame =
        serde_json::from_value(value).map_err(|e| format!("invalid encoder result frame: {e}"))?;
    if frame.protocol_version != 1 || frame.id != id || !valid_time(frame.encoding_ms) {
        return Err("invalid encoder result identity or timing".into());
    }
    if frame.vectors.len() != texts.len() {
        return Err(format!(
            "encoder returned {} vectors for {} texts",
            frame.vectors.len(),
            texts.len()
        ));
    }
    Ok(frame.vectors)
}

fn read_frame(reader: &mut impl BufRead) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let n = reader
        .take((MAX_FRAME + 1) as u64)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| format!("encoder response read failed: {e}"))?;
    if n == 0 {
        return Err("encoder closed stdout".into());
    }
    if n > MAX_FRAME || !bytes.ends_with(b"\n") {
        return Err("encoder frame exceeds 1 MiB or is unterminated".into());
    }
    bytes.pop();
    Ok(bytes)
}

fn valid_time(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn request_frame_len(id: u64, role: &str, texts: &[&str]) -> Option<usize> {
    let mut length =
        format!("{{\"protocol_version\":1,\"id\":{id},\"role\":\"{role}\",\"texts\":[")
            .len()
            .checked_add(3)?; // closing bracket/object and JSONL newline
    for (index, text) in texts.iter().enumerate() {
        if index > 0 {
            length = length.checked_add(1)?;
        }
        length = length.checked_add(json_string_len(text)?)?;
    }
    Some(length)
}

fn json_string_len(value: &str) -> Option<usize> {
    let mut length = 2usize; // quotes
    for character in value.chars() {
        let extra = match character {
            '"' | '\\' | '\u{0008}' | '\u{000c}' | '\n' | '\r' | '\t' => 2,
            c if c <= '\u{001f}' => 6,
            c => c.len_utf8(),
        };
        length = length.checked_add(extra)?;
    }
    Some(length)
}

fn drain_stderr(mut stderr: impl Read, tail: Arc<Mutex<Vec<u8>>>) {
    let mut buffer = [0u8; 8192];
    loop {
        match stderr.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let Ok(mut tail) = tail.lock() else { continue };
                if n >= STDERR_CAP {
                    tail.clear();
                    tail.extend_from_slice(&buffer[n - STDERR_CAP..n]);
                } else {
                    let overflow = tail.len().saturating_add(n).saturating_sub(STDERR_CAP);
                    if overflow > 0 {
                        tail.drain(..overflow);
                    }
                    tail.extend_from_slice(&buffer[..n]);
                }
            }
        }
    }
}

fn stderr_text(stderr: &Mutex<Vec<u8>>) -> String {
    stderr
        .lock()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

fn with_stderr(message: &str, stderr: &str) -> String {
    if stderr.is_empty() {
        message.to_owned()
    } else {
        format!("{message}; encoder stderr: {stderr}")
    }
}

fn stop_actor(
    tx: SyncSender<CommandMsg>,
    actor: JoinHandle<()>,
    pid: u32,
    killed: &AtomicBool,
    stderr: &Mutex<Vec<u8>>,
) -> String {
    let _ = tx.try_send(CommandMsg::Stop);
    kill_pid(pid, killed);
    drop(tx);
    let _ = actor.join();
    stderr_text(stderr)
}

fn kill_child(child: &mut Child, killed: &AtomicBool) {
    if !killed.swap(true, Ordering::SeqCst) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.kill();
    }
    let _ = child.wait();
}

fn kill_pid(pid: u32, killed: &AtomicBool) {
    if !killed.swap(true, Ordering::SeqCst) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
}

fn reap_pid(pid: u32) {
    #[cfg(unix)]
    unsafe {
        loop {
            let result = libc::waitpid(pid as i32, std::ptr::null_mut(), 0);
            if result >= 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                break;
            }
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
}
