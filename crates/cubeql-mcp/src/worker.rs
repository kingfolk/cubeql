//! A dedicated thread owns inference/device state and borrows its text index within that thread.
//! This avoids self-referential structs and does not require providers/backends to be Send/Sync.
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::mpsc::{self, SyncSender, TrySendError},
    time::Instant,
};

use cubeql_code::{
    CodeChunk, CodeIndex, CodeSearchIndex,
    reporting::CodeFilter,
    retrieval::{RetrievalOptions, SearchStrategy, SnapshotTextIndex},
};
use cubeql_embeddings::{
    EmbeddingProvider, EncoderProcessConfig, HashProvider, ProcessProvider, embed_query,
};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::contract::{
    CheckRequest, CheckResponse, Location, MAX_RESPONSE_BYTES, ReadRequest, ReadResponse,
    ReloadRequest, SearchHit, SearchRequest, SearchResponse, Strategy, ToolFailure, prefix_end,
};

#[derive(Clone, Debug)]
pub struct ServerOptions {
    pub index: PathBuf,
    /// Optional fixed input path for explicit snapshot reload; never supplied by a tool caller.
    pub reload_index: Option<PathBuf>,
    pub model_dir: Option<PathBuf>,
    /// Explicit resident encoder launch profile; incompatible snapshots fail before search.
    pub encoder_config: Option<PathBuf>,
    /// Load only host text/symbol adapters; semantic/hybrid requests then return an explicit error.
    pub text_only: bool,
    /// Explicit authority for live freshness checks. Never use the index's saved root implicitly.
    pub workspace_root: Option<PathBuf>,
    /// Reject other requested strategies instead of silently changing the experiment arm.
    pub search_strategy: Option<Strategy>,
}

pub(crate) enum Request {
    Search(SearchRequest),
    Read(ReadRequest),
    Check(CheckRequest),
    Reload(ReloadRequest),
}
impl Request {
    fn method(&self) -> &'static str {
        match self {
            Self::Search(_) => "search_code",
            Self::Read(_) => "read_code",
            Self::Check(_) => "check_code",
            Self::Reload(_) => "reload_snapshot",
        }
    }
}
struct Job {
    request: Request,
    enqueued: Instant,
    reply: oneshot::Sender<Result<Value, ToolFailure>>,
}

#[derive(Clone)]
pub(crate) struct Worker {
    sender: SyncSender<Job>,
}
impl Worker {
    pub async fn start(options: ServerOptions) -> Result<Self, ToolFailure> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(8);
        let (ready, startup) = oneshot::channel();
        let task = move || run(options, receiver, ready);
        std::thread::Builder::new()
            .name("cubeql-retrieval".into())
            .spawn(task)
            .map_err(failed)?;
        startup.await.map_err(|_| {
            ToolFailure::new("unavailable", "retrieval worker exited during startup")
        })??;
        Ok(Self { sender })
    }

    pub async fn call(&self, request: Request) -> Result<Value, ToolFailure> {
        match &request {
            Request::Search(request) => {
                request.validate()?;
            }
            Request::Read(request) => request.validate()?,
            Request::Check(request) => request.validate()?,
            Request::Reload(request) => request.validate()?,
        }
        let (reply, response) = oneshot::channel();
        self.sender
            .try_send(Job {
                request,
                enqueued: Instant::now(),
                reply,
            })
            .map_err(|e| match e {
                TrySendError::Full(_) => {
                    ToolFailure::new("busy", "retrieval queue is full; retry later")
                }
                TrySendError::Disconnected(_) => {
                    ToolFailure::new("unavailable", "retrieval worker stopped")
                }
            })?;
        response
            .await
            .map_err(|_| ToolFailure::new("unavailable", "retrieval worker stopped"))?
    }
}

struct SnapshotState {
    id: String,
    resident: Option<CodeSearchIndex>,
    host: Option<CodeIndex>,
    backend: String,
}
impl SnapshotState {
    fn prepare(id: String, code: CodeIndex, full: bool) -> Result<Self, ToolFailure> {
        if full {
            let device = cubecl::Device::default();
            let resident = CodeSearchIndex::new(&device, code).map_err(failed)?;
            Ok(Self {
                id,
                resident: Some(resident),
                host: None,
                backend: format!("{device:?}"),
            })
        } else {
            Ok(Self {
                id,
                resident: None,
                host: Some(code),
                backend: "Host".into(),
            })
        }
    }
    fn code(&self) -> &CodeIndex {
        self.resident
            .as_ref()
            .map(CodeSearchIndex::code)
            .or(self.host.as_ref())
            .unwrap()
    }
}

struct Completion {
    reply: oneshot::Sender<Result<Value, ToolFailure>>,
    sequence: u64,
    start: Instant,
    queue_ms: f64,
    method: &'static str,
}
impl Completion {
    fn finish(
        self,
        id: &str,
        provider: &Option<Box<dyn EmbeddingProvider>>,
        result: Result<Value, ToolFailure>,
    ) {
        let result = result.and_then(bounded);
        eprintln!(
            "{}",
            json!({"format_version":1, "event":"request_finished", "request_id":self.sequence,
            "method":self.method, "snapshot_id":id, "queue_ms":self.queue_ms,
            "elapsed_ms":self.start.elapsed().as_secs_f64()*1000.0,
            "strategy":result.as_ref().ok().and_then(|v| v.get("strategy")),
            "candidate_k":result.as_ref().ok().and_then(|v| v.get("candidate_k")),
            "read_chunk_id":(self.method == "read_code").then(|| result.as_ref().ok().and_then(|v| v.get("chunk_id"))).flatten(),
            "checked_chunk_id":(self.method == "check_code").then(|| result.as_ref().ok().and_then(|v| v.get("chunk_id"))).flatten(),
            "freshness":result.as_ref().ok().and_then(|v| v.get("status")),
            "returned_chunk_ids":result.as_ref().ok().and_then(|v| v.get("hits")).and_then(Value::as_array)
                .map(|hits| hits.iter().map(|h| h["chunk_id"].clone()).collect::<Vec<_>>()),
            "embedding_execution":provider.as_ref().map(|p| p.execution_kind()),
            "previous_snapshot_id":result.as_ref().ok().and_then(|v| v.get("previous_snapshot_id")),
            "changed":result.as_ref().ok().and_then(|v| v.get("changed")),
            "error":result.as_ref().err().map(|e| e.code)})
        );
        let _ = self.reply.send(result);
    }
}

struct PendingReload {
    completion: Completion,
    previous_id: String,
    preflight_ms: f64,
    prepare_ms: f64,
}

struct ReloadTimings {
    preflight_ms: f64,
    prepare_ms: f64,
    text_index_ms: f64,
    reload_ms: f64,
}
fn reload_result(
    state: &SnapshotState,
    previous_id: &str,
    changed: bool,
    provider_reused: bool,
    timings: ReloadTimings,
) -> Value {
    json!({"format_version":1, "source":"snapshot_reload", "previous_snapshot_id":previous_id,
        "snapshot_id":state.id, "changed":changed, "chunks":state.code().chunks().len(),
        "provider_reused":provider_reused, "preflight_ms":timings.preflight_ms, "prepare_ms":timings.prepare_ms,
        "text_index_ms":timings.text_index_ms, "reload_ms":timings.reload_ms})
}

fn run(
    options: ServerOptions,
    receiver: mpsc::Receiver<Job>,
    ready: oneshot::Sender<Result<(), ToolFailure>>,
) {
    let startup_start = Instant::now();
    if options.text_only
        && options
            .search_strategy
            .is_some_and(|s| SearchStrategy::from(s).needs_embedding())
    {
        let _ = ready.send(Err(ToolFailure::invalid(
            "text-only mode cannot restrict search to an embedding strategy",
        )));
        return;
    }
    let workspace = match options
        .workspace_root
        .as_ref()
        .map(|path| cap_std::fs::Dir::open_ambient_dir(path, cap_std::ambient_authority()))
        .transpose()
    {
        Ok(workspace) => workspace,
        Err(error) => {
            let _ = ready.send(Err(failed(error)));
            return;
        }
    };
    let (id, code, mut provider, provider_startup_ms) = match initialize(&options) {
        Ok(value) => value,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut state = match SnapshotState::prepare(id, code, provider.is_some()) {
        Ok(state) => state,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut ready = Some(ready);
    let mut pending: Option<PendingReload> = None;
    let mut sequence = 0u64;
    loop {
        // Borrow within one generation: no unsafe self-reference or leaked old snapshots.
        let code = state.code();
        let text = SnapshotTextIndex::new(code);
        let text_start = Instant::now();
        text.warm_up(SearchStrategy::Hybrid);
        let text_index_ms = text_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(ready) = ready.take() {
            eprintln!(
                "{}",
                json!({"format_version":1, "event":"ready", "snapshot_id":state.id,
                "chunks":code.chunks().len(), "text_only":options.text_only, "backend":state.backend,
                "precision":code.precision(), "workspace_checks":workspace.is_some(),
                "search_strategy":options.search_strategy, "reload_enabled":options.reload_index.is_some(),
                "embedding_model":code.embedding_config().map(|c| c.model.as_str()),
                "embedding_execution":provider.as_ref().map(|p| p.execution_kind()),
                "provider_startup_ms":provider_startup_ms,
                "startup_ms":startup_start.elapsed().as_secs_f64()*1000.0})
            );
            if ready.send(Ok(())).is_err() {
                return;
            }
        }
        if let Some(reload) = pending.take() {
            let result = reload_result(
                &state,
                &reload.previous_id,
                true,
                provider.is_some(),
                ReloadTimings {
                    preflight_ms: reload.preflight_ms,
                    prepare_ms: reload.prepare_ms,
                    text_index_ms,
                    reload_ms: reload.completion.start.elapsed().as_secs_f64() * 1000.0,
                },
            );
            reload.completion.finish(&state.id, &provider, Ok(result));
        }
        let transition = loop {
            let job = match receiver.recv() {
                Ok(job) => job,
                Err(_) => return,
            };
            if job.reply.is_closed() {
                continue;
            }
            sequence += 1;
            let method = job.request.method();
            let completion = Completion {
                reply: job.reply,
                sequence,
                start: Instant::now(),
                queue_ms: job.enqueued.elapsed().as_secs_f64() * 1000.0,
                method,
            };
            let result = match job.request {
                Request::Reload(request) => {
                    // Compare before opening the input. A stale queued reload cannot replace a
                    // newer active generation. Cancellation after admission does not undo commit.
                    if request.expected_snapshot_id != state.id {
                        completion.finish(
                            &state.id,
                            &provider,
                            Err(ToolFailure::new(
                                "snapshot_mismatch",
                                "active snapshot changed; search again before reloading",
                            )),
                        );
                        continue;
                    }
                    let Some(path) = options.reload_index.as_ref() else {
                        completion.finish(
                            &state.id,
                            &provider,
                            Err(ToolFailure::new(
                                "reload_disabled",
                                "operator did not configure --reload-index",
                            )),
                        );
                        continue;
                    };
                    let start = Instant::now();
                    let candidate = match crate::snapshot::load_candidate(
                        path,
                        provider.as_ref().map(|p| p.config()),
                        crate::snapshot::MAX_RELOAD_BYTES,
                    ) {
                        Ok(candidate) => candidate,
                        Err(error) => {
                            completion.finish(&state.id, &provider, Err(error));
                            continue;
                        }
                    };
                    let preflight_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if candidate.snapshot_id == state.id {
                        let result = reload_result(
                            &state,
                            &state.id,
                            false,
                            provider.is_some(),
                            ReloadTimings {
                                preflight_ms,
                                prepare_ms: 0.0,
                                text_index_ms: 0.0,
                                reload_ms: completion.start.elapsed().as_secs_f64() * 1000.0,
                            },
                        );
                        completion.finish(&state.id, &provider, Ok(result));
                        continue;
                    }
                    let start = Instant::now();
                    let next = match SnapshotState::prepare(
                        candidate.snapshot_id,
                        candidate.code,
                        provider.is_some(),
                    ) {
                        Ok(next) => next,
                        Err(error) => {
                            completion.finish(
                                &state.id,
                                &provider,
                                Err(ToolFailure::new("reload_failed", error.message)),
                            );
                            continue;
                        }
                    };
                    break (
                        next,
                        PendingReload {
                            completion,
                            previous_id: state.id.clone(),
                            preflight_ms,
                            prepare_ms: start.elapsed().as_secs_f64() * 1000.0,
                        },
                    );
                }
                Request::Search(request)
                    if options
                        .search_strategy
                        .is_some_and(|s| s != request.strategy) =>
                {
                    Err(ToolFailure::new(
                        "strategy_restricted",
                        "this server restricts search_code to its configured --search-strategy; supply that strategy explicitly",
                    ))
                }
                Request::Search(request) => search(
                    &state.id,
                    code,
                    &text,
                    state.resident.as_ref(),
                    &mut provider,
                    &state.backend,
                    request,
                ),
                Request::Read(request) => read(&state.id, code, request),
                Request::Check(request) => check(&state.id, code, workspace.as_ref(), request),
            };
            completion.finish(&state.id, &provider, result);
        };
        // All fallible parse/compatibility/device preparation succeeded. Drop the borrowed
        // adapters before replacing the owner; rebuild host tables before acknowledging.
        // Allocation aborts/panics are process failures, not recoverable reload errors.
        drop(text);
        state = transition.0;
        pending = Some(transition.1);
    }
}

type Initialized = (String, CodeIndex, Option<Box<dyn EmbeddingProvider>>, f64);
fn initialize(options: &ServerOptions) -> Result<Initialized, ToolFailure> {
    if (options.text_only && (options.model_dir.is_some() || options.encoder_config.is_some()))
        || (options.model_dir.is_some() && options.encoder_config.is_some())
    {
        return Err(ToolFailure::invalid(
            "--encoder-config, --model-dir and --text-only are mutually exclusive",
        ));
    }
    // Hash and parse the same bytes, rather than opening the path twice across a replacement.
    let bytes = std::fs::read(&options.index).map_err(failed)?;
    let id = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let code = CodeIndex::read_ipc(Cursor::new(bytes)).map_err(failed)?;
    let provider_start = Instant::now();
    let provider = if options.text_only {
        None
    } else {
        Some(saved_provider(
            &code,
            options.model_dir.as_deref(),
            options.encoder_config.as_deref(),
        )?)
    };
    let provider_startup_ms = if options.text_only {
        0.0
    } else {
        provider_start.elapsed().as_secs_f64() * 1000.0
    };
    Ok((id, code, provider, provider_startup_ms))
}

fn saved_provider(
    code: &CodeIndex,
    directory: Option<&Path>,
    encoder_config: Option<&Path>,
) -> Result<Box<dyn EmbeddingProvider>, ToolFailure> {
    let config = code.embedding_config().ok_or_else(|| {
        ToolFailure::invalid("semantic search requires a provider-aware snapshot")
    })?;
    if let Some(path) = encoder_config {
        let options = EncoderProcessConfig::load(path).map_err(failed)?;
        return Ok(Box::new(
            ProcessProvider::spawn(&options, Some(config)).map_err(failed)?,
        ));
    }
    let provider: Box<dyn EmbeddingProvider> = match config.model.as_str() {
        "cubeql-hash-tokens" => {
            if directory.is_some() {
                return Err(ToolFailure::invalid(
                    "hash snapshots do not use --model-dir",
                ));
            }
            eprintln!(
                "Token-hash snapshot: vectors are deterministic fixtures, not learned semantic embeddings."
            );
            Box::new(HashProvider::new(config.dimensions).map_err(failed)?)
        }
        "sentence-transformers/all-MiniLM-L6-v2" => {
            #[cfg(feature = "local-embeddings")]
            {
                Box::new(
                    cubeql_embeddings::MiniLm::from_dir(directory.ok_or_else(|| {
                        ToolFailure::invalid(
                            "MiniLM requires --model-dir; download it separately with cubeql-code",
                        )
                    })?)
                    .map_err(failed)?,
                )
            }
            #[cfg(not(feature = "local-embeddings"))]
            {
                return Err(ToolFailure::invalid(
                    "MiniLM requires --features local-embeddings",
                ));
            }
        }
        _ => {
            return Err(ToolFailure::invalid(
                "unsupported embedding provider; supply --encoder-config for a compatible resident encoder, or use --text-only for lexical/symbol search",
            ));
        }
    };
    config
        .ensure_compatible(provider.config())
        .map_err(failed)?;
    Ok(provider)
}

fn search(
    id: &str,
    code: &CodeIndex,
    text: &SnapshotTextIndex<'_>,
    resident: Option<&CodeSearchIndex>,
    provider: &mut Option<Box<dyn EmbeddingProvider>>,
    backend: &str,
    request: SearchRequest,
) -> Result<Value, ToolFailure> {
    let candidate_k = request.validate()?;
    let strategy = SearchStrategy::from(request.strategy);
    let start = Instant::now();
    let query = if strategy.needs_embedding() {
        let provider = provider.as_mut().ok_or_else(|| {
            ToolFailure::new(
                "unsupported_strategy",
                "text-only server supports only lexical and symbol; select strategy explicitly",
            )
        })?;
        Some(embed_query(provider.as_mut(), &request.query).map_err(failed)?)
    } else {
        None
    };
    let embedding_ms = if query.is_some() {
        start.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let start = Instant::now();
    let filter = CodeFilter {
        language: request.language,
        path_prefix: request.path_prefix,
    };
    // Do not prepare/upload a device selection for host-only requests, even on a full server.
    let selected = text
        .prepare(
            &filter,
            if strategy.needs_embedding() {
                resident
            } else {
                None
            },
        )
        .map_err(failed)?;
    let result = selected
        .query(
            &request.query,
            query.as_ref(),
            RetrievalOptions {
                strategy,
                top_k: request.top_k,
                candidate_k,
                deduplicate: true,
            },
        )
        .map_err(failed)?;
    let hits = result
        .hits
        .into_iter()
        .map(|hit| {
            let chunk = &code.chunks()[hit.row];
            let end = prefix_end(&chunk.source, 240);
            SearchHit {
                location: location(hit.row, chunk),
                score: hit.score,
                provenance: hit.provenance,
                preview: chunk.source[..end].into(),
                preview_truncated: end < chunk.source.len(),
                suppressed_overlap_count: hit.suppressed_overlaps.len(),
            }
        })
        .collect();
    serde_json::to_value(SearchResponse {
        format_version: 1,
        snapshot_id: id.into(),
        source: "snapshot",
        strategy: request.strategy,
        metric: strategy.score_kind(),
        backend: if strategy.needs_embedding() {
            backend
        } else {
            "Host"
        }
        .into(),
        candidate_k,
        hits,
        embedding_ms,
        retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
    .map_err(failed)
}

fn read(id: &str, code: &CodeIndex, request: ReadRequest) -> Result<Value, ToolFailure> {
    request.validate()?;
    if request.snapshot_id != id {
        return Err(ToolFailure::new(
            "snapshot_mismatch",
            "snapshot_id does not match this server; search again",
        ));
    }
    let chunk = code
        .chunks()
        .get(request.chunk_id)
        .ok_or_else(|| ToolFailure::invalid("unknown chunk_id"))?;
    if request.offset > chunk.source.len() || !chunk.source.is_char_boundary(request.offset) {
        return Err(ToolFailure::invalid(
            "offset must be a UTF-8 boundary within the saved chunk",
        ));
    }
    let remainder = &chunk.source[request.offset..];
    let end = request.offset + prefix_end(remainder, request.max_bytes);
    serde_json::to_value(ReadResponse {
        format_version: 1,
        snapshot_id: id.into(),
        source: "snapshot",
        location: location(request.chunk_id, chunk),
        offset: request.offset,
        next_offset: (end < chunk.source.len()).then_some(end),
        total_bytes: chunk.source.len(),
        text: chunk.source[request.offset..end].into(),
    })
    .map_err(failed)
}

fn location(row: usize, chunk: &CodeChunk) -> Location {
    Location {
        chunk_id: row,
        path: chunk.path.clone(),
        language: chunk.language.clone(),
        symbol: chunk.symbol.clone(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        content_hash: chunk.content_hash.clone(),
    }
}

fn check(
    id: &str,
    code: &CodeIndex,
    workspace: Option<&cap_std::fs::Dir>,
    request: CheckRequest,
) -> Result<Value, ToolFailure> {
    request.validate()?;
    if request.snapshot_id != id {
        return Err(ToolFailure::new(
            "snapshot_mismatch",
            "snapshot_id does not match this server; search again",
        ));
    }
    let chunk = code
        .chunks()
        .get(request.chunk_id)
        .ok_or_else(|| ToolFailure::invalid("unknown chunk_id"))?;
    let expected = code.file_hash(&chunk.path);
    let checked = crate::freshness::check_file(
        workspace,
        &chunk.path,
        expected,
        crate::freshness::MAX_CHECK_BYTES,
    );
    serde_json::to_value(CheckResponse {
        format_version: 1,
        snapshot_id: id.into(),
        source: "workspace_check",
        location: location(request.chunk_id, chunk),
        status: checked.status,
        reason: checked.reason,
        expected_file_hash: expected.map(str::to_owned),
        actual_file_hash: checked.actual_hash,
        checked_bytes: checked.bytes,
    })
    .map_err(failed)
}
fn bounded(value: Value) -> Result<Value, ToolFailure> {
    if serde_json::to_vec(&value).map_err(failed)?.len() > MAX_RESPONSE_BYTES {
        return Err(ToolFailure::new(
            "response_limit",
            "result exceeds 512 KiB; request fewer hits or source bytes",
        ));
    }
    Ok(value)
}
fn failed(error: impl std::fmt::Display) -> ToolFailure {
    ToolFailure::new("retrieval_failed", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_request() -> Request {
        Request::Read(ReadRequest {
            snapshot_id: "test".into(),
            chunk_id: 0,
            offset: 0,
            max_bytes: 8192,
        })
    }

    #[tokio::test]
    async fn saturated_or_disconnected_workers_fail_without_blocking_the_async_runtime() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (reply, _response) = oneshot::channel();
        assert!(
            sender
                .try_send(Job {
                    request: read_request(),
                    enqueued: Instant::now(),
                    reply,
                })
                .is_ok()
        );
        let worker = Worker { sender };
        let error = worker.call(read_request()).await.unwrap_err();
        assert_eq!(error.code, "busy");
        drop(receiver);
        let error = worker.call(read_request()).await.unwrap_err();
        assert_eq!(error.code, "unavailable");
    }

    #[test]
    fn oversized_results_are_rejected_instead_of_silently_truncated() {
        let small = json!({"text":"small"});
        assert_eq!(bounded(small.clone()).unwrap(), small);
        let large = json!({"text":"x".repeat(MAX_RESPONSE_BYTES)});
        assert_eq!(bounded(large).unwrap_err().code, "response_limit");
    }
}
