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
use cubeql_embeddings::{EmbeddingProvider, HashProvider, embed_query};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::contract::{
    CheckRequest, CheckResponse, Location, MAX_RESPONSE_BYTES, ReadRequest, ReadResponse,
    SearchHit, SearchRequest, SearchResponse, Strategy, ToolFailure, prefix_end,
};

#[derive(Clone, Debug)]
pub struct ServerOptions {
    pub index: PathBuf,
    pub model_dir: Option<PathBuf>,
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
}
impl Request {
    fn method(&self) -> &'static str {
        match self {
            Self::Search(_) => "search_code",
            Self::Read(_) => "read_code",
            Self::Check(_) => "check_code",
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
        let task = move || {
            let startup_start = Instant::now();
            if options.text_only
                && options.search_strategy.is_some_and(|s| {
                    cubeql_code::retrieval::SearchStrategy::from(s).needs_embedding()
                })
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
            // Construct every non-Send model/device value inside its owning thread.
            let initialized = initialize(&options);
            let (id, code, mut provider) = match initialized {
                Ok(value) => value,
                Err(error) => {
                    let _ = ready.send(Err(error));
                    return;
                }
            };
            let (resident, host, backend) = if provider.is_some() {
                let device = cubecl::Device::default();
                match CodeSearchIndex::new(&device, code) {
                    Ok(resident) => (Some(resident), None, format!("{device:?}")),
                    Err(error) => {
                        let _ = ready.send(Err(failed(error)));
                        return;
                    }
                }
            } else {
                (None, Some(code), "Host".into())
            };
            let code = resident
                .as_ref()
                .map(CodeSearchIndex::code)
                .or(host.as_ref())
                .unwrap();
            let text = SnapshotTextIndex::new(code);
            text.warm_up(SearchStrategy::Hybrid);
            eprintln!(
                "{}",
                json!({"format_version":1, "event":"ready", "snapshot_id":id, "chunks":code.chunks().len(),
                "text_only":options.text_only, "backend":backend, "precision":code.precision(),
                "workspace_checks":workspace.is_some(),
                "search_strategy":options.search_strategy,
                "embedding_model":code.embedding_config().map(|c| c.model.as_str()),
                "startup_ms":startup_start.elapsed().as_secs_f64()*1000.0})
            );
            if ready.send(Ok(())).is_err() {
                return;
            }
            let mut sequence = 0u64;
            for job in receiver {
                // Cancelled queued calls never pay for inference; running kernels are not preempted.
                if job.reply.is_closed() {
                    continue;
                }
                sequence += 1;
                let start = Instant::now();
                let queue_ms = job.enqueued.elapsed().as_secs_f64() * 1000.0;
                let method = job.request.method();
                let result = match job.request {
                    Request::Search(request) if options.search_strategy.is_some_and(|s| s != request.strategy) => Err(ToolFailure::new("strategy_restricted", "this server restricts search_code to its configured --search-strategy; supply that strategy explicitly")),
                    Request::Search(request) => search(
                        &id,
                        code,
                        &text,
                        resident.as_ref(),
                        &mut provider,
                        &backend,
                        request,
                    ),
                    Request::Read(request) => read(&id, code, request),
                    Request::Check(request) => check(&id, code, workspace.as_ref(), request),
                }
                .and_then(bounded);
                eprintln!(
                    "{}",
                    json!({"format_version":1, "event":"request_finished", "request_id":sequence,
                    "method":method, "snapshot_id":id, "queue_ms":queue_ms,
                    "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
                    "strategy":result.as_ref().ok().and_then(|v| v.get("strategy")),
                    "candidate_k":result.as_ref().ok().and_then(|v| v.get("candidate_k")),
                    "read_chunk_id":(method == "read_code").then(|| result.as_ref().ok().and_then(|v| v.get("chunk_id"))).flatten(),
                    "checked_chunk_id":(method == "check_code").then(|| result.as_ref().ok().and_then(|v| v.get("chunk_id"))).flatten(),
                    "freshness":result.as_ref().ok().and_then(|v| v.get("status")),
                    "returned_chunk_ids":result.as_ref().ok().and_then(|v| v.get("hits")).and_then(Value::as_array)
                        .map(|hits| hits.iter().map(|h| h["chunk_id"].clone()).collect::<Vec<_>>()),
                    "error":result.as_ref().err().map(|e| e.code)})
                );
                let _ = job.reply.send(result);
            }
        };
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

type Initialized = (String, CodeIndex, Option<Box<dyn EmbeddingProvider>>);
fn initialize(options: &ServerOptions) -> Result<Initialized, ToolFailure> {
    if options.text_only && options.model_dir.is_some() {
        return Err(ToolFailure::invalid(
            "--model-dir cannot be used with --text-only",
        ));
    }
    // Hash and parse the same bytes, rather than opening the path twice across a replacement.
    let bytes = std::fs::read(&options.index).map_err(failed)?;
    let id = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let code = CodeIndex::read_ipc(Cursor::new(bytes)).map_err(failed)?;
    let provider = if options.text_only {
        None
    } else {
        Some(saved_provider(&code, options.model_dir.as_deref())?)
    };
    Ok((id, code, provider))
}

fn saved_provider(
    code: &CodeIndex,
    directory: Option<&Path>,
) -> Result<Box<dyn EmbeddingProvider>, ToolFailure> {
    let config = code.embedding_config().ok_or_else(|| {
        ToolFailure::invalid("semantic search requires a provider-aware snapshot")
    })?;
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
                "unsupported embedding provider; use --text-only for lexical/symbol search",
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
