# Read-only code search over MCP

`cubeql-mcp` is the first Milestone 8 adapter. It exposes the existing retrieval APIs as two
tools in a persistent local process. It is not a new database server or a coding-agent harness.
The generic `cubeql`, `cubeql-code` retrieval implementation and embedding-provider interfaces
are unchanged. The adapter uses the official [Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk)
and [stdio transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports).

## Quick start

Build a snapshot with the existing indexer, then build the MCP binary:

```bash
cargo run -p cubeql-code -- index ../cubecl --output /tmp/cubecl-mcp.arrow
cargo build -p cubeql-mcp --release
```

The default indexer uses deterministic hash vectors for offline testing, not learned semantic
embeddings. To try text/symbol search without any model or device initialization, configure an
MCP client to launch:

```bash
/absolute/path/to/cubeql/target/release/cubeql-mcp /tmp/cubecl-mcp.arrow --text-only
```

For clients accepting an `mcpServers` configuration, an illustrative entry is:

```json
{
  "mcpServers": {
    "cubeql": {
      "command": "/absolute/path/to/cubeql/target/release/cubeql-mcp",
      "args": ["/tmp/cubecl-mcp.arrow", "--text-only"]
    }
  }
}
```

Use your client's configuration location and schema. Replace the binary path with the actual
absolute path. Arguments are not a shell command: do not rely on `~` expansion or a working
directory. No client configuration is installed or changed automatically by CubeQL.

The client starts the process, initializes MCP, lists tools and invokes them. Merely running
the command in a terminal waits for newline-delimited JSON-RPC input; it is not an interactive
search prompt. Stdout contains only protocol messages; logs and traces go to stderr.

### Learned embeddings and device scoring

Download a model explicitly, outside the server, then create a compatible snapshot:

```bash
cargo run -p cubeql-code --features local-embeddings -- model download \
  --directory /tmp/cubeql-minilm
cargo run -p cubeql-code --features local-embeddings -- index ../cubecl \
  --embedding minilm --model-dir /tmp/cubeql-minilm \
  --precision fp16 --output /tmp/cubecl-minilm-mcp.arrow
cargo build -p cubeql-mcp --release --features local-embeddings
```

Launch with `/tmp/cubecl-minilm-mcp.arrow --model-dir /tmp/cubeql-minilm`, without `--text-only`.
Semantic and hybrid requests then embed query text and use the resident CubeCL vector index.
The full saved embedding configuration must match the loaded model; there is no silent fallback
to hash vectors. Startup never downloads models. Unsupported providers fail startup unless
`--text-only` is selected, which does not instantiate a provider at all.

CPU is the default backend. For native Metal, build with:

```bash
cargo build -p cubeql-mcp --release --no-default-features \
  --features metal,local-embeddings
```

CUDA, HIP and WGPU features are forwarded in the same way. Feature availability does not prove
runtime compatibility on hardware that has not been tested. MiniLM inference remains on CPU;
the selected CubeCL backend runs vector scoring/top-k, not this embedding model.

## Tool contract v1

MCP envelopes are handled by the SDK. Each successful tool returns the following application
object in `structuredContent`, plus the same JSON as a text content block for compatible clients.
This contract's `format_version: 1` is separate from the existing CLI search JSON version 2.

### `search_code`

Example tool arguments:

```json
{
  "query": "where are unsupported vector operations split?",
  "strategy": "hybrid",
  "top_k": 5,
  "candidate_k": 50,
  "language": "rust",
  "path_prefix": "crates/cubecl-opt/"
}
```

`strategy` is `semantic`, `lexical`, `symbol` or `hybrid`, defaulting to `hybrid`. Text-only
sessions require an explicit `lexical` or `symbol` strategy: unsupported requests return an
error, not an undisclosed strategy change. Filters have the same case-sensitive literal
semantics as the library. Deduplication is enabled. Scores and provenance follow
[hybrid retrieval](hybrid-retrieval.md), not probability/confidence semantics.

Results include:

- `snapshot_id`: BLAKE3 of the exact Arrow IPC bytes read at startup.
- `source: "snapshot"`, strategy, metric, backend, candidate budget and timings.
- Hits containing `chunk_id`, path, language, symbol, whole-chunk line range, content hash,
  score and per-source ranks/raw scores.
- A preview of at most 240 UTF-8 bytes, `preview_truncated` and `suppressed_overlap_count`.
  Full suppressed-hit evidence remains available through the library, not duplicated in this
  compact tool response.

There is no generated summary, live ripgrep search, automatic router or model-selected context.
The first integration deliberately leaves the agent's ordinary grep/file-reading tools intact.

### `read_code`

Copy a handle from a search response:

```json
{
  "snapshot_id": "blake3:<copy the full value from search_code>",
  "chunk_id": 17,
  "offset": 0,
  "max_bytes": 8192
}
```

The response contains the saved source `text`, its chunk location/hash, `offset`, `total_bytes`
and `next_offset`. If `next_offset` is not null, repeat with that byte offset and the same handle
to reconstruct the complete chunk. Offsets are relative to the exact saved UTF-8 chunk, not to
the entire file. Non-character-boundary offsets are rejected; returned pages never split a
character. Line ranges always describe the complete chunk, even for a partial page. A chunk can
start/end mid-line, so this is not a full-file line reader.

Row-based chunk IDs are valid only with their matching snapshot ID. A restart with different
IPC bytes produces a different ID even if the source text happens to be unchanged. A mismatched
handle is rejected before looking up its row. No caller-controlled file is opened by this tool.

## Lifetime, boundaries and limits

One dedicated worker thread reads and validates the IPC bytes once, creates the text/symbol
caches once, and, in full mode, loads one provider and one resident vector index. Requests are
serialized on that thread. It owns the values directly, avoiding self-referential application
objects and avoiding a new Send/Sync requirement on the existing provider trait. The async MCP
handler sends work through a bounded queue and awaits a reply without blocking its runtime.

- One immutable snapshot per process; no reload, watcher, update or write tool.
- Restart after separately building an updated snapshot. Replacing/removing the IPC file or
  editing the repository does not change a running session.
- No live-source freshness validation is implemented. `source: "snapshot"` is explicit: verify
  current files before using retrieved code to make edits. Read saved chunks with `read_code`
  when interpreting saved line numbers; do not assume they still match your checkout.
- At most eight queued jobs plus one active job. Additional admitted tool calls get `busy`.
  Cancelled queued calls are skipped; already running inference/kernels are not preempted.
- Query: at most 8192 UTF-8 bytes, nonblank. `top_k`: 1..=100. Per-source candidates:
  `top_k..=1000`, default `max(50, top_k)`.
- Language filter: at most 256 bytes; path prefix: at most 4096 bytes.
- Source page: 4..=65536 bytes, default 8192; entire application response at most 512 KiB before
  the SDK adds its text fallback/envelope. The duplicated fallback means wire bytes can be larger.
- Limits constrain tool work/results, not total process memory or the SDK's incoming message
  framing. Loaded snapshots/caches/models and transport buffering still consume memory.

Invalid tool arguments, stale handles, unsupported strategies and retrieval failures are
reported with `isError: true` and `{"error":{"code":"...","message":"..."}}`. The SDK rejects
unknown argument fields and type/deserialization failures before invoking the handler, returning
its own textual tool error. Unknown tools return protocol errors. These errors do not terminate
a healthy session. Startup failures go to stderr with a nonzero exit status.

No network listener or network search is enabled. Both tools advertise read-only, non-destructive,
closed-world hints. These hints do not replace client permission checks. A client allowed to
invoke these tools can retrieve any chunk in the configured snapshot: search filters are not
access controls. Source text and previews are untrusted data and can contain prompt injection;
the agent must not treat their contents as instructions. Do not index secrets into a snapshot
you intend to expose to an agent or hosted model.

## Tracing and evaluation status

Stderr emits versioned JSON `ready` and `request_finished` events. Readiness records snapshot ID,
backend, precision, model identity and total startup time. Completed admitted requests record
a process-local sequence ID, tool name, queue/worker latency, successful strategy/candidate budget,
returned chunk IDs (or read chunk ID) and error code. Raw query text and source are not logged.
SDK/startup diagnostics can also appear as non-JSON stderr lines. Argument rejections before
enqueue, queue-full errors and cancelled queued work do not emit completion events.

Search results separate query embedding time from retrieval/filter/result-packing time. These
are not isolated GPU timings. Startup time includes model/index loading and host cache warm-up,
but first-use kernel compilation can still occur on the first semantic request. Traces do not
measure what the agent actually read into its context, model tokens or task success.

The protocol tests launch the real binary, negotiate MCP, discover/call tools, follow UTF-8 pages,
verify snapshot/model boundaries, preserve source after repository/index changes, and exercise
repeated FP32/FP16 semantic/hybrid calls. An opt-in test uses a local learned MiniLM model.

```bash
cargo test -p cubeql-mcp
cargo test -p cubeql-mcp --no-default-features --features metal
CUBEQL_MINILM_DIR=/tmp/cubeql-minilm cargo test -p cubeql-mcp \
  --features local-embeddings -- --include-ignored
```

This establishes an executable adapter, **not completion of Milestone 8's agent-level acceptance
criteria**. No real client is auto-configured, and no end-to-end harness improvement has been
measured. Next, attach it to one real client and compare the same model/tasks/budget with (A)
ordinary search tools, (B) ordinary tools plus CubeQL lexical, and (C) ordinary tools plus hybrid.
Use reviewed real-repository tasks, repeat trials, and measure success, tool calls, tokens and
total latency. The old whole-query literal baseline is not a substitute for that experiment.
