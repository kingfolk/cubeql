# Resident local code encoder

A configured local process can implement `EmbeddingProvider` without adding Python or model
runtime dependencies to the generic Arrow/CubeCL engine. Its model/tokenizer load once and
serve serialized document/query calls. The bundled worker shares the pinned CodeRank/Jina
inference recipe with [the offline exporter](precomputed-vectors.md).

## Launch profile and library API

Install the matching pinned requirements into a separate environment, then explicitly fetch
and verify the model with `tools/encode_vectors.py fetch` as documented in the offline guide.
The live worker verifies local files and runs offline; it never downloads a missing model.

Create a local launch profile with absolute paths (replace the example paths):

```json
{
  "format_version": 1,
  "command": "/absolute/code-env/bin/python",
  "args": ["-B", "/absolute/cubeql/tools/serve_encoder.py",
           "--model", "coderank", "--model-dir", "/absolute/models/coderank",
           "--max-tokens", "256", "--device", "cpu"],
  "startup_timeout_ms": 120000,
  "request_timeout_ms": 30000
}
```

Keep the interpreter/library versions, device and full model configuration consistent with
the document index. Jina is an optional CC-BY-NC-4.0 reference, not an automatically selected
production default. Use its separate environment and explicit `--model jina` profile.

```rust,ignore
use cubeql_embeddings::{EncoderProcessConfig, ProcessProvider};
let launch = EncoderProcessConfig::load("/absolute/encoder.json")?;
// New index creation can discover the configured encoder's contract.
let mut documents = ProcessProvider::spawn(&launch, None)?;
// Searching a saved index must supply and match its entire EmbeddingConfig.
let mut queries = ProcessProvider::spawn(&launch, Some(saved_config))?;
```

The two lines illustrate separate clients; a long-lived application normally retains one
provider rather than launching a process for each query. No shell interpolation is used.
The launch profile runs a trusted user-selected executable; it is not a process sandbox.

## CLI and persistent MCP search

Save the launch profile above as `/absolute/encoder.json`, then build and query a snapshot:

```sh
cargo build -p cubeql-code -p cubeql-mcp --release
target/release/cubeql-code index crates/cubeql-code/tests/fixtures/semantic-demo \
  --embedding process --encoder-config /absolute/encoder.json \
  --output /tmp/code-resident.arrow --format json

target/release/cubeql-code search /tmp/code-resident.arrow \
  "Where are configuration settings saved?" \
  --encoder-config /absolute/encoder.json --format json

target/release/cubeql-mcp /tmp/code-resident.arrow \
  --encoder-config /absolute/encoder.json
```

Each CLI invocation owns a separate encoder lifetime. The MCP server retains one provider
and vector index across its client's `search_code` calls. In an MCP client configuration,
pass the snapshot path and `--encoder-config` as separate args; see
[the MCP guide](mcp-code-search.md). Perform `initialize`, `tools/list`, then search and
`read_code` using the returned snapshot/chunk handles. Stdout remains MCP JSON-RPC only.

The same profile also works with `update`, `evaluate` and `benchmark run`; new indexing and
benchmark runs require `--embedding process`. Saved snapshots select their own vector-space
contract, so querying/updating needs only `--encoder-config`. Existing imported snapshots
can use a matching live provider without document re-encoding. Omit it for lexical/symbol
CLI searches; MCP offers explicit `--text-only` for a completely model-free session.

`--encoder-config` conflicts with `--model-dir` and `--vectors`, and with MCP `--text-only`.
It never silently selects a different model. Model inference runs on the profile's device;
CubeCL vector scoring uses the binary's CPU/Metal/CUDA/HIP/WGPU features independently.
No `local-embeddings` Rust feature is required for the external process provider.

CLI search reports startup as `model_load_ms`, query inference as `embedding_ms`, and
upload/retrieval separately. MCP's ready trace adds `provider_startup_ms` and
`embedding_execution`; responses report `embedding_ms` and `retrieval_ms` per search.
Startup includes process launch, artifact verification and model loading, so it is not a
pure model-load timer. Repeated MCP searches exclude this startup work.

## Protocol and failure behavior

A strict JSONL ready frame declares protocol version 1, full embedding config and model load
time. Requests carry monotonic IDs, query/document role and up to 32 texts. Responses carry
matching IDs, vectors and encoding time. Prefixes are applied once by the encoder; raw input
text stays unprefixed. Frames are limited to 1 MiB, and vectors must have the expected row
count/dimensions, finite values and declared normalization.

Calls are serialized with bounded pending work. Startup and request deadlines include I/O;
a broken/oversized response, child exit, invalid contract or timeout fails explicitly. Fatal
failures poison that provider; create a new provider/server to retry. There is no automatic
hash fallback or hidden inference retry. Owned process groups are cleaned up on Unix; this
optional adapter currently targets POSIX while the generic retrieval library stays portable.

The bundled worker writes protocol only to stdout and bounded metadata/error diagnostics to
stderr, without query text or vector values. `--max-rss-mb N` optionally checks peak RSS before
and after inference, then fails closed if the sampled ceiling is exceeded. It cannot prevent
a transient allocation spike or replace an OS memory limit. Peak RSS is not current live RSS.
Repeated-query memory and startup/encoding/retrieval timings need separate measurements;
small fixtures do not establish production memory bounds or semantic quality.

Source updates change the offline encoder recipe identity. Incomplete runs from an older
recipe must use their captured original source to resume; completed benchmark evidence stays
unchanged. The live vector-space configuration retains the pinned preprocessing recipe.

## Validation

Run the model-free provider and protocol checks with:

```sh
cargo test -p cubeql-embeddings
python3 -B -m unittest discover -s tools/tests -v
```

The Rust process fixtures cover document batching (32+1 rows), query/document roles, input
limits, full config mismatch, malformed results, startup/request deadlines including blocked
stdin, stderr propagation, and child cleanup. Python fixtures cover prefix application,
stdout isolation, vector validation, bounded input reads and the sampled RSS ceiling.

A small local CodeRankEmbed check used the pinned CPU FP32 recipe with a 256-token limit:
one three-document request and 24 alternating query requests shared one worker/model load.
Vectors matched the existing offline fixture exactly (maximum absolute error 0); the two
query texts ranked `load_config` and `save_config` first respectively. Peak RSS was about
1.48 GiB at both the first and final requests, with 8.64 seconds total wall time including
startup. These are integration observations from a tiny fixture, not a throughput benchmark,
independent quality samples, or a production memory guarantee. No paid model calls were used.

The CLI/MCP layer was also checked with the real CodeRank model over a persisted Arrow
snapshot containing the three frozen offline document vectors. A real stdio client discovered
all three tools, then asked two new natural-language questions (loading settings from disk and
persisting them to the filesystem). Across eight alternating searches and eight source reads,
all top results matched the intended functions, and one worker loaded the model once. Sampled
peak RSS remained about 1.50 GiB. Warm query encoding was about 43–45 ms; first-query work was
higher. This tiny CPU fixture establishes wiring and reuse only, not repository recall,
production latency/memory bounds or voluntary use by a coding agent.
