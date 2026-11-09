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
