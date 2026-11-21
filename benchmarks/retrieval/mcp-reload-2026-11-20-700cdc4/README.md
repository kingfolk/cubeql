# Incremental update and MCP snapshot reload acceptance

Two explicit local lifecycle runs passed: a deterministic hash-vector fixture and a real
CodeRankEmbed-256 FP32 CPU condition. Both used two generated Rust files with one function
each. This is operational acceptance, not retrieval relevance, agent adoption, throughput,
large-repository latency, or an independently reviewed validation result.

The measured MCP implementation is `700cdc4`, with executable and producer SHA-256 receipts
checked before and after each run. The acceptance runner was added on top without Rust changes.
Hashes identify bytes, not a hermetic build. Native inputs and raw RPC traces are archived.

## Verified behavior

- Initial index encoded two chunks. Editing only the loader function and invoking the existing
  `cubeql-code update` encoded one chunk and reused the unchanged chunk.
- The persistent MCP session continued to return the original source until explicit reload.
  Freshness checks reported the edited file as modified before the switch.
- A corrupt replacement failed with reload_failed; the old snapshot remained searchable.
- A complete snapshot was atomically published at the operator-configured reload path. Explicit
  reload changed the snapshot ID, invalidated old read/check handles, returned the edited source,
  and made its current workspace hash check unchanged.
- Reloading identical bytes was a no-op. The same server process continued serving all calls.
- Three semantic requests before and three after reload exercised actual query encoding. Their
  success checks lifecycle availability, not whether the top-ranked result is a correct answer.

## Observed real-model resource receipts

| Observation | CodeRank local two-file run |
| --- | ---: |
| Initial index command wall time | 6,591 ms |
| Incremental update command wall time | 4,811 ms |
| MCP startup through initialize | 4,606.86 ms |
| Changed reload worker time | 0.862292 ms |
| Candidate read/hash/parse/compatibility | 0.694959 ms |
| Candidate resident-index preparation | 0.067500 ms |
| New host text/symbol tables | 0.071625 ms |
| MCP encoder ready/model-load events | 1 |
| MCP query-encoding events | 6 |
| Encoder self-reported peak RSS at first/last query | 1,618,919,424 bytes / 1,618,919,424 bytes |

The encoder wrapper independently records PID, parent PID, ready and result events. All six
MCP query events belong to one encoder PID with one ready event. The initial-index and update
CLI each launch a separate encoder: three model lifetimes occurred across the whole workflow.
The model identity, weights/config hashes, encoding implementation and preprocessing are in
archived input receipts and native index output. No model weights are bundled or downloaded.

Reload timing excludes queue wait and client transport, and does not include document encoding.
It is one tiny sample, not an SLA, P95, speedup or a full-corpus reload measurement. Command
wall times include their separate provider startups. The two MCP-parent RSS samples are
74,383,360 and 57,507,840 bytes; they are point observations, not peak RSS or a memory improvement.
Encoder peak RSS is a cumulative self-reported high-water mark. Neither observation measures
process-tree peak, transient old/new vector coexistence, or the overlapping update encoder.
No CPU/Metal or old/new controlled performance comparison was performed.

## Reproduce into a new directory

Build the binaries and run [the acceptance tool](../../../tools/mcp_reload_smoke.py):

```sh
cargo build --locked -p cubeql-code -p cubeql-mcp --bins
python3 -B tools/mcp_reload_smoke.py \
  --code-binary /absolute/path/to/cubeql-code \
  --mcp-binary /absolute/path/to/cubeql-mcp \
  --output /new/output/directory
```

Add `--encoder-config /absolute/path/to/local-encoder.json` to exercise a configured local
model. Hash mode uses fixtures only. The tool creates and edits only its two generated files,
retains command and protocol failures, and refuses an existing output directory. Commands and
RPCs default to a 180-second timeout, with the existing bounded collector transports. Failed
attempts have failure.json and no successful completion manifest. Runtime process cleanup
requires POSIX process visibility. Supplied encoder profiles must reference existing local files;
profile validation itself may reject unidentifiable arguments. Paid API/agent trials are outside
this tool's scope. A custom encoder can have arbitrary effects: use a trusted local profile.

The generic runner records provider_reused as a server declaration. Independent model-load/PID
verification in this report additionally used the archived instrumented serve_encoder wrapper.
The wrapper only observes local diagnostic events; it preserves embedding behavior. Archived
absolute paths are historical provenance, not portable commands.

## Verify saved evidence without models

```sh
python3 -B benchmarks/retrieval/mcp-reload-2026-11-20-700cdc4/verify.py
```

The standard-library verifier extracts nothing and executes no archived code. It checks exact
52-member inventories, hashes, both completion manifests, producer identity receipts, incremental
counts, raw JSON-RPC/structured-text agreement, failed and successful reload outcomes, old/new
source reads, stale handle errors, no-op behavior, one server ready event per run, and independent
encoder parent/PID/load observations. Small generated Arrow fixture snapshots are included;
model weights, executables, runtime caches and third-party source are excluded. Positive runtime
checks cannot prove the absence of every leak or allocation failure. Source/model compatibility,
serialized queue behavior and error cases are also covered by Rust tests described in
[the reload contract](../../../docs/mcp-snapshot-reload.md).
