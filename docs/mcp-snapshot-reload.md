# MCP snapshot reload contract

The MCP server's reload source is a fixed path selected by the operator at startup. A request
selects a previously configured reload source by its tool name and supplies an
`expected_snapshot_id`; requests never carry a filesystem path. The server reads and validates
the candidate before changing the active snapshot.

Preflight opens one regular file, checks its reported size, reads at most the configured byte
limit plus one byte, and rejects growth beyond that limit. MCP uses a fixed 512 MiB replacement limit; library preflight callers supply their own bound. It
hashes the bounded byte buffer with BLAKE3 and parses that same buffer as CubeQL IPC, producing
an identity of the form `blake3:<hex>`. This ties the identity to the exact bytes parsed. The
input-byte limit does not bound decoded memory use or parsing time.

When semantic search is configured, a candidate must include a provider-aware embedding
configuration exactly compatible with the active provider, including dimensions, model and
revision, normalization, implementation, pooling, token limit, prefixes, and artifact hashes.
Text-only operation can accept valid legacy snapshots without a provider contract.

The candidate remains immutable during preflight. During a switch, the old and candidate
snapshots may coexist in memory while the candidate's text index is prepared and warmed and its
resident vector index is uploaded. The embedding model/provider is reused; reload does not
create or download another model. Reload work shares the serialized worker queue, so a large
candidate can temporarily delay searches and checks. A failed preflight or preparation leaves
the old snapshot active. A successful switch changes the current snapshot identity; requests
carrying an older ID fail with `snapshot_mismatch`.

## Opt-in session switching

```sh
cubeql-mcp initial.arrow --reload-index reload.arrow --encoder-config encoder.json
```

Without `--reload-index`, the catalog and callable router retain the original three tools.
The target may be absent at startup; it is opened only on an explicit `reload_snapshot` call.
The operator grants read authority to this fixed path. Do not configure it inside a directory
that an untrusted participant can replace. The tool accepts no path, encoder configuration,
workspace root, or strategy change. Text-only sessions need no encoder configuration.

Build a new snapshot with the existing `cubeql-code update` command and publish the completed
file using a same-filesystem atomic rename to the configured reload path. Never write an IPC
file in place while it may be read. Snapshot identity binds the exact parsed bytes, but a
loader cannot make an in-place concurrent writer produce a coherent publication.

```json
{"name":"reload_snapshot","arguments":{"expected_snapshot_id":"blake3:<current 64-digit hash>"}}
```

Obtain the expected ID from a recent `search_code` response. The compare-and-swap check occurs
when the serialized worker executes the request, before opening the target. A stale expected
ID returns `snapshot_mismatch`. A queued request can therefore become stale before execution.
Malformed IDs or unknown fields fail validation. Reload has readOnlyHint=false and
idempotentHint=false because it changes session state and the configured target can change.
It never writes repository files, rebuilds an index, or runs document encoding.

A successful response has format_version=1, source=snapshot_reload, previous_snapshot_id,
snapshot_id, changed, chunks and provider_reused. Identical bytes return changed=false and
preserve all handles. `provider_reused` is true only when an actual semantic provider remains
loaded; text-only sessions report false. Changed snapshots rebuild device vectors and host
text/symbol tables. The model is not restarted, and the launch profile is not reread.

All fallible file/format/embedding/device preparation completes while the old state remains
owned. The worker then releases old adapters, replaces the owning snapshot, builds the new
infallible host tables, and acknowledges the switch. Ordinary errors keep the old snapshot;
process aborts, allocation failures and panics are not recoverable rollback guarantees.
No request can interleave with preparation or observe a partially switched generation.
Old snapshots are dropped, not retained as a historical cache. A content ID can become valid
again if the operator deliberately reloads exactly the same earlier bytes.

`preflight_ms` covers bounded input reading, hashing, parsing and compatibility; `prepare_ms`
covers candidate resident-index construction; `text_index_ms` covers new host-table warmup.
`reload_ms` covers worker execution through the completed switch, including old-state release,
without queue wait or client transport. No-op replies have zero prepare/text-index times.
Existing request_finished traces retain queue_ms and elapsed_ms and add reload outcome IDs.
There is no hard parsing/upload timeout, memory cap or nonblocking-search promise. Up to eight
jobs can queue behind the active request; excess requests still return busy. Cancelled queued
calls are skipped, but an admitted reload may complete after the caller stops waiting.
After uncertain completion, search for the active snapshot ID before retrying.

Source `read_code` and `check_code` require the new snapshot_id after changed success.
Search remains snapshot-local; freshness checks retain the original explicitly configured
workspace capability. Reload never adopts the candidate's saved workspace root or changes a
configured strategy restriction. Updating source files alone does not refresh the active session.

Library callers constructing `ServerOptions` now set `reload_index: None` for the existing
behavior or `Some(path)` to opt in. No snapshot persistence or search/read response version is
changed. This feature changes lifecycle behavior, not ranking policy or relevance judgments.

## Run local incremental-update acceptance

`tools/mcp_reload_smoke.py` runs a create-new two-file fixture through index, update,
MCP discovery, failed reload retention, explicit changed reload, stale-handle rejection,
source freshness, and a no-op reload. It accepts a trusted local encoder profile or uses
hash fixtures. See the [measured CodeRank lifecycle report](../benchmarks/retrieval/mcp-reload-2026-11-20-700cdc4/README.md)
for exact commands, independent encoder PID/load receipts, timing and memory limits.
This is separate from retrieval-quality validation and the deferred human-review backlog.

Reload does not restart an unhealthy encoder or change its configuration; recover a failed
provider by restarting the server. The replacement byte limit does not change initial-load behavior.
