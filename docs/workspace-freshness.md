# Workspace freshness for saved code hits

An agent can search a valid saved snapshot while the live repository has already changed.
The code chunks and vectors still agree with each other, but saved line numbers may no longer
describe the current checkout. This Milestone 8 follow-up adds a conservative, read-only guard:
`check_code`. It does not rebuild the index or turn snapshot retrieval into live search.

## Example

Build or update an index with the current `cubeql-code`, then grant the MCP server an explicit
workspace directory. For a model-free example:

```bash
cargo run -p cubeql-code -- index ../cubecl --output /tmp/cubecl-freshness.arrow
cargo build -p cubeql-mcp --release
```

Configure your MCP client to launch the built binary with arguments such as:

```json
[
  "/tmp/cubecl-freshness.arrow",
  "--text-only",
  "--workspace-root", "/absolute/path/to/cubecl"
]
```

Search with `strategy: "lexical"` or `"symbol"` in text-only mode. For semantic/hybrid, retain
the existing model/backend arguments and omit `--text-only`. No client configuration is changed
automatically, and no agent process or hosted-model run is started by this feature.

After `search_code` returns a hit, call:

```json
{
  "snapshot_id": "blake3:<copy the exact search response value>",
  "chunk_id": 17
}
```

`check_code` returns the snapshot ID and saved chunk location, with fields like:

```json
{
  "format_version": 1,
  "source": "workspace_check",
  "status": "unchanged",
  "reason": "file_hash_match",
  "expected_file_hash": "<saved whole-file BLAKE3>",
  "actual_file_hash": "<observed whole-file BLAKE3>",
  "checked_bytes": 1234
}
```

The hashes/count above are illustrative, not measured results. Chunk-level `content_hash` in
the accompanying location remains the hash of the saved chunk; it is not the whole-file hash.

| Status | Meaning | Suggested caller behavior |
| --- | --- | --- |
| `unchanged` | Observed complete file bytes matched the saved file hash | Saved source/locations have point-in-time content evidence; still verify before editing |
| `modified` | Completed hash comparison found different file bytes | Do not apply saved line numbers blindly; read live code or build a new snapshot |
| `missing` | The indexed file could not be found under the configured root | Treat the saved location as stale; investigate a move/deletion |
| `unverifiable` | No trustworthy comparison could be completed | Do not treat absence of evidence as unchanged |

For example, a snapshot contains `load_config` on line 20. Adding a comment at the top of its
file leaves the function's own chunk content hash unchanged but shifts its location. Checking
the complete file detects this as `modified`. Editing only another function in the same file
also returns `modified`: this is intentionally conservative, not a per-function relocation tool.
Restoring exactly the original file bytes returns `unchanged` even if timestamps changed.

`read_code` always returns the original saved source, including after a check reports modified
or missing. Checks do not refresh vectors, rewrite source or update handles. Restart the MCP
server after separately building an updated snapshot if fresh retrieval is required.

## File manifest and compatibility

New code indexes store `cubeql.code.file_hashes` as JSON Arrow schema metadata: a mapping of
repository-relative file paths to lowercase BLAKE3 hashes. Each hash covers the exact UTF-8
file bytes that the builder used for chunk extraction, including BOMs, line endings, comments
and unchunked gaps. It is computed from that already-read source, not from a second file read.
Accepted empty files are recorded; ignored, unsupported, binary and oversized files are not.

This metadata is optional and additive to code snapshot format 1; existing columns and model
contracts are unchanged. The reader validates manifest shape, hashes and coverage for chunk
paths when it is present. IPC roundtrips and FP16/FP32 conversions preserve it. The library
exposes `CodeIndex::file_hash(path)` for other adapters without introducing filesystem checks
into the generic Arrow/vector engine.

Older snapshots without the manifest remain searchable/readable, but return `unverifiable`
with reason `legacy_snapshot` when a workspace is configured. Incremental `update` reads current
files and adds hashes without re-embedding unchanged chunks. A line-only or comment-only edit
can therefore update file evidence and line numbers while retaining existing vector values.
After updating, restart the server with the new snapshot; it has a different snapshot ID.

File hashes and snapshot IDs are content identifiers, not signatures or provenance attestation.
Use snapshots from a trusted indexer. A deliberately forged manifest is not made authentic by
having a syntactically valid digest, and a matching hash says nothing about code safety.

## Filesystem authority and failure behavior

The operator must explicitly provide `--workspace-root`. The saved `cubeql.code.root` is never
used as implicit filesystem authority. This allows a relocated checkout, but verifying one
hit does not prove the rest of that checkout matches the original repository.

Without the flag, `check_code` returns `unverifiable` / `workspace_not_configured` without
opening any workspace source. With it, the worker opens a directory capability at startup.
File checks resolve only the saved path associated with the supplied snapshot/chunk handle:
the tool accepts no arbitrary file path. [cap-std](https://github.com/bytecodealliance/cap-std/blob/main/cap-std/README.md)
bounds traversal and symlink resolution to that directory. Ambiguous paths containing parent
components, absolute roots, backslashes or drive-like colons are rejected before opening. Such
unusual paths can still be searched in a snapshot, but cannot be verified by this adapter.

External symlinks and inaccessible targets return `unverifiable`, never a hash of outside data.
Checks require regular files. Unix opens use nonblocking mode so replacing a saved source path
with a FIFO cannot block the worker while opening it. Directories and other non-regular targets
are not hashed. Only read operations are performed; no workspace content is returned or written.

Each check hashes at most 16 MiB using a fixed 64 KiB buffer, with a bounded read that detects
growth beyond that cap. Larger files return `unverifiable` / `size_limit`; this is independent
of the index builder's configurable size limit. Metadata/read failures likewise never yield
`unchanged`. Before/after handle size and modification-time checks catch some concurrent writes.
No result is cached between checks. Checks share the retrieval worker and its bounded queue,
so their I/O time can delay later searches; no hard wall-clock timeout or memory-budget claim
is made for slow/network filesystems.

This is **not an atomic edit precondition**: there is no file lock, compare-and-swap write,
watcher, transactional filesystem view or protection against arbitrary concurrent replacement.
A file can change while or immediately after it is checked. Do not use `unchanged` as a promise
that future edits are safe, or as a replacement for the agent's own current-source verification.

Unknown chunk IDs, mismatched snapshot IDs and malformed arguments remain tool/protocol errors.
An ordinary inability to verify a valid handle is a successful tool response with the explicit
`unverifiable` status and machine-readable reason. Stderr traces record the checked chunk ID,
status and elapsed time, not live source or query text.

## Verification and remaining Milestone 8 work

Tests cover whole-file versus chunk hashes, exact UTF-8/CRLF/BOM bytes, IPC/precision roundtrips,
legacy loading/updating, malformed manifests, comments outside a chunk, unrelated function edits,
restoration, file deletion, relocated roots, absent authority, old handles, byte caps, directory
targets, unsafe paths, external symlinks and substituted FIFOs. MCP cases launch the actual binary.

```bash
cargo test -p cubeql-code --test file_manifest
cargo test -p cubeql-mcp
cargo test --workspace
cargo test --workspace --no-default-features --features metal
```

Run feature variants sequentially when they share a Cargo target directory. The stdio tests
launch `target/debug/cubeql-mcp`; a concurrent different-feature build can replace that binary
before a later test spawns it. Use separate target directories if running such variants in parallel.

This improves the interface an agent can use before acting on saved results. It does not configure
a real harness, measure what the model consumed, or demonstrate end-to-end task improvement.
Those Milestone 8 acceptance criteria remain open; ANN/scale work remains measurement-driven.
