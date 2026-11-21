# MCP snapshot reload contract

The MCP server's reload source is a fixed path selected by the operator at startup. A request
selects a previously configured reload source by its tool name and supplies an
`expected_snapshot_id`; requests never carry a filesystem path. The server reads and validates
the candidate before changing the active snapshot.

Preflight opens one regular file, checks its reported size, reads at most the configured byte
limit plus one byte, and rejects growth beyond that limit. The default limit is 512 MiB. It
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
