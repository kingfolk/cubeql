# Persistent MCP repository collection

`tools/repository_mcp.py` evaluates the local `cubeql-mcp` server against a frozen RQ2 plan.
It verifies the plan and exact source inventory, builds a fresh Arrow snapshot for each repository,
and keeps one server process alive for every query and strategy in that repository. It does not
reuse a snapshot from outside the output directory. The collector requires no Python packages
beyond the standard library.

Example:

```bash
python tools/repository_mcp.py \
  --plan plan.json --roots roots.json \
  --binary /absolute/path/to/cubeql-code \
  --mcp-binary /absolute/path/to/cubeql-mcp \
  --output /tmp/mcp-run --strategy lexical --strategy symbol \
  --top-k 10 --candidate-k 100 --read-budget-bytes 262144
```

`roots.json` maps every frozen repository ID to its root path. The output path must not exist and
must be outside all frozen source roots. Semantic/hybrid collection starts the MCP server with the
same selected local encoder profile used to build the snapshot. Hash embeddings need no profile.
The server is initialized with MCP protocol version `2025-11-25`; both `search_code` and `read_code`
must be advertised before queries run.

The search request carries the frozen query text, selected strategy, `top_k`, `candidate_k`, and
the exact declared `path_prefix` when one exists. It does not add a language filter. Each search
response must identify a saved snapshot, echo the strategy and candidate budget, fit the requested
top-k, and provide finite native `embedding_ms` and `retrieval_ms`. The returned locations are
validated through the common RQ2 scorer for source inventory, ranges, filters, and duplicate spans.
For each ranked handle, the collector reads saved source pages until the per-query byte budget is
spent. Read evidence is checked against the corresponding source span and recorded with byte
counts and hashes. `read_truncated` marks queries whose page stream could not be fully read inside
the budget. A covered location is an evaluation result; it does not prove that the full source was
returned to a consumer.

## Timing and evidence

`elapsed_ms` on each standard run result measures the resident `tools/call(search_code)` wall time,
including local JSON-RPC framing and decoding. Its boundary is
`resident request wall ms; excludes indexing/startup and source reads`. Indexing and MCP startup
are recorded separately. `embedding_ms` and `retrieval_ms` are the server's native phase timings;
source reads have separate byte counts and are not added to search latency.

The manifest stores per-session request IDs/methods and timings, initialization/tool-list data,
query phase timings, read evidence, index executions and snapshot hashes. Byte-identical client
stdin, server stdout and server stderr are retained as `mcp.N.stdin`, `mcp.N.stdout`, and
`mcp.N.stderr`. Each JSONL frame is limited to 2 MiB, aggregate session stdout/stderr to 16 MiB,
and request deadlines include nonblocking writes and skipped server notifications. A timed-out or
malformed protocol session is poisoned and is not implicitly restarted. Shutdown closes stdin,
waits briefly, then applies the existing collector's private-session process cleanup if needed.

The collector rechecks source inventories and fingerprints for the executable/model profile before
writing successful runs and the report. Failures remain in the complete frozen query denominator.
This is a native retrieval comparison, not a controlled encoder-only attribution or an agent-task
improvement claim. The report measures whether ranked locations cover frozen anchors; the source
pages are retained as evidence of what `read_code` returned.

Offline transport tests:

```bash
python -m unittest tools/tests/test_repository_mcp.py
```
