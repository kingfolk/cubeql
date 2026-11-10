# Offline Semble result import

`tools/repository_peer.py` imports supplied Semble JSON result payloads into the frozen repository evaluator. It is an offline adapter: it never launches Semble, starts a model, downloads files, or produces a native peer result. The input manifests and payloads should come from an independently recorded native execution. Handwritten fixtures are useful for checking the importer but are not evidence of Semble quality.

```sh
python3 tools/repository_peer.py \
  --plan plan.json \
  --manifest peer-manifest.json \
  --baseline repository-baseline.json \
  --output peer-import-001
```

Each output path must be new. The importer writes `run.json`, `report.json`, `provenance.json`, exact input bytes as `manifest.json`, a canonical snapshot as `plan.json`, and byte-for-byte copies of any supplied payloads under `evidence/`. It will not overwrite inputs or an existing output directory. Preserve the original manifest and payloads alongside the import for audit.

## Manifest format 1

The manifest is strict JSON and has exactly these fields:

```json
{
  "format_version": 1,
  "plan_fingerprint": "fingerprint from the frozen plan",
  "name": "peer run name",
  "tool": {"name": "semble", "revision": "source revision"},
  "configuration": {"snapshot": "declared metadata"},
  "candidate_k": 100,
  "results": [
    {
      "query_id": "query id from the plan",
      "query_text": "exact frozen query text",
      "applied_filter": {"path_prefix": "src/"},
      "status": "ok",
      "payload": "payloads/000001.json"
    }
  ]
}
```

`candidate_k` must be at least 10. `results` must cover every frozen query exactly once; failures, timeouts, and unsupported queries remain in that denominator. Each record has exactly `query_id`, `query_text`, `applied_filter`, and `status`, plus optional `payload` and `error`. `query_text` must match the frozen query, and `applied_filter` must equal its complete filter object: `{}` when there is no filter, or `{"path_prefix":"src/"}` when one is declared. Scalar strings and `null` are invalid. `ok` requires a payload and forbids `error`. Other statuses are `failed`, `timeout`, or `unsupported` and require a nonempty error. Their optional payload is copied as unparsed diagnostic evidence.

Payload paths are normalized relative POSIX paths resolved from the manifest's directory. Absolute paths, `..`, symlinks, missing files, and non-regular files are rejected. Raw payload bytes and SHA-256 digests are retained.

## Accepted native payloads

The pinned native response shape is `{ "query": string, "results": [...] }`, optionally with an empty `repos` object. Every result has `file_path`, one-based inclusive `start_line` and `end_line`, and finite numeric `score`; `content` may be present as a string. The query echo must match the frozen query. The sole accepted successful empty response is exactly `{ "error": "No results found." }`. Other error payloads fail import.

File paths must already be normalized repository-relative POSIX paths and are checked against the frozen file inventory, ranges, and query path-prefix by `repository_eval.score`. A corpus may legitimately contain a directory named `repo`; its paths are preserved as written. The importer never strips prefixes or guesses paths, and it does not reorder, truncate, deduplicate, or post-filter rankings. Native scores and content remain in the copied raw payload; evaluation uses only path and inclusive span. A nonempty filter is only declared and matched in the manifest. This check cannot attest that Semble enforced the filter while retrieving results. Stock CLI behavior that cannot apply the filter before top-k should be recorded as `unsupported`, or the peer adapter must enforce it natively.

`configuration` is descriptive metadata. The adapter cannot independently attest snapshot/source identity, model, precision, budgets, or filter enforcement. Keep those limits explicit in analysis. `--baseline` can be repeated; validated baselines appear before the imported run in the paired report. The importer does not tune against baseline outcomes.

The accepted location schema follows the pinned Semble source
[`format_results`](https://github.com/MinishLab/semble/blob/4633ee75a1d8c9ce3998b610a14fc80e7fb7c802/src/semble/utils.py)
and its [inclusive chunk line calculation](https://github.com/MinishLab/semble/blob/4633ee75a1d8c9ce3998b610a14fc80e7fb7c802/src/semble/chunking/chunking.py).
A newer or different native schema needs a separately reviewed adapter condition.
