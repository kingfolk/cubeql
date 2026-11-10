# Frozen repository product comparisons

`tools/repository_eval.py` provides the first RQ2 product-evaluation building block:
freeze source and reviewed queries, then score ordered source locations from independently
run search tools. It uses only the Python standard library. It does not run a model or agent,
change a ranking policy, generate relevance labels, or establish a peer-quality result.

## Freeze before running retrieval

```sh
python3 tools/repository_eval.py freeze \
  --spec benchmarks/retrieval/fixtures/product-comparison/spec.json \
  --root benchmarks/retrieval/fixtures/product-comparison/corpus \
  --output /tmp/product-plan.json
```

Outputs are create-new files. Choose a fresh path for each experiment. The checked-in example
is a synthetic development smoke fixture, not an independently reviewed validation set.
Version 1 supports **one repository per plan**. Keep query labels and output files outside
the source root. Freeze a deliberately prepared source-only checkout: the manifest inventories
regular files recursively, excluding `.git`, rather than applying CubeQL's parser/ignore rules.
Symlink directories are skipped and symlink files are rejected. Line numbers count LF-delimited
source lines (including a final line without LF); CRLF occupies one line separator.
Do not include build directories, downloaded weights, secrets, or model caches in that root.

The spec declares `format_version`, `name`, `split` (`development` or `validation`),
`repositories` (`id`, `revision`), and `queries`. Each query has a unique `id`, `repository`,
`category`, `text`, `filter`, and nonempty `anchors`. An anchor has a query-local unique `id`,
repository-relative POSIX `path`, and inclusive one-based `start_line` and `end_line`.
The supported filter is a literal, case-sensitive `path_prefix`; `{}` applies no filter.
The tool rejects unknown fields, invalid spans, path traversal and ambiguous identities.

Freeze records file SHA-256 hashes and line counts, the complete original spec, its canonical
JSON digest, and a plan fingerprint. The declared revision is caller-supplied provenance;
file hashes bind the bytes actually read, including local modifications. The fingerprint
detects accidental changes; it is not a signature or an independent review attestation.
Freeze an immutable checkout, with no concurrent file writers.

## Export native rankings

Each tool adapter writes this interchange format, with exactly one result for **every** frozen
query. Use the exact frozen query text, repository, and filter. Preserve the native ranking
order; do not reorder hits by their scores across different tools.

```json
{
  "format_version": 1,
  "plan_fingerprint": "<fingerprint from product-plan.json>",
  "name": "cubeql-symbol-smoke",
  "tool": {"name": "cubeql-code", "revision": "<evaluated source revision>"},
  "configuration": {
    "strategy": "symbol",
    "model": "hash-fixture-only",
    "timing_boundary": "fresh CLI process per query"
  },
  "candidate_k": 10,
  "results": [
    {
      "query_id": "config", "status": "ok", "elapsed_ms": 12.5,
      "hits": [{"path": "config.rs", "start_line": 1, "end_line": 3}]
    },
    {
      "query_id": "ranking", "status": "ok",
      "hits": [{"path": "search.rs", "start_line": 1, "end_line": 3}]
    },
    {
      "query_id": "both", "status": "unsupported",
      "error": "This condition supports exact symbols only", "hits": []
    }
  ]
}
```

For CubeQL, run `search SNAPSHOT TEXT --top-k K --candidate-k K --strategy STRATEGY
--include-source --format json`, passing any frozen path-prefix filter. Project each native
`hits` entry to its `path`, `start_line`, `end_line`; do not use snapshot-local row IDs as peer
identities. Build the snapshot from the frozen source and verify its saved source against that
checkout. For peer tools, use their own native ranked source spans and preserve raw output.
The scorer validates paths, spans, query coverage and filters, but cannot attest that an
adapter actually ran the declared tool, used the declared configuration, or indexed those bytes.

`candidate_k` is the exported ranking window (and MRR cutoff), not a claim that tools examined
the same number of internal candidates. Record per-branch budgets, chunking, deduplication,
precision, model/prompt identities, executable and snapshot hashes in `configuration` and retain
original artifacts. A native-pipeline comparison and a common-unit/model control are separate
conditions. The scorer does not perform automatic peer execution or import old evaluation reports.

Use `failed`, `timeout`, or `unsupported` with a nonempty `error` and empty `hits` when a tool
cannot answer. An `ok` result with no hits is a valid search miss. Missing/extra query IDs,
duplicate exact spans, invalid/out-of-corpus spans and incompatible plan fingerprints fail
validation rather than silently reducing the denominator. Partial/overlapping snippets remain
separate ranking entries and consume rank positions; no automatic overlap deduplication occurs.

## Score and compare

```sh
python3 tools/repository_eval.py score --plan /tmp/product-plan.json \
  --run /tmp/baseline.json --run /tmp/candidate.json --output /tmp/product-report.json
python3 -m unittest discover -s tools/tests -p test_repository_eval.py
```

The first run is the paired baseline. Reports retain all run inputs and their digests,
per-query anchor ranks, overall/category/repository macro metrics, status counts and paired
wins/ties/losses. Failures and unsupported queries contribute zero to quality metrics over
the **entire frozen query set**, alongside their explicit coverage counts.

An anchor is found only when a single returned hit contains its entire labeled span on the
same path. Partial overlaps and separate snippets covering different parts do not count.
Choose short, manually reviewed evidence spans; a whole-file label would make a valid small
implementation snippet count as a miss. Several anchors may be satisfied by one hit, and
several hits for the same anchor earn no additional recall credit.

- `recall_at_10` is the fraction of required anchors covered by the first ten hits.
- `all_required_at_10` is one only if every required anchor is covered there.
- `recall_at_50`/`recall_at_100` are null when the declared window is below that cutoff.
- Reciprocal rank is based on the first hit covering any anchor, bounded by `candidate_k`.
  Compare MRR only across equal windows. Paired Recall@10 is comparable across exported
  windows of at least ten; the configuration still needs a fair budget interpretation.

Aggregates are unweighted query means. This is anchor recall, not nDCG, coding-task success,
or the percentage of useful source lines returned. Large native chunks can cover more anchors
and cost more context: span scoring alone does not establish fair context budgets. Preserve
source sizes and compare common-unit conditions before attributing quality gains to ranking.
Timing metadata is retained, not normalized or interpreted as a resident-service benchmark.

This infrastructure does not complete RQ2: independently reviewed repository validation labels,
strong-model runs, independent peer rankings, failure analysis and separate engine controls
remain necessary. Keep the preserved CubeCL development labels unchanged and freeze a separate
validation set before tuning. No new default or 80% validation-target claim follows from this fixture.
