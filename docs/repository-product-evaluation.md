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
Version 1 supports **one repository per plan**; version 2 freezes multiple repositories
under one query specification. Keep query labels and output files outside
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

The `overall` and category aggregates are unweighted query means. Version 2 also reports
`repository_macro`, an equal-weight mean of the per-repository query means (see below).
This is anchor recall, not nDCG, coding-task success,
or the percentage of useful source lines returned. Large native chunks can cover more anchors
and cost more context: span scoring alone does not establish fair context budgets. Preserve
source sizes and compare common-unit conditions before attributing quality gains to ranking.
Timing metadata is retained, not normalized or interpreted as a resident-service benchmark.

This infrastructure does not complete RQ2: independently reviewed repository validation labels,
strong-model runs, independent peer rankings, failure analysis and separate engine controls
remain necessary. Keep the preserved CubeCL development labels unchanged and freeze a separate
validation set before tuning. No new default or 80% validation-target claim follows from this fixture.

## Freeze several repositories in a version 2 plan

Use a spec with `format_version: 2` and a nonempty `repositories` list of distinct IDs and
declared revisions. Each query names exactly one of these IDs; each declared repository must
have at least one query. Query IDs remain globally unique. Anchor IDs are local to a query.
Paths are local to that query's repository: two repositories may both contain `src/lib.rs`
without sharing labels or file identities. A query cannot span repositories; a cross-file
query still collects evidence from files within its named repository.

Pass a JSON root map with exactly one source directory per declared repository:

```json
{"alpha": "/source/alpha", "beta": "/source/beta"}
```

```sh
python3 tools/repository_eval.py freeze \
  --spec benchmarks/retrieval/fixtures/multi-repository/spec.json \
  --roots benchmarks/retrieval/fixtures/multi-repository/roots.json \
  --output /tmp/multi-repository-plan.json
```

The checked-in map resolves relative to the **process working directory**, so run that
example from the repository root. Real experiments can use absolute roots. Source directories
are not saved in the portable plan. Identical resolved roots (including aliases through a
symlink) are rejected. Missing/extra root IDs, unknown repository references, repositories
without queries, or duplicate repository IDs fail before a plan is written. Inventory and
anchor validation use each repository's own bytes and line counts. Freeze immutable source
checkouts without concurrent writers; several inventories are sequential observations,
not an atomic cross-repository filesystem snapshot.

Use `--root` for a v1 spec, or `--roots` for a v2 spec; these options are mutually exclusive.
Version 2 permits a single repository too, using a one-entry map. Existing v1 plans, digests,
reports and commands retain their original behavior. Plan and spec versions must agree.

The native run interchange remains `format_version: 1`, including for a v2 plan. Each run
must provide exactly one result for every query across **all** repositories. Its globally
unique `query_id` supplies the repository context, so hit paths have no separate repository
field. Never pool hits from other repositories into that query's ranking. Record any
repository-specific snapshots/model settings under the run's free-form `configuration`.
All runs being compared must bind the same whole-plan fingerprint; results from independently
frozen plans cannot be concatenated and scored as a paired trial.

Scoring commands are unchanged. For v2 plans, the report's `format_version` is 2 and each run
contains both:

| Report field | Averaging rule |
| --- | --- |
| `overall` | Every query has equal weight across the entire plan |
| `by_repository` | Query mean within each repository, with its counts and failure coverage |
| `repository_macro` | Every repository has equal weight, using its query mean |

For example, if alpha has three queries with Recall@10 = 1 and beta has one with Recall@10 = 0,
`overall.recall_at_10` is **0.75**, while `repository_macro.recall_at_10` is **0.50**.
The latter also averages coverage equally by repository; raw status counts remain in `overall`
and `by_repository`. Failures still score zero, and unavailable Recall@50/100 remains null.
Freeze the repository/query mixture and primary averaging rule before interpreting validation
targets. Neither averaging rule establishes repository independence or absence of training exposure.

Each v2 paired comparison adds `by_repository`, containing query wins/ties/losses and mean
metric changes within each repository, plus `repository_macro_delta` for equal-repository
changes. Existing global query changes remain in `macro_delta`. Paired MRR deltas are null
at every level when candidate windows differ. This is descriptive paired arithmetic, without
a significance test or confidence-interval claim.

## Collect native CubeQL results automatically

The [repository collector](repository-collection.md) builds new snapshots from frozen source
roots, executes the same queries and filters, retains native responses and failed queries,
and writes scoring runs directly. Fresh CLI invocation timings are explicitly separated
from resident-service measurements.

## Import independently captured peer output

The [offline Semble adapter](repository-peer-import.md) validates native result JSON against
the same frozen plan and retains original payloads, query/filter declarations and provenance.
It can produce paired reports with collected CubeQL runs. Importer fixtures are format checks;
actual native peer execution and source/model/budget attestation remain separate work.

## Collect through a resident MCP session

[Persistent MCP collection](repository-mcp-collection.md) builds fresh snapshots and reuses
one server/provider session per repository. It retains protocol discovery, search and bounded
source-read evidence, with startup and resident-request timing recorded separately.
