# Native repository baseline recipes

`tools/repository_baseline.py` freezes and executes named embedding conditions using the
existing `cubeql-code index` and `evaluate` commands. It uses the collector's bounded process
runner and source hashing. Python's standard library is sufficient; model inference remains
in the explicitly selected provider. This workflow does not download models or run an agent.

Unlike the source-span interchange, this workflow accepts the original native `(path, symbol)`
query labels, including language filters. In particular, the preserved
[64-query CubeCL development set](../benchmarks/retrieval/cubecl-2026-10-29-77b2d6af8514/queries.json)
is passed to Rust byte-for-byte. It is not converted into a new validation set or retuned.
Rust checks label resolution against each newly built snapshot before query inference.

## Prepare without inference

Write a JSON configuration like this, replacing the absolute paths:

```json
{
  "format_version": 1,
  "name": "cubecl-development-baseline",
  "binary": "/checkout/cubeql/target/release/cubeql-code",
  "root": "/frozen/cubecl",
  "queries": "/checkout/cubeql/benchmarks/retrieval/cubecl-2026-10-29-77b2d6af8514/queries.json",
  "profiles": [
    {"name": "minilm", "embedding": "minilm", "model_dir": "/models/minilm"},
    {"name": "code-model", "embedding": "process", "encoder_config": "/profiles/coderank.json"}
  ],
  "candidate_k": 100,
  "repeats": 3,
  "compare_fp16": false,
  "timeout_seconds": 3600,
  "fallback_lines": 80,
  "max_file_bytes": 2097152
}
```

```sh
python3 -B tools/repository_baseline.py prepare \
  --config /tmp/baseline-config.json --output /tmp/baseline-recipe-001
```

Paths in the configuration resolve relative to the configuration file's directory. Provider
launch-profile contents are not rewritten; use absolute executable, script and model paths
inside encoder profiles too. A MiniLM binary needs `local-embeddings`; an external process
provider uses the [resident encoder contract](resident-encoder.md). Use a hash profile
`{"name":"offline-smoke","embedding":"hash"}` for a pipeline fixture only.

Configuration fields are strict. The required fields are shown above through `profiles`;
defaults for the remaining fields are respectively 100, 3, false, 600, 80, and 2097152.
Profile names must be distinct. Hash has no provider arguments, MiniLM requires only
`model_dir`, and process requires only `encoder_config`. No API-backed paid condition is added.

Preparation runs no executable or model. It records source-file hashes and line counts,
the executable hash, exact label bytes, profile artifacts and orchestration source hashes.
MiniLM directories and absolute directory arguments in process profiles are inventoried;
absolute file arguments and the interpreter are hashed. Launch paths preserve a virtual
environment's interpreter symlink. Transitive imports, installed packages, environment
variables and the complete execution environment are not hermetically captured.

The create-new recipe contains `recipe.json` with a canonical fingerprint and an exact
`queries.json` copy. The copy is the execution input; moving or modifying the original label
file does not rewrite the recipe. Keep source roots immutable and outputs outside source and frozen provider directories.
Recipe files contain local paths and should normally stay outside Git. A source revision
alone does not attest an existing binary's build; the executable hash identifies its bytes.

## Execute explicitly

```sh
python3 -B tools/repository_baseline.py run \
  --recipe /tmp/baseline-recipe-001 --output /tmp/baseline-run-001 --execute
```

`--execute` is required. For each profile, the runner creates a new snapshot once and invokes
the native evaluator once. The evaluator retains one query provider for its query loop and
reuses the encoded query vectors across its strategy/precision comparisons. Each profile is
a separate condition and provider lifetime. The same fixed candidate budget, repeats,
chunking options, labels and source bytes apply to every profile. Choose the model/token
limit and controls before inspecting quality outcomes.

The runner verifies frozen inputs before execution and after each profile. Changed source,
binary, profile artifacts, copied labels or orchestration code invalidate the run; partial
diagnostics remain, without a completed summary. Hash checks are observations around
execution, not an atomic filesystem snapshot or proof that no transient edit occurred.

The output contains:

- Original index/evaluate stdout and stderr, including complete native per-query reports.
- Per-condition status records, execution commands, wall times and snapshot fingerprints.
- `summary.json` with every expected condition, successful count, all native strategy and
  precision summaries, source/candidate context in the copied recipe, and runtime metadata.
- `manifest.json` with byte hashes of the retained output files (excluding itself).

Index failures and timeouts remain explicit conditions, and later conditions can still run.
A malformed native report, changed query echo, denominator or strategy coverage becomes a
failed condition. Failed conditions have no valid quality score; there is no average across
conditions that could silently exclude them. Retry into a fresh output directory after
examining the retained failure. Nothing is automatically restarted or overwritten.

## Metrics and timing

The six native strategies are semantic, literal, experimental RRF, lexical, symbol and hybrid.
The summary preserves the native evaluator's Recall@5/10 and MRR@candidate_k; it does not
invent Recall@50/100 from the ten display hits, nDCG, alternative-answer judgments, or coding
task success. Native source-span reports and these symbol-label reports remain distinct.
See [the original evaluation guide](code-search-evaluation.md) for metric denominators,
precision controls and synthetic-fixture limits.

Raw native timers separate provider startup, query encoding and retrieval phases. First
query invocations and warm repeats retain the existing evaluator's interpretation; deferred
device work can cross host setup timers. CLI wall time includes process startup, indexing
or evaluation setup, and reporting. This is not an isolated kernel benchmark or a claim
about end-to-end resident MCP service latency.

For the original CubeCL development run, use the recorded corpus revision and retain its
MiniLM results. Strong-model runs, independently reviewed validation labels and actual native
peer comparisons remain separate measurements. A successful hash fixture establishes only
that this workflow runs. No RQ2 quality target or agent benefit follows from it.
