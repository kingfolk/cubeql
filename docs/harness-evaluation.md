# Coding-agent retrieval evaluation runner

This Milestone 8 follow-up provides a real **Codex CLI adapter** and an offline-testable runner.
It compares ordinary agent search with the same agent given CubeQL lexical or hybrid tools.
It does not implement a new agent, judge free-form explanations with another model, or claim
that an offline protocol fixture establishes real agent quality.

The runner is `scripts/harness_eval.py`, using Python 3.12+ and only the standard library. The
Arrow/vector engine stays Rust and has no agent-specific dependency. The MCP server adds an
optional `--search-strategy` guard to make experiment arms enforceable rather than prompt-only.

## Three arms

| Arm | Ordinary shell/search/read | CubeQL MCP | Server policy |
| --- | --- | --- | --- |
| baseline | available | absent | none |
| lexical | available | search/read/check | lexical only, text-only initialization |
| hybrid | available | search/read/check | hybrid only, embedding provider + device |

The model may choose whether to call CubeQL. A configured but unused server is not counted as
retrieval adoption. The guard rejects a mismatching strategy with `strategy_restricted`; it does
not silently rewrite the query. The same prompt template, model ID, reasoning effort, source,
schema and wall-clock timeout apply to each arm, except for the available-tool description.
Arm order rotates across tasks/repeats; this is not randomized causal inference.

## 1. Prepare without language-model requests

Build binaries once and do not rebuild the MCP executable while an experiment is running:

```bash
cargo build -p cubeql-code -p cubeql-mcp --features local-embeddings
```

For real repositories, prefer a release build and the corresponding `target/release/` paths:

```bash
cargo build --release -p cubeql-code -p cubeql-mcp --features local-embeddings
```

Debug MiniLM encoding can exceed preparation's 600-second subprocess timeout. This is
a local indexing failure, not a model request. Keep failed output and choose a new
experiment directory rather than overwriting it.

Use an existing pinned MiniLM model directory (download separately with `cubeql-code model
download` if needed). Choose the Codex model explicitly; the runner never selects an unspecified
latest/default model. Replace `YOUR_MODEL_ID` below with a model your authenticated CLI supports.

```bash
python3 scripts/harness_eval.py prepare \
  --repo /absolute/path/to/cubeql \
  --revision cd98d53e61689081891192ba8c767522faca8c50 \
  --tasks evals/code-discovery.json \
  --output /tmp/cubeql-agent-experiment-v1 \
  --cubeql-code /absolute/path/to/cubeql/target/debug/cubeql-code \
  --cubeql-mcp /absolute/path/to/cubeql/target/debug/cubeql-mcp \
  --model-dir /absolute/path/to/pinned-minilm \
  --model YOUR_MODEL_ID --reasoning-effort medium \
  --repeats 2 --timeout 300
```

Preparation:

1. Resolves the revision to a commit and exports committed source with `git archive`; dirty and
   untracked changes never enter the corpus. Git export attributes can affect the exported tree.
2. Excludes `.git`, `.codex`, `.agents` and `evals` directory components, plus the supplied task
   file if it is tracked under the source repository. Symlinks/special files are rejected. Source
   size is capped at 256 MiB after obtaining the archive; archive generation itself is not streamed.
3. Builds one provider-aware Arrow snapshot from that source, recording indexing time separately.
   MiniLM encoding runs locally and can take time; preparation never starts a Codex/model request.
4. Resolves every annotated path/qualified symbol to an exact saved chunk and line range with
   host-only symbol search. Missing/ambiguous labels fail preparation rather than grading guesses.
5. Saves a source fingerprint, snapshot/MCP/runner/audit-script SHA-256, model settings,
   private labels and symbol aliases, schema and a three-arm schedule in v2 `plan.json`.

An existing output directory is never overwritten. A failed preparation can leave partial output;
preserve/inspect it and choose a new directory. The example labels target functions in the pinned
CubeQL revision above. Other revisions must resolve the labels again. These three hand-authored
smoke tasks are not a representative held-out benchmark.

`--embedding hash` is supported for offline plumbing tests, not semantic/hybrid quality claims.
There is no automatic download, checkout/reset of your working tree, or modification of your
Codex configuration/authentication. The runner reads no auth files itself; Codex uses its existing
authentication when explicitly executed.

## 2. Inspect, then explicitly execute

```bash
python3 scripts/harness_eval.py report /tmp/cubeql-agent-experiment-v1/plan.json

# Real MCP startup/catalog check for both augmented arms, no Codex/model request.
python3 scripts/harness_eval.py preflight /tmp/cubeql-agent-experiment-v1/plan.json \
  --output /tmp/cubeql-agent-preflight-v1

# This can consume paid API credits or your Codex account allowance and send code to the model.
python3 scripts/harness_eval.py run /tmp/cubeql-agent-experiment-v1/plan.json \
  --codex /absolute/path/to/codex --execute --max-runs 18
```

Three tasks × two repeats × three arms = 18 trials. `--max-runs` limits newly started trials in
this invocation, not tokens, dollars or cumulative runs across invocations. A smaller limit leaves
an explicitly incomplete matrix. Completed trials are skipped on subsequent runs; incomplete trial
directories are not overwritten or automatically retried. Failed completed trials remain in the
denominator. Use a new experiment directory for a deliberate rerun rather than cherry-picking it.

`run` also performs this non-model preflight automatically before any paid execution in
each invocation, saving logs under a unique `preflight-*` directory. Missing tools or startup
failure abort before model execution. Preflight verifies **server connectivity/catalog only**;
it does not prove that a subsequent model sees or chooses a tool.

For the actual augmented trial, a transparent `scripts/mcp_audit.py` stdio proxy records
successful initialization and `tools/list` responses from that **same Codex exec client**.
The tool definitions are forwarded unchanged. Search arguments and code responses are not
written to the discovery audit. The audit is bounded to 4 MiB, protocol lines to 1 MiB, and
the parent trial timeout still applies. Managed command allowlists must permit this Python
wrapper; policy failures are not bypassed.

If the actual client never discovers the complete expected read-only catalog, the trial is
invalid and scheduling stops before more model executions. A prior preflight cannot rescue
such a trial. Client discovery is stronger than independently listing a server's tools, but
still is **not** proof of model attention or adoption; search events are counted separately.
A CLI using cached catalogs without `tools/list` will fail this audit conservatively rather
than silently be counted as a validated augmented arm.

New runs require v2 plans. v1 plans are report/regrade-only, and changed runner/audit scripts
require a new plan. Do not mix newly graded/executed trials into old experiment directories.

Each trial gets a new temporary source copy and Git index, outside the artifact directory, with
no labels/previous answers copied into it. The user's checkout is not edited. The runner invokes
`codex exec --json --ignore-user-config --ephemeral --sandbox read-only`, supplies a structured
answer schema, and passes per-process configuration overrides. Ordinary local tools remain
available; web search, multi-agent tools, memories, hooks and plugins are disabled for these runs.
No execpolicy rules are disabled, no write sandbox is enabled, and no bypass flags are used.
Inherited administrator requirements can reject settings; such failures are recorded, not bypassed.

The CLI must support the flags checked by the runner. Its version is recorded and must remain
the same when continuing a plan. MCP startup is required in augmented arms; failure must not
quietly turn them into baseline runs. Host/device startup and query inference are included in the
trial's wall time; this is a cold-process agent comparison, not a warm GPU microbenchmark.

Source/snapshot/schema/binary fingerprints are checked before execution. Copying the source is
outside the timed interval; agent startup through exit and pipe drainage are inside it. The runner
checks workspace bytes afterward for modifications. It caps each stdout/stderr log at 32 MiB and
terminates only its newly created process group on timeout/log overflow. Live execution currently
requires POSIX (macOS/Linux). If the host denies process-group signaling, it stops the direct
child, records that cleanup limitation, and stops scheduling further trials. This is not a robust
container/process supervisor; operators should inspect such failed runs and any descendants.

## Results and what they mean

Each trial directory contains:

- `invocation.json`: exact argv, prompt and harness version, but no environment/auth dump.
- `events.jsonl`: bounded raw Codex stdout events.
- `stderr.log`: bounded CLI/MCP diagnostics.
- `answer.json`: the final structured model response, when produced.
- `result.json`: execution status, parsed metrics, policy checks and location grading.
- `mcp-discovery.jsonl`: actual client's discovery responses for augmented trials.

Use `report` to summarize without any model request. Its stdout JSON can be saved by your normal
shell/workflow; it does not overwrite an existing report. Missing runs, invalid runs and valid
runs are separate. The completed-run location-success denominator includes failures as zero;
it is not reported as a representative full task-success metric.

**Location grading (v2):** each required path and symbol identity must be cited with a valid
line range contained within the corresponding saved chunk. Cross-file tasks require all labels.
Besides the exact snapshot-local symbol, preparation records conservative aliases for
conventional Cargo `src/lib.rs`, `src/main.rs`, `src/foo.rs`, and `src/foo/mod.rs` layouts.
For example, `build_repository_batched` in `crates/cubeql-code/src/lib.rs` can also be cited
as `cubeql_code::build_repository_batched`; `Collector::emit_range` retains its impl scope.
Crate/module/impl prefixes are **not** stripped arbitrarily. Wrong namespaces, files and
line ranges still fail. Explicit Cargo target paths, `#[path]` attributes and `include!`
disable alias inference conservatively; this is not a full Rust name resolver and can
produce false negatives for nonstandard layouts. The exact snapshot spelling always works.
This checks finding/citing annotated code, not explanation correctness, absence of irrelevant
extra citations, successful code modification, or passing project tests. Explanations explicitly
remain `not_automatically_graded` and should be manually reviewed.

The first real smoke experiment exposed false negatives in v1's exact-spelling grader.
Regrade saved answers **offline into a separate new file** without changing the original
plan, answers, run validity, tokens or scores:

```bash
python3 scripts/harness_eval.py regrade /tmp/old-experiment/plan.json \
  --output /tmp/old-experiment/regraded-v2.json
```

The derivative records its grader version, original plan hash, prior grading per run and
`post_hoc_regrade: true`; it refuses an existing output path. Original `report` results
stay unchanged. Old runs cannot gain retroactive discovery evidence: reports warn about
unobserved discovery and zero search adoption even when corrected location scores improve.

**Tool adoption:** counts deduplicate `item.started` / `item.completed` by item identity. Unexpected
MCP calls in baseline, foreign MCP servers, wrong successful retrieval strategies, unknown item
types, web search/file changes and broken/incomplete event streams invalidate a trial. Failed
attempts can still contribute tool-call counts; transcript inspection is required for diagnosis.
These are observed-event checks, not a security monitor for arbitrary shell-command contents.
Reports separate `client_discovery_observed_runs`, `search_adoption_runs` (attempts), and
`successful_search_runs` (non-error responses with the correct strategy and a hits array).
Failed calls alone do not remove the warning about missing successful retrieval evidence.

**Context:** `read_bytes_returned` counts UTF-8 bytes from observed `read_code` results, including
repeated pages, without double-counting structured/text fallback copies. Distinct chunk counts
use snapshot/chunk identity. Missing result payloads produce null, not a fabricated zero. Shell
reads/previews/context truncation are not measured as consumed context. This does not reveal
which code the model actually used in reasoning.

**Tokens:** use the CLI's `turn.completed.usage`. Input, cached input, output and optional reasoning
token values stay separate; missing values remain null. Do not add cached input to input as an
extra cost category or assume reasoning tokens are disjoint from output. Reports show observed
coverage; totals are null if any included run lacks that field. These are harness-reported usage,
not independently measured context tokens, spend or a hard token budget.

The CLI adapter follows official [non-interactive JSONL/structured-output guidance](https://learn.chatgpt.com/docs/non-interactive-mode)
and [per-server MCP configuration](https://learn.chatgpt.com/docs/extend/mcp). It was developed
against local CLI 0.159.2; upstream event/config changes may require adapter updates. Preserve raw
evidence and treat unexpected schemas as a compatibility problem rather than a quality gain.

## Trust and evaluation boundaries

Use trusted repositories, plans and executables. Code, raw transcripts and prompts can contain
secrets. Output directories are created with mode 0700, but this is not encryption. `--execute`
uses existing CLI credentials/allowances and can transmit repository code to its configured model.
The runner does not read/copy auth files or enable a model provider automatically.

Read-only shell sandboxing is not a confidentiality boundary. Prompts prohibit reading outside
the workspace and no answer labels are copied in, but a malicious/cheating agent or repository
is not securely isolated from every host path, user-level instruction, model prior or side channel.
Global authentication, managed policies and shared caches remain; the experiment is not fully
hermetic. Use an externally isolated environment for adversarial or privacy-sensitive evaluation.

No real paid agent experiment is automatically run by tests. No improvement is claimed merely
because the runner exists. Start with reviewed tasks, inspect actual tool adoption and failures,
then expand labels/repeats before drawing conclusions about semantic search, fusion or GPU value.

## Offline validation

```bash
python3 -m unittest discover -s scripts/tests -v
cargo build -p cubeql-code -p cubeql-mcp
CUBEQL_EVAL_REAL_BINARIES=1 python3 -m unittest discover -s scripts/tests -v
cargo test -p cubeql-mcp
```

The integration fixture emulates only Codex CLI events. It launches the **real CubeQL indexer and
MCP server**, runs all three arms and verifies parsing/grading, fixed source despite a dirty
checkout, policy flags, and artifact integrity. It is labeled OFFLINE and is not an LLM. Real model
execution requires the separate operator opt-in above. Run Cargo feature variants sequentially
when they share `target/debug/cubeql-mcp`, or use separate Cargo target directories.

Regressions also cover scoped Rust aliases and rejected namespaces, immutable offline
regrading, a callable server with missing client discovery (stop scheduling), and a failed
MCP startup (no model execution, preserved diagnostics).
