# CubeQL repository collection

`tools/repository_collect.py` runs CubeQL's native `index` and `search` CLI for a frozen RQ2 plan. It creates one snapshot for each repository, then starts a fresh CLI process for every query and requested strategy. The module uses only Python's standard library and the read-only `repository_eval.py` APIs.

Example (from `rq2-stack/stage`):

```sh
python3 tools/repository_collect.py \
  --plan /path/to/plan.json --roots /path/to/roots.json \
  --binary /absolute/path/to/cubeql-code --output /path/to/new-collection \
  --strategy lexical --strategy symbol --candidate-k 100 --top-k 100
```

`roots.json` is an object whose keys exactly match the repository IDs in the plan, including format-version-1 plans. Output directories are create-new. The plan and root map are copied into the collection. Each native invocation has numerical stdout/stderr filenames, a timeout, and a combined byte limit; partial output is retained. Invocation, exit, timeout, malformed response, and source-validation failures remain explicit failed or timeout query records. Index failure is propagated to every query in that repository and every strategy, preserving full query coverage.

Supported strategies are `lexical`, `symbol`, `semantic`, and `hybrid`. Search receives the frozen query literally after `--`, the frozen path-prefix filter when present, `--include-source`, and explicit top-k/candidate-k budgets. Provider arguments are passed to indexing, and to semantic/hybrid queries. `hash` rejects model arguments; `minilm` requires `--model-dir`; `process` requires `--encoder-config`. No provider downloads, command retries, or long-lived server process are started. Collection currently rejects non-POSIX hosts. On POSIX, each command is started in a private session. Timeout and output-limit cleanup terminates its process group and, for process encoder profiles, enumerates process groups in that private session so workers in separate groups are included. The collector checks that session enumeration is available before launching a process encoder and returns a failed execution record if it cannot verify cleanup. If enumeration fails during a command, the result records the cleanup limitation; it never signals a process group unless it was identified in the private session.

The run's `candidate_k` is the exported top-k window given to the scorer; internal `candidate_k` remains in the run configuration. Per-command elapsed time is the full fresh-process wall time (index build or search), not a resident-server or kernel-only timing. Complete native responses and their native timing fields are preserved in the stdout evidence files; the manifest records command lines, command timings, snapshots, inputs, and SHA-256 hashes for all output artifacts.

Before collection, the source inventory must exactly match the plan. It is rechecked after all commands, alongside the CubeQL executable, model directory, encoder configuration, and encoder executable and absolute launch-file arguments named in the configuration. Any change prevents run and report creation; command logs and execution records remain. The collector rejects an output directory inside any source root. This check detects source/profile changes around execution, but cannot prove behavior inside a trusted executable or ensure a model directory's semantic suitability. The collector does not tune winners, reduce denominators, or claim that fresh-process timing represents a resident deployment.
