# Native Semble collection

`tools/repository_semble.py` collects one frozen repository plan against the pinned local Semble source and an already-present local model. It starts one worker process per repository, builds one `SembleIndex` during `initialize`, then reuses that index for each query in that repository. The caller must provide the source checkout and model directory; the worker runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, its own `SEMBLE_CACHE_LOCATION`, and `SEMBLE_INDEX_WORKERS=0`. It does not resolve model IDs, download assets, or alter user settings.

The collector preserves Semble's ordinary code-search defaults. It does not override alpha, reranking, fusion, or branch candidate count. The worker reports Semble's `format_results` payload, native total search time, build time, indexed file/chunk counts, and package version. Semble internally retrieves `5 * top_k` candidates per semantic and BM25 branch; this budget is recorded because it differs from fixed-candidate comparisons. The standard run's `elapsed_ms` is resident JSON-RPC search wall time. It excludes indexing/startup. Semble returns full chunk content inside its search response, so response bytes and serialization work are included in that wall time. The timing boundary label alone does not make latency an equal-context comparison; this collector does not claim latency speedups.

For a frozen `path_prefix`, the worker enumerates indexed chunk paths beginning with that literal prefix and passes the resulting file list through Semble's native `filter_paths` selector before ranking/top-k. If no indexed path matches, it returns an empty native result without invoking `search`; Semble treats an empty selector as unfiltered. An empty filter object makes an ordinary unfiltered call. Language filtering is not inferred or combined with path filters.

## Invocation

```sh
python3 tools/repository_semble.py \
  --plan frozen-plan.json \
  --roots roots.json \
  --python /absolute/path/to/python \
  --source /absolute/path/to/semble-source \
  --model-dir /absolute/path/to/local-model \
  --source-revision 4633ee... \
  --output /absolute/path/to/new-output \
  --top-k 10 \
  --timeout-seconds 60
```

`roots.json` is an object mapping each frozen repository ID to its absolute source root. The output must not already exist and must be outside the frozen source roots, Semble source tree, and model directory. The output contains `run.json`, `report.json`, a copied `plan.json`, the exact frozen query/filter declarations in `manifest.json`, native payloads in `payloads/`, per-worker JSON-RPC/stdout/stderr traces, isolated runtime caches, and `provenance.json`. Each worker command, declared revision, source/model/interpreter fingerprints, native build metadata, request records, and payload digest are retained. The declared source revision is provenance supplied by the caller; source byte fingerprints provide the executable-source identity.

This collector validates the frozen plan and full repository inventories before and after collection. It validates each formatted payload using the same native payload rules as the offline Semble peer importer, then scores the results with the shared repository evaluator. A failed initialize marks every query for that repository failed; a failed or invalid search gets zero hits for that query. Outputs are create-new artifacts so an earlier run is not overwritten.

The collector records what actually ran, but does not claim that a declared revision alone cryptographically attests source history, that installed dependency versions reproduce a historical environment, or that development queries/labels establish independent held-out quality. A real Semble engine run with a downloaded local model is still a local development pilot.
