# Native CodeRank MCP and Semble development pilot

Measured on 2026-11-11 America/New_York. This is an eight-query, comment-rich synthetic development fixture. Both engines and models actually ran locally. The labels are not independently reviewed or held out. This report does not complete RQ2, establish the 80% multi-repository validation target, or measure agent task success.

## Frozen conditions

- Corpus: the checked-in four-file, 24-function Rust retrieval fixture at CubeQL base `ca31444`. The existing executables were not rebuilt for these Python changes; exact binary SHA-256 values are in `experiment.json`.
- Queries: the original eight fixture texts and IDs, with explicitly listed implementation-span anchors. The original query JSON is retained unchanged in the evidence. Rust language filters are redundant on this all-Rust corpus; path-prefix filters are retained.
- CubeQL: actual CodeRankEmbed, 256-token FP32 CPU condition, fresh Arrow snapshot, one MCP session shared across four strategies and all eight questions. The index-building encoder is a separate lifetime. Startup was observed once in the MCP trace; raw JSONL includes initialization, discovery, searches and source reads.
- Peer: real Semble 0.7.1 source at `4633ee75a1d8c9ce3998b610a14fc80e7fb7c802`. All 68 supplied source files were verified against the pinned Git tree blob hashes before use. It indexed four files into eight native chunks. Runtime dependency versions are locked separately; this is not a reproduction of Semble's historical published benchmark environment.
- Peer model: local potion-code-16M-v2 at revision `e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b`, loaded offline. Source/model/interpreter fingerprints and native build metadata are retained in peer provenance.
- Final top-k: 10. CubeQL candidate budget: 100 per source. Semble retains its default 5*top-k candidates per branch, auto routing and native reranking. Models, chunk units, internal budgets and returned source sizes differ; this is a native product-condition pilot, not a controlled model or ranking ablation.

## Measured location coverage

| Condition | Queries | Anchor Recall@10 | All required locations@10 |
| --- | ---: | ---: | ---: |
| cubeql-coderank-lexical | 8 | 1.0000 | 1.0000 |
| cubeql-coderank-symbol | 8 | 0.2500 | 0.2500 |
| cubeql-coderank-semantic | 8 | 1.0000 | 1.0000 |
| cubeql-coderank-hybrid | 8 | 1.0000 | 1.0000 |
| semble-potion-native | 8 | 1.0000 | 1.0000 |

Ten results can cover all eight Semble chunks, whereas CubeQL indexes 24 function units. Together with the tiny comment-rich corpus, this makes the fixture too easy for a comparative relevance claim. Symbol-only behavior is expected to miss natural-language queries. No default was tuned or changed from these outcomes.

The MCP trace records 32 successful searches with no source-read truncation under the configured budget. The counts measure observed collector behavior, not voluntary adoption by an agent. All-required metrics remain source-location coverage, not evidence that a model used the bytes.

## Timing and evidence

Resident request wall-time observations, source-read evidence and native embedding/retrieval/build phases remain in the raw collectors. Search latency excludes MCP read_code calls; Semble serializes full chunks in its search response. No equal-context speedup, CPU/Metal comparison or isolated kernel claim is made. See `diagnostics.json` for counts, timing boundaries and per-query first ranks.

`evidence.tar.gz` contains native stdout/stderr/JSONL, original queries, launch configuration, source/model/binary fingerprints and the first-party orchestration source. It excludes model weights, third-party source, binaries, Arrow snapshots and runtime caches. Some local absolute paths remain as provenance. `evidence-members.json` binds every archived file.

```sh
python3 -B benchmarks/retrieval/native-pilot-2026-11-11-ca31444/verify.py
python3 -B tools/repository_diagnostics.py \
  --plan benchmarks/retrieval/native-pilot-2026-11-11-ca31444/plan.json \
  --run benchmarks/retrieval/native-pilot-2026-11-11-ca31444/runs/cubeql-coderank-hybrid.json \
  --run benchmarks/retrieval/native-pilot-2026-11-11-ca31444/runs/semble-potion-native.json \
  --output /tmp/native-pilot-rescore-001
```

## Original CubeCL64 preparation

A separate create-new native baseline recipe was prepared locally with the original 64 query bytes, CodeRank256 CPU, candidate_k=100 and three repeats. No full-corpus encoding or 64-query evaluation was invoked. The source was exported from CubeCL `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`, avoiding generated target files. The initial strict-inventory rejection was retained: that archive includes 20 internal LICENSE symlinks. Only the generated copy's license links were materialized to their exact target bytes with a recorded conversion manifest; source code and the original checkout were untouched. Inspect that preparation before explicitly running the full condition.

Independent multi-repository labels, a full native peer comparison, common-unit/model controls and separately budgeted agent outcomes remain future work.
