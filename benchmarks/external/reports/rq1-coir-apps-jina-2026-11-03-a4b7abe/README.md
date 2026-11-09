# RQ1 production vector workflow and Jina reference — 2026-11-03

RQ1 makes externally encoded code vectors usable through CubeQL's supported CLI and library
APIs. This report validates persistence, query-only encoding and incremental reuse, then
runs a stronger local embedding reference through that same production import path.
All work remains local and uncommitted. No paid model calls were made.

## Measured source and frozen condition

- CubeQL base commit: `a4b7abe1437c0cf7066b852bf5c9b7a84da7b060` **plus the uncommitted RQ1 delta**.
  The directory suffix identifies the base, not a clean committed build. The archive contains
  the changed/new source snapshot, tracked patches and exact executable SHA-256.
- CubeCL: `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`; Apple M1, 16 GiB.
- CoIR APPS: all 8,765 candidate documents and all 3,765 official test queries, full judged
  denominator, fixed Top-100. Canonical input order/text match the RQ0 dataset hashes.
  Preparation exports no relevance judgments to the encoder.
- Jina: `jinaai/jina-code-embeddings-0.5b`, revision
  `4db235132dafbe56a8b9c5f59b59795ecf58a4a7`, 896 dimensions, FP32/SDPA, context 1,024,
  last nonpadding token, L2 normalization, official NL2Code query/document prefixes.
- Inference: PyTorch MPS. Retrieval: `Cpu(CpuDevice)` via CubeCL, FP32 cosine.
  MPS inference is distinct from a CubeCL Metal retrieval run. NumPy uses the identical saved
  FP32 vectors with exhaustive float64 cosine and four BLAS threads.
- The CPU condition and separate MPS condition were declared before relevance scoring.
  CPU encoding was interrupted because of runtime cost. MPS exhausted its allocator memory
  limit on `2` attempts; fresh processes resumed the identical
  frozen contract and reused validated checkpoints. This is a resource limitation of the
  optional local inference recipe. Long jobs on this machine needed manual resume;
  a resident inference service and bounded MPS memory use remain future work.
  No allocator safety limit was disabled and no CPU/MPS matrices were mixed.

Input lengths beyond the 1,024-token limit (after each role's prefix) were
`70` / 8,765 documents and
`72` / 3,765 queries. Those inputs were truncated, not dropped.
This shorter local condition was frozen before scoring and was not tuned against its results.

Exact artifact/model/input/library identities and attempt history are in `evidence.tar.gz`.
The encoder producer source is retained under `encoding/recipe-source` in the archive.
After its completed stage was sealed, an unused machine-specific staging import fallback
was removed and the eight Python checks passed again; normal repository lookup, inference
math and embedding configuration did not change. Replay does not execute the encoder.
The later source snapshot retains this cleanup separately from the frozen producer.

The original validation workspace name ends in `61f991f` because it was allocated before
RQ0 was merged. Actual encoder/replay provenance records the measured merge base above.

## Quality on the full frozen task

The following are unrounded direct pytrec metrics on the exported Top-100, after the
explicit self-ID policy. nDCG and reciprocal rank are normalized metrics, not accuracy.

| Engine, identical Jina vectors | nDCG@10 | Recall@10 | Recall@100 | Reciprocal rank in Top-100 |
|---|---:|---:|---:|---:|
| CubeQL CPU, FP32 cosine | 0.8386415772 | 0.9333333333 | 0.9856573705 | 0.8104653020 |
| NumPy, exhaustive float64 cosine | 0.8386415772 | 0.9333333333 | 0.9856573705 | 0.8104653020 |

Native CoIR scores and its stable-score MRR tie policy are preserved separately in
[summary.json](summary.json); direct pytrec may order exact score ties differently.
The complete raw rankings and every query's metrics are retained, including misses and ties.

For historical context, the frozen RQ0 experiments measured native nDCG@10 of 0.06596 for
MiniLM-256, 0.19840 for CodeRankEmbed-256 and 0.23460 for CodeRankEmbed-8192. Those are
separate model/configuration experiments, not a controlled ablation of model training or
context length, and were not re-encoded for this report. Matching dataset hashes and all
3,765 query IDs, Jina has higher per-query direct nDCG@10 on `2895` queries,
lower on `90`, and equal on `780` versus the saved
CodeRankEmbed-8192 control. Exact deltas are in
[paired-model-contrast.json](paired-model-contrast.json). This is an observed model/configuration
contrast; the inspected APPS test set is not an untouched validation set.

Jina's paper reports APPS nDCG@10 × 100 of 84.17; a public peer experiment reports 83.86.
They are author/peer reference values, not locally run competitor products. The paper used
8,192 tokens, FP16 and batch size 8; this run uses 1,024 tokens, FP32 and deterministic
length-based batches. Our context,
precision, preprocessing and evaluation details must accompany comparisons; agreement after
rounding is not proof of reproducing the entire published setup. The measured native
nDCG@10 × 100 is `83.864`; its difference from
the paper value is `-0.306` points.
This remains a local implementation of the disclosed recipe, rather than a verified exact
reproduction of the authors' hardware/precision pipeline. The paper explicitly lists
APPS training data, so this task is partly in-domain and does not establish performance on
unseen repositories. CodeRankEmbed's model card describes CoRNStack training; unknown APPS
exposure must not be inferred either way.

## Numerical control and acceptance checks

All parity diagnostics, including any near-tie order differences, remain in the archive.
Top-10 order agrees for `3765` / 3,765 queries; Top-100 order
for `3742` / 3,765; Top-100 candidate sets for
`3765` / 3,765. The largest reference-score gap in reordered
positions is `6.68276796789e-07`. None of those queries were excluded or perturbed.
Maximum returned cosine-score error: `1.74980328371e-06`;
the declared local bound was `5e-6`. Native aggregate nDCG equal: `True`;
native aggregate MRR equal: `True`.
A separately recorded fixed 128-document CPU/MPS sample had maximum coordinate error
`2.8796494007e-6` and minimum cosine `0.999999999810103`. This is a numerical sample,
not a backend performance comparison; full inference vectors all come from MPS.

| Acceptance check | Observed result |
|---|---|
| Default `cubeql-code` and `cubeql-embeddings` Rust suite | 63 passed |
| Final identified-vector/evaluator integration suite | 7 passed after timer-label changes |
| Python bundle/recipe/reference checks | 8 passed |
| Existing native/scoring/checkpoint recipe suite | 13 passed |
| Existing real MiniLM provider integration | Passed with verified local model artifacts |
| Real CodeRank CPU, Jina CPU and Jina MPS tiny workflows | Persisted indexes; `load_config` Top-1 |
| New query with query-only encoding | `save_config` Top-1; document corpus not encoded again |
| No-op external-model update | 0 embedded, 3 reused |
| One modified function body | 1 embedded, 2 reused |
| Source/order/text/config mismatch and invalid-vector controls | Rejected; no index output |
| Rust formatting and diff whitespace | Passed |
| Full production CLI replay, NumPy reference, native scores | 3,765 queries retained and verified |

The earlier tiny harness expected a nested JSON hit field whereas the actual CLI flattens
source fields. Its failure is retained separately; correcting that harness did not change
ranking. Initial invocation of the ignored MiniLM test omitted its required model-directory
environment variable; the documented invocation with the local cache passed.

## Timings and resource interpretation

- Model load on the successful resume attempt: `4.440` s.
- Successful document batches' inference, including checkpoints from earlier attempts:
  `3023.316` s;
  queries: `3147.165` s.
  These successful batch durations include per-batch tokenization, transfer, forward, pooling
  and readback; they exclude the preliminary length scan and checkpoint I/O. Failed work,
  interruption and model-load time are
  additional. They are not uninterrupted wall-clock throughput measurements.
- Document vectors reused on the successful attempt: `8765`.
- Replay document-vector lookup/validation: `37.628` ms;
  all query lookups: `65.801` ms; index setup/upload: `13.510` ms.
- Total query retrieval timers: `23300.255` ms, mean `6.189` ms/query.
  This is one sequential local run, includes its cold query, and has no repeated resident
  benchmark or controlled competing load. It is not an end-to-end query-encoding latency.

The CLI labels replay as `precomputed-vector-lookup`; imported lookup timers must not be
reported as model inference. This work establishes neither CPU-versus-Metal speed nor agent
success improvements. See [measurements.json](measurements.json) for machine-readable values.

## Reproduction and retained evidence

Follow [the supported vector workflow](../../../../docs/precomputed-vectors.md) and
[portable benchmark recipe](../../README.md). Use new output directories for new jobs;
resume only incomplete jobs with the exact original request and source identity.

The production sequence is `benchmark prepare` → external `encode` →
`benchmark run --embedding precomputed --vectors ...` → independent `reference` → native
`score` for both engines. Pinned model specs and requirements ship under `tools/`.

[evidence.tar.gz](evidence.tar.gz) retains ID/score rankings, TREC exports, per-query and
aggregate metrics, parity diagnostics, attempt/failure history, frozen requests, source
snapshots, command exit records, test evidence and provenance. [manifest.json](manifest.json)
lists archive members and SHA-256 checksums, plus the archive checksum. Weights, corpus/query
text, relevance judgments, raw vector matrices, Arrow indexes and evaluator source are
excluded. The original full local stage retains those separately for continued development.
A checksum establishes integrity, not an independent build attestation.

This is one CoIR task and a local reference, not the complete CoIR suite, an independently
reproduced competitor product, or a new untouched held-out evaluation. Jina weights are
CC-BY-NC-4.0 and remain an optional noncommercial reference; CodeRankEmbed is MIT according
to its pinned model card. Heavy model inference remains outside the generic core library.
RQ2 is still needed for independently run search-product and exact-engine comparisons.

Sources: [Jina model card](https://huggingface.co/jinaai/jina-code-embeddings-0.5b),
[Jina paper, including training-data disclosure](https://arxiv.org/html/2508.21290v2),
[CodeRankEmbed model card](https://huggingface.co/nomic-ai/CodeRankEmbed),
[pinned peer experiment](https://github.com/utkrshm/agentic-retrieval/blob/095f2b7326ae632ec4a0c6e9286c44a380181ecb/docs/experiments/07-benchmark-apps.md).
