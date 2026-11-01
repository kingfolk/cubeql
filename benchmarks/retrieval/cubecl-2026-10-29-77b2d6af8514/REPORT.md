# CubeCL repository retrieval benchmark — 2026-10-29

On this 64-query development set, hybrid achieved **69.14% anchor Recall@10**, compared
with **54.95% for lexical** and **42.71% for semantic**. Hybrid did not win every category,
wording variant or ranking metric. Evidence completeness across files and robustness to
rephrasing remain substantial weaknesses. These results apply to the pinned CubeCL source,
MiniLM model and annotations below; they do not measure coding-agent task success.

This run added 24 source-grounded queries to the original 40 and checked candidate budgets,
paired wording, deduplication, source-span coverage and ambiguous labels. Retrieval algorithms
were not changed or tuned. No paid language-model or agent comparisons were run.

## Reproducibility and provenance

| Item | Recorded configuration |
|---|---|
| CubeQL checkout at evaluation time | `77b2d6af8514bbfee78819eea88c872f57d465b0` |
| Evaluated release executable SHA-256 | `112b823f16741b0334265fffe69228ea42f6d98be6c2593c88db86ea6a946272` |
| CubeCL source revision | `5037c0639ce3b1205392bdbe43dcf25e2ea09e29` |
| Corpus | Fresh committed-source export; 1,127 indexed files, 27,289 chunks; excludes queries, labels and reports |
| Corpus fingerprint | `3ad4afd5220b95c3987c463f6ebe009c1c5adab2ccac8bc256da69bf8d7176cb` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Encoding | Candle CPU FP32, 384 dimensions, attention-mask mean pooling, L2 normalization, 256-token limit, no prefixes |
| Retrieval | CubeCL CPU, cosine, FP32; uniform `language=rust` filter; no answer-path filters |
| Main settings | 50 candidates per source, final window 50; separate 100-candidate run; 3 warm timing repeats |
| Labels | Original 40 plus 24 queries; 16 per category; 66 unique anchors in 33 files |

The directory suffix identifies the CubeQL checkout recorded during the experiment, not
the later publication commit. The existing binary was identified separately by its hash;
its build revision was not independently embedded or attested. Rebuilding that checkout
is therefore a reproduction attempt, not a claim to reproduce the exact historical binary.
The pinned CubeCL source was available locally; availability of that revision on upstream
GitHub was not verified. No upstream source links are substituted for the preserved evidence.

[metadata.json](metadata.json) records source, model and binary identities. The original
query files, SHA freezes, unique-label validation and invocation records are preserved in
[artifacts.tar.gz](artifacts.tar.gz). [queries.json](queries.json) is the final 64-query set;
all labels resolve by `path + symbol`, and cross-file queries require at least two files.
Unique resolution checks label validity, not whether a question is unambiguous.

The original 40 queries were retained byte-for-byte. Separate original/extension runs and
the combined run produced identical per-query quality metrics. The main evaluations ran
at 06:33–06:34 UTC; paired diagnostics ran at 12:11–12:12 UTC on October 29. The intervening
gap was not model encoding time. Index construction ran approximately 06:10–06:33 UTC
(about 23 minutes), without a dedicated precise build-time measurement.

Hash-vector fixtures were used only for annotation and pipeline checks. All quality
results below use real MiniLM embeddings.

## Metrics and retrieval modes

**Recall@5/10** is the fraction of a query's labeled anchors found in the first 5/10 hits,
averaged over queries. **MRR@50** averages the reciprocal rank of the first labeled anchor
within the first 50 hits. **All anchors@10** is the fraction of queries whose entire labeled
evidence set appears in the top 10. Finding one function does not establish a complete
cross-file chain.

Labels are positive anchors, not exhaustive relevance judgments. An unlabeled hit is not
automatically a false positive, and a missed anchor does not prove the results are useless.
No precision, no-answer detection or alternative-OR-answer metrics are claimed.

| Mode | Implementation evaluated |
|---|---|
| literal | Case-sensitive whole-query substring matching, with symbol priority; not a full ripgrep comparison |
| symbol | Symbol-index matching, including qualified names and identifier atoms in the query |
| lexical | Identifier-aware token coverage with IDF weights and an exact-phrase bonus; not BM25 |
| semantic | Exact cosine search over MiniLM vectors |
| experimental_rrf | Legacy evaluator baseline: semantic + literal, equal-weight RRF, constant 60 |
| hybrid | Current retrieval entry point: semantic + lexical + symbol, equal-weight RRF, constant 60, then deduplication |

Each source has the same candidate limit and each mode the same final output window.
Hybrid can combine up to three candidate lists and does more work: this is not an
equal-compute or equal-union-budget ablation. Legacy semantic/literal/RRF do not use the
current entry point's deduplication; a matched diagnostic below checks that confound.

## Overall results

64 queries, candidate_k=50. Recall and completeness are percentages.

| Mode | Recall@5 | Recall@10 | MRR@50 | All anchors@10 |
|---|---:|---:|---:|---:|
| literal | 25.00 | 25.00 | 0.2500 | 25.00 |
| symbol | 25.00 | 25.00 | 0.2500 | 25.00 |
| lexical | 41.93 | 54.95 | 0.3822 | 48.44 |
| semantic | 27.86 | 42.71 | 0.2282 | 37.50 |
| experimental_rrf | 43.49 | 56.77 | 0.3618 | 51.56 |
| hybrid | 60.81 | 69.14 | 0.4974 | 60.94 |

By per-query Recall@10, hybrid had 15 wins, 2 losses and 47 ties against lexical, and
21 wins, 1 loss and 42 ties against semantic. Lexical had 18 wins, 10 losses and 36 ties
against semantic. These counts do not replace ranking or evidence-completeness analysis.

The original and extension results remain separate:

| Set | lexical R@10 / MRR@50 | semantic R@10 / MRR@50 | hybrid R@10 / MRR@50 |
|---|---:|---:|---:|
| Original 40 | 57.08% / 0.3978 | 37.92% / 0.1973 | 72.29% / 0.5192 |
| Added 24 | 51.39% / 0.3563 | 50.69% / 0.2796 | 63.89% / 0.4611 |

Lexical and semantic differ by only 0.69 percentage points on the extension's Recall@10;
this subset does not establish a stable ordering between them. On the six added behavioral
queries, semantic was better: **66.67% R@10 / 0.5590 MRR**, against hybrid's
**58.33% / 0.2433** and lexical's **33.33% / 0.0828**.

## Categories and representative cases

Each category contains 16 queries. Values below are Recall@10.

| Category | literal | symbol | lexical | semantic | Legacy RRF | hybrid |
|---|---:|---:|---:|---:|---:|---:|
| identifier | 100.00% | 100.00% | 81.25% | 37.50% | 93.75% | 93.75% |
| conceptual | 0 | 0 | 56.25% | 37.50% | 37.50% | 56.25% |
| behavioral | 0 | 0 | 50.00% | 59.38% | 59.38% | 78.12% |
| cross_file | 0 | 0 | 32.29% | 36.46% | 36.46% | 48.44% |

Exact-name controls show why symbol/literal paths remain useful. Entire natural-language
queries usually do not occur as contiguous source substrings, so legacy RRF's quality
metrics equal semantic's in the other three categories. Hybrid adds lexical and symbol
evidence; these results do not establish that RRF alone inherently improves retrieval.

Ranks below are from the candidate_k=50 reports:

| Query ID / intent | Observed result and source check |
|---|---|
| `v2-behavior-zero-patience`: measure once even with zero patience | Semantic rank 1, hybrid rank 10, lexical absent from top 50. `PatienceTracker::exhausted` uses `min_measured.max(1)`. Fusion demotes the correct semantic first hit. |
| `v2-concept-layout-selection`: select address mapping from shape/strides | Lexical rank 2, hybrid rank 9, semantic absent from top 50. `register` selects Plain / Strided / Permuted branches. |
| `v2-behavior-swizzle-disabled`: zero mask avoids byte conversion | Semantic/hybrid rank 3, lexical rank 7. `Swizzle::apply` returns offset directly for zero mask; only the other branch multiplies by type_size. |
| `concept-packed-conversion`: split packed cast lanes | Semantic/hybrid miss the exact method in top 50; lexical rank 35. Semantic's first hit is the relevant impl's comment/header, lines 485–488; the actual `unroll` method is a separate chunk. Conceptual proximity does not deliver the implementation evidence. |
| `cross-cuda-artifact-and-load`: compile PTX and load its entry point | All three modes find only one of two anchors in top 10. Hybrid's first hit is a `CudaArtifactCompiler` declaration, not the required `finalize` implementation. |

Of 16 cross-file queries, all anchors appeared in top 10 for **1/16 lexical**, **3/16
semantic** and **3/16 hybrid**. Hybrid improves cross-file anchor recall but still completes
only 18.75% of the labeled chains. Cross-file MRR is actually higher for lexical: 0.4820,
versus hybrid 0.3468 and semantic 0.1772. The first relevant anchor is not the whole answer.

Non-target results include related declarations, reinterpret casts and other addressing
implementations. Some are useful navigation; the actionable shortfall is missing concrete
implementations or the remaining evidence chain. The archive's validation records give
source paths, line ranges and file hashes for checking the pinned source export.

## Coverage expansion and bias diagnostics

The original 40 queries cover only 20 evidence files and repeatedly target unrolling,
CUDA compilation and device lifecycle. Before seeing semantic rankings, 24 queries were
added, six per category, covering autotuning, timing noise, tensor layouts/contiguity,
swizzling, quantization-scale rounding and roofline limits.

The extension and combined sets were frozen at 06:16 UTC. The protocol, freezes and
original query files are archived unchanged. This is a pre-frozen developer extension,
not independent held-out validation: the same assistant authored it against public source.
Initial `PatienceTracker` / `SampleSet` labels failed validation because struct and impl
headers shared a symbol; they were replaced with unique method names before rankings.
The correction note and both invalid drafts are retained.

### Candidate budget: nearly flat overall, not monotonic per query

Increasing candidates per source from 50 to 100 changes hybrid Recall@10 from **69.14%
to 69.40%**; lexical and semantic stay unchanged. Legacy RRF changes from 56.77% to 58.33%.
A larger window does not remove the main differences.

For hybrid, the CUDA artifact/load question improves from 1/2 to 2/2 anchors, while the
layout-selection and zero-patience questions lose their top-10 anchor. Additional candidates
change which rows receive support from multiple lists and reorder fusion results. A flat
aggregate does not establish per-query stability. All changes are retained in the summary.
MRR@100 and MRR@50 have different cutoffs and are not treated as interchangeable metrics.

### Six paired wording diagnostics

Six alternative wordings were frozen before rankings, preserving the same anchors.
These are paired diagnostics, not six additional independent tasks.

| Mode | Original mean R@10 | Rephrased mean R@10 |
|---|---:|---:|
| lexical | 50.00% | 0.00% |
| semantic | 66.67% | 33.33% |
| hybrid | 83.33% | 33.33% |

For the loaded-kernel fast path, rephrasing moves lexical's anchor from rank 4 to 32;
semantic stays at 8 and hybrid moves from 1 to 3. Semantic evidence helps here. But after
rephrasing the inplace-write question, semantic/hybrid both lose the target from top 50.
On the zero-patience wording, semantic falls from rank 1 to 34. Neither semantic nor hybrid
is a reliable paraphrase-understanding solution on this evidence.

### Deduplication, source coverage and ambiguity

Eight prespecified questions, two per category, were searched with lexical/semantic/hybrid,
with deduplication on and off: 48 local CLI searches. **All 24 matched pairs have identical
R@5, R@10, reciprocal rank and all-anchor metrics.** This subset does not support dedup as
the main explanation of the gains; it does not rule out effects on the other questions.

Two additional diagnostics were exploratory, after scores were available. They neither
change the main judgments nor select replacement questions:

- **Source-span containment:** counting returned chunks that fully contain each anchor's
  line range gives the same mean as exact-anchor recall for every mode. The gap here is
  not caused by returning an enclosing implementation chunk under another symbol.
  Merely finding an anchor's file yields 75.52% for lexical, 72.66% for semantic and 89.84%
  for hybrid. File presence is navigation coverage, not answer completeness.
- **Ambiguous `anchor` query:** this one-word query has other same-named functions in the
  corpus. A uniquely resolved label does not uniquely specify user intent. Other `anchor`
  functions cannot automatically be judged wrong. The original query and scores remain;
  excluding it only in a sensitivity check gives 63-query R@10 of 55.82% lexical, 43.39%
  semantic and 70.24% hybrid. Future annotation review should specify autotuning intent
  or represent equivalent answers, rather than treating this as clean algorithm failure.

## Timing boundaries

Full local MiniLM document encoding took about 23 minutes and produced a 56,261,290-byte
Arrow index. These are initialization measurements from the combined 64-query process:

| Phase | Time |
|---|---:|
| Read Arrow | 56.74 ms |
| Load MiniLM | 299.05 ms |
| Encode all 64 queries | 713.10 ms (about 11.14 ms/query) |
| Build text index | 538.92 ms |
| Current retrieval index upload | 21.44 ms |
| Legacy baseline index upload | 21.17 ms |

For each query, take the median of three warm retrieval repeats, then the median across
64 queries:

| Mode | Resident retrieval ms |
|---|---:|
| literal | 13.364 |
| symbol | 0.007 |
| lexical | 17.656 |
| semantic | 8.669 |
| experimental_rrf | 21.278 |
| hybrid | 27.389 |

Retrieval excludes model loading, query encoding, filter preparation, result materialization
and process startup. Hybrid includes its multiple searches, fusion and deduplication.
These are not CLI end-to-end latencies. First semantic invocations are recorded separately
but are not fresh-process cold starts. There is no controlled OS-cache cold-start experiment
or CPU-versus-Metal comparison. Warm repeats are timing samples, not independent quality
samples; run order and system load can influence latency. Hardware/OS details were not
captured in the frozen run, further limiting performance comparisons across machines.

## Uncertainty, limits and next steps

Queries sharing anchors form 41 connected clusters. A 2,000-resample paired descriptive
cluster bootstrap gives hybrid minus lexical R@10 of +14.19 percentage points, with an
interval around [+5.87, +23.33]. Lexical minus semantic is +12.24, with [-1.98, +25.00].
The extension-only hybrid-minus-lexical interval also crosses zero. These describe internal
development-set uncertainty, not population significance for real user traffic. Shared
topics, author style and other dependencies remain.

Limits include a single familiar Rust repository; AI-authored, public queries and labels
pending independent human review; incomplete positive judgments; and a category mixture
that does not represent production traffic. MiniLM is a general English model with a
256-token limit. Long-function truncation may matter, but no causal ablation establishes
its effect here. The schema does not represent no-answer or OR-equivalent evidence.

The expansion and counterexamples support a limited engineering conclusion: retain the
symbol path for precise names; hybrid improves current anchor coverage; semantic supplies
useful evidence for some behavior questions. Single-pass fusion remains insufficient for
stable cross-file evidence gathering or rephrasing. Repeating this same developer set
cannot establish independent generalization.

Next, independently review ambiguous/cross-file labels, collect real harness queries and
freeze a separate validation set. A concrete feature to evaluate is navigation from a
retrieved file/declaration to full implementations and related evidence. Code-specific
embeddings, query decomposition and fusion changes require new frozen experiments;
none were implemented or tuned in this run.

## Artifacts and reproduction

- [summary.json](summary.json): all four main run summaries, categories, timing, intervals
  and diagnostics, with no changes to historical measurements.
- [per-query-metrics.csv](per-query-metrics.csv): all six modes for original, extension,
  combined-k50 and combined-k100 runs. Publication uses LF line endings; the archive
  retains the original CSV bytes, with identical records.
- [queries.json](queries.json) and [wording-queries.json](wording-queries.json): labeled tasks.
- [artifacts.tar.gz](artifacts.tar.gz): original result JSON, all 48 dedup results, query
  versions and invalid drafts, validations, logs, SHA freezes and historical scripts.
- [metadata.json](metadata.json): provenance, archive inventory/hashes and excluded large
  artifacts. Historical local paths in raw records are retained as provenance.

Verify this publication without a model, network, Git writes or extracting the archive:

```sh
python3 -B benchmarks/retrieval/cubecl-2026-10-29-77b2d6af8514/verify.py
```

To rerun, obtain a CubeQL build with CPU and `local-embeddings`, a local repository that
contains the pinned CubeCL commit, and the pinned MiniLM artifacts matching metadata.
CubeQL's manifests use relative paths to a sibling CubeCL checkout, so preserve that
layout when building the historical version. Build separately from an active checkout;
record the new binary hash and actual dependency state. Model acquisition/building are
preparation steps, not part of the timings above.

The following commands use an already-built executable and validated model directory.
Set `CUBEQL_CODE`, `CUBECL_REPO`, `MODEL_DIR` and `RUN_DIR` to absolute paths; `RUN_DIR`
must be a new directory. Run from the CubeQL repository root so the label path resolves.
The source export must stay separate from all reports and labels:

```sh
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/source"
git -C "$CUBECL_REPO" archive 5037c0639ce3b1205392bdbe43dcf25e2ea09e29 \
  | tar -xf - -C "$RUN_DIR/source"
"$CUBEQL_CODE" index "$RUN_DIR/source" \
  --embedding minilm --model-dir "$MODEL_DIR" \
  --output "$RUN_DIR/minilm-fp32.arrow"
"$CUBEQL_CODE" evaluate "$RUN_DIR/minilm-fp32.arrow" \
  benchmarks/retrieval/cubecl-2026-10-29-77b2d6af8514/queries.json \
  --model-dir "$MODEL_DIR" --candidate-k 50 --repeats 3 --format json \
  > "$RUN_DIR/combined-k50.json"
```

Check the resulting corpus fingerprint, model configuration and index precision against
metadata before comparing scores. Reruns belong in a new dated/revisioned directory;
changing the model, corpus or labels is a separate experimental condition.
