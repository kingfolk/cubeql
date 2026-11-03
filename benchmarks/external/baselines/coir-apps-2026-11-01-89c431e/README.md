# Frozen CoIR APPS baseline — 2026-11-01

Historical locally measured results over all 8,765 programs and 3,765 official test queries.
Packaging these files does not constitute a new inference run or a competitor reproduction.

| Configuration | nDCG@10 | Recall@10 | MRR@100 |
|---|---:|---:|---:|
| minilm | 0.06596 | 9.91% | 0.06082 |
| bm25_word | 0.01152 | 1.62% | 0.01153 |
| bm25_code | 0.01501 | 2.28% | 0.01452 |
| experimental_rrf | 0.06354 | 9.64% | 0.05325 |
| coderank_256_cubeql | 0.19840 | 27.70% | 0.18205 |
| coderank_256_reference | 0.19840 | 27.70% | 0.18205 |
| coderank_8192_cubeql | 0.23460 | 32.88% | 0.21387 |
| coderank_8192_reference | 0.23460 | 32.88% | 0.21387 |

nDCG/MRR are normalized, not accuracy. MRR is limited to exported Top-100 candidates.
This is one CoIR task, not its ten-task average or an agent evaluation. Each query has one labeled positive; equivalent unlabeled programs may exist.

## Verify and rescore

From the repository root, audit archive bytes without installing models or extracting corpus data:

```sh
python3 -B benchmarks/external/runner.py audit-baseline benchmarks/external/baselines/coir-apps-2026-11-01-89c431e
```

Pass this directory's actual path to the command. See [the portable recipe](../../README.md) for dataset preparation and `score-baseline`.
The archive contains own ID/score exports, per-query metrics, original provenance and failed preflight diagnostics. It excludes corpus/query text, qrels, vectors, weights and third-party source.

The original MiniLM run used a dirty adapter checkout before commit `89c431e`; do not infer binary build identity from the directory suffix. Original executable/source/dependency fingerprints are preserved.
RRF contains many ties: native CoIR's stable JSON sort and pytrec's document-ID tie ordering have different MRR. Both values remain in baseline.json and the evidence; scores are not perturbed.
Published model/peer numbers are not substituted for local measurements. The CodeRank context comparison was exploratory after the MiniLM result, and changes both query/document token ceilings.
