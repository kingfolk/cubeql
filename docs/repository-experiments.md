# Frozen repository experiments

This offline workflow binds reviewed validation plans to declared experiment arms
and then scores the complete set of submitted runs. It builds on
[validation review](repository-validation.md) and the existing
[repository scorer](repository-product-evaluation.md). It runs no retrieval,
model inference, paid API call, or coding agent task.

## Freeze conditions

Prepare an existing version 2 multi-repository validation plan, its review record,
and the development plans actually used during exploration. The review audit must
be ready: at least two repositories, supplied development coverage, human approval
declarations for every query, no requested changes, and no detected overlap.
Reviewers must perform the work themselves before recording approval. Do not put
placeholder approvals in a real review record.

The experiment specification has this shape (identity strings below are examples):

```json
{
  "format_version": 1,
  "name": "declared-fusion-comparison",
  "arms": [
    {
      "name": "baseline",
      "tool": {"name": "cubeql-code", "revision": "baseline-commit"},
      "configuration": {
        "model_identity": "checkpoint-and-encoder-digest",
        "corpus_identity": "snapshot-digest",
        "chunking_identity": "chunker-version-and-settings",
        "filter_policy": "literal-path-prefix",
        "mode": "hybrid"
      },
      "candidate_k": 100
    },
    {
      "name": "candidate",
      "tool": {"name": "cubeql-code", "revision": "candidate-commit"},
      "configuration": {
        "model_identity": "checkpoint-and-encoder-digest",
        "corpus_identity": "snapshot-digest",
        "chunking_identity": "chunker-version-and-settings",
        "filter_policy": "literal-path-prefix",
        "mode": "hybrid"
      },
      "candidate_k": 100
    }
  ]
}
```

At least two distinct conditions are required. Arm names are unique, the first
arm is always the baseline, and `candidate_k` is an integer of at least ten.
Configuration accepts additional finite JSON values and requires the four
nonempty identity fields shown above. Every field is matched exactly in submitted
run metadata. Identity strings are declarations: include concrete digests and
retain producer receipts separately. This tool does not verify model weights,
source checkout contents at execution, equal branch budgets, or execution settings.
Different identity values are allowed for deliberately different experimental
conditions; a comparison is not automatically a common-model control.

```sh
python3 -B tools/repository_experiment.py freeze \
  --plan validation-plan.json --review validation-review.json \
  --development-plan development-plan.json \
  --spec experiment-spec.json --output protocol.json
```

Repeat `--development-plan` for every known development plan. The protocol embeds
the full plan, review, development plans, specification, and derived audit, sealed
by a canonical SHA-256 fingerprint. Outputs are create-new. Nothing is signed,
timestamped, or sent to a registry. Freeze and preserve it **before** running the
conditions. Its hash detects changes but does not prove temporal preregistration,
reviewer identity, unseen questions, semantic disjointness, or absence from model
pretraining. Supplied development plans may be incomplete; the workflow cannot
discover undisclosed exploration.

## Score the whole experiment

Collect existing native run interchange version 1 files against the protocol's
plan fingerprint. Use exactly the frozen arm `name`, `tool`, `configuration` and
`candidate_k`. Every arm and query is required exactly once; failed, unsupported,
and timed-out queries must remain explicit records with no hits and an error.

```sh
python3 -B tools/repository_experiment.py score --protocol protocol.json \
  --run baseline.json --run candidate.json --output experiment-report.json
```

Input run order does not change the baseline. Missing/extra conditions, changed
configuration or budgets, stale review/audit data, and malformed source plans fail
before output creation. A validly regenerated protocol is a new protocol; changing
inputs and recomputing all hashes cannot be detected as historical misconduct.

The `evaluation` field preserves the existing scorer's raw runs and full query
denominators, query/category/repository metrics, equal-repository averages, and
paired per-query gains and regressions. Unequal candidate windows leave MRR
deltas null. A small final window leaves larger recall cutoffs null. Results
describe positive-anchor containment in the frozen source spans; unjudged chunks
are not labeled false positives, and no task-success conclusion follows.

`declared_elapsed_ms` reports the number of queries with supplied observations,
their sum, and a complete sum only when every query has a time. This includes
failed-query observations when supplied. Missing values remain missing. These
sums are not process wall time, model inference time, resource-budget enforcement,
or evidence of a controlled speed comparison. Retain external command/deadline,
hardware, memory, and stdout/stderr receipts for those questions.

The old `repository_eval.py score` remains available for exploratory development
and old formats. This command gates only its own frozen-experiment path. No
production ranking, RRF weight, branch budget, or default setting is changed.

## Checks

```sh
python3 -B -m unittest discover -s tools/tests -p 'test_repository_*.py' -q
```

Tests use synthetic plans, runs, and explicitly fictional human declarations.
They establish format and arithmetic behavior, not independently reviewed real
validation results. Original64 queries, labels and prior evidence remain unchanged.
