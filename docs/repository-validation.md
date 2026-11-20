# Repository validation review audit

`tools/repository_validation.py` audits a frozen validation plan against a review
packet and zero or more frozen development plans. It is a workflow check for
separating query authorship, human review, and development material before an
evaluation run. It does not authenticate identities, prove when a plan or review
was created, or establish semantic independence. The plan fingerprints and
review hash are integrity checks over the submitted data; hashing does not prove
temporal preregistration.

## Inputs

The validation plan must pass `repository_eval._verify_plan` and declare
`spec.split` as `validation`. A ready audit also requires at least two declared
repositories. The review JSON has this exact shape:

```json
{
  "format_version": 1,
  "plan_fingerprint": "<validation plan fingerprint>",
  "queries": [
    {
      "query_id": "q-1",
      "author": {"identity": "author-1", "kind": "agent"},
      "reviews": [
        {"identity": "reviewer-1", "kind": "human", "decision": "approved", "notes": "Reviewed independently."}
      ]
    }
  ]
}
```

The JSON above is an illustrative example only; no human reviews were performed
for this implementation. The review must cover each validation query exactly
once. Authors and reviewers have a nonempty `identity` and a `kind` of `human` or
`agent`; an identity keeps the same kind throughout the packet. A reviewer cannot be the query author, and
reviewer identities cannot repeat within one query. Review decisions are
`approved` or `changes_requested`, and notes must be nonempty. Empty review
arrays are valid and produce `pending` status. Agent approvals are recorded but
do not satisfy the human approval gate. A query with any changes request has
`changes_requested` status, even if another reviewer approved it.

Every supplied development plan must pass the same plan verifier and declare
`spec.split` as `development`. Development plan fingerprints must be unique.

## Overlap check and readiness

For every validation query and every query in every development plan, the audit
checks two independent signals:

- `text`: query text is identical after Unicode NFKC normalization, casefolding,
  and collapsing whitespace.
- `anchor`: at least one anchor points to a file with the same whole-file
  SHA-256 digest and has an inclusive line range that intersects a development
  anchor. Repository IDs and file paths do not affect this comparison.

One overlap record is emitted per validation/development query pair, with
`reasons` containing `text`, `anchor`, or both. This is a conservative signal,
not a semantic similarity detector; absence of these matches is not proof that
queries or judgments are independent.

`ready` is true only when the validation plan declares at least two
repositories, at least one development plan is supplied, every validation
query has a human approval and no changes request, and no text or anchor
overlaps are found. The report lists explicit blockers for unmet conditions.
It represents readiness according to this declared workflow only. `human` and
`agent` are submitted labels; the audit does not verify those declarations.
Test fixtures use fictional identities and decisions and are not evidence of
actual review.

## CLI

```sh
python tools/repository_validation.py \
  --plan frozen-validation-plan.json \
  --review review.json \
  --development-plan frozen-development-plan.json \
  --output validation-audit.json
```

Repeat `--development-plan` to compare against multiple development plans; it
is optional, but without it the audit is not ready. Inputs are read with the
strict JSON loader from `repository_eval`. The output is created only if the
path does not already exist. A valid audit exits with status 0 even when
`ready` is false; malformed inputs or an existing output path exit with status 2. The result includes a canonical
`review_sha256`, sorted development plan fingerprints, query statuses (including
`pending` and `changes_requested`), overlap records, blockers, readiness, and a
canonical report fingerprint.

This audit is not wired into the existing `repository_eval score` command. That
CLI remains exploratory and is not gated by review audit readiness. Existing
plan, run, scoring, and ranking formats are unchanged.
