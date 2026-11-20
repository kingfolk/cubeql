"""Independent review audit for frozen repository validation plans.

This reports workflow readiness from declared human and agent reviews and
checks for exact text or frozen-anchor overlap with development plans. It does
not authenticate reviewers or prove temporal preregistration or semantic
independence.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata

import repository_eval as ev


def _identity(value, where):
    ev._object(value, ("identity", "kind"), where=where)
    ev._string(value["identity"], f"{where}.identity")
    if value["kind"] not in ("human", "agent"):
        raise ev.EvaluationError(f"{where}.kind must be human or agent")


def _query_file_hashes(plan, query):
    repo_id = query["repository"]
    hashes = {}
    for file in plan["files"]:
        if file["repository"] == repo_id:
            hashes[file["path"]] = file["sha256"]
    return hashes


def _normalized_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _overlap_reasons(validation_query, validation_hashes, development_query, development_hashes):
    reasons = []
    if _normalized_text(validation_query["text"]) == _normalized_text(development_query["text"]):
        reasons.append("text")
    found_anchor_overlap = False
    for va in validation_query["anchors"]:
        vhash = validation_hashes.get(va["path"])
        if vhash is None:
            continue
        for da in development_query["anchors"]:
            if (vhash == development_hashes.get(da["path"])
                    and va["start_line"] <= da["end_line"]
                    and da["start_line"] <= va["end_line"]):
                found_anchor_overlap = True
                break
        if found_anchor_overlap:
            break
    if found_anchor_overlap:
        reasons.append("anchor")
    return reasons


def audit(plan, review, development_plans):
    """Validate a review packet and return deterministic audit readiness."""
    ev._verify_plan(plan)
    if plan["spec"]["split"] != "validation":
        raise ev.EvaluationError("plan.spec.split must be validation")
    if not isinstance(development_plans, list):
        raise ev.EvaluationError("development_plans must be an array")

    dev_by_fingerprint = {}
    for i, dev_plan in enumerate(development_plans):
        ev._verify_plan(dev_plan)
        if dev_plan["spec"]["split"] != "development":
            raise ev.EvaluationError(f"development_plans[{i}].spec.split must be development")
        fingerprint = dev_plan["fingerprint"]
        if fingerprint in dev_by_fingerprint:
            raise ev.EvaluationError("duplicate development plan fingerprint")
        dev_by_fingerprint[fingerprint] = dev_plan

    ev._object(review, ("format_version", "plan_fingerprint", "queries"), where="review")
    if type(review["format_version"]) is not int or review["format_version"] != 1:
        raise ev.EvaluationError("review.format_version must be 1")
    if review["plan_fingerprint"] != plan["fingerprint"]:
        raise ev.EvaluationError("review plan_fingerprint mismatch")
    if not isinstance(review["queries"], list):
        raise ev.EvaluationError("review.queries must be an array")

    plan_queries = plan["spec"]["queries"]
    plan_ids = [q["id"] for q in plan_queries]
    review_by_id = {}
    identity_kinds = {}
    for qi, rq in enumerate(review["queries"]):
        ev._object(rq, ("query_id", "author", "reviews"), where=f"review.queries[{qi}]")
        ev._string(rq["query_id"], f"review.queries[{qi}].query_id")
        if rq["query_id"] in review_by_id:
            raise ev.EvaluationError("duplicate review query_id")
        if rq["query_id"] not in set(plan_ids):
            raise ev.EvaluationError(f"unknown review query_id: {rq['query_id']}")
        _identity(rq["author"], f"review.queries[{qi}].author")
        author_id = rq["author"]["identity"]
        author_kind = rq["author"]["kind"]
        prior_kind = identity_kinds.setdefault(author_id, author_kind)
        if prior_kind != author_kind:
            raise ev.EvaluationError(f"identity has inconsistent kind: {author_id}")
        if not isinstance(rq["reviews"], list):
            raise ev.EvaluationError(f"review.queries[{qi}].reviews must be an array")
        reviewer_ids = set()
        for ri, reviewer in enumerate(rq["reviews"]):
            ev._object(reviewer, ("identity", "kind", "decision", "notes"),
                       where=f"review.queries[{qi}].reviews[{ri}]")
            ev._string(reviewer["identity"], f"review.queries[{qi}].reviews[{ri}].identity")
            ev._string(reviewer["notes"], f"review.queries[{qi}].reviews[{ri}].notes")
            if reviewer["kind"] not in ("human", "agent"):
                raise ev.EvaluationError(f"review.queries[{qi}].reviews[{ri}].kind must be human or agent")
            if reviewer["decision"] not in ("approved", "changes_requested"):
                raise ev.EvaluationError(f"review.queries[{qi}].reviews[{ri}].decision must be approved or changes_requested")
            identity = reviewer["identity"]
            if identity in reviewer_ids:
                raise ev.EvaluationError(f"duplicate reviewer identity for query {rq['query_id']}")
            reviewer_ids.add(identity)
            if identity == author_id:
                raise ev.EvaluationError(f"reviewer must differ from author for query {rq['query_id']}")
            prior_kind = identity_kinds.setdefault(identity, reviewer["kind"])
            if prior_kind != reviewer["kind"]:
                raise ev.EvaluationError(f"identity has inconsistent kind: {identity}")
        review_by_id[rq["query_id"]] = rq
    missing_ids = set(plan_ids) - review_by_id.keys()
    if missing_ids:
        raise ev.EvaluationError(f"review query coverage mismatch (missing={sorted(missing_ids)})")

    statuses = []
    for query in plan_queries:
        decisions = review_by_id[query["id"]]["reviews"]
        if any(r["decision"] == "changes_requested" for r in decisions):
            status = "changes_requested"
        elif any(r["kind"] == "human" and r["decision"] == "approved" for r in decisions):
            status = "approved"
        else:
            status = "pending"
        statuses.append({"query_id": query["id"], "status": status})

    overlaps = []
    for query in plan_queries:
        validation_hashes = _query_file_hashes(plan, query)
        for dev_fingerprint in sorted(dev_by_fingerprint):
            dev_plan = dev_by_fingerprint[dev_fingerprint]
            for dev_query in dev_plan["spec"]["queries"]:
                reasons = _overlap_reasons(
                    query, validation_hashes, dev_query,
                    _query_file_hashes(dev_plan, dev_query))
                if reasons:
                    overlaps.append({
                        "query_id": query["id"],
                        "development_plan_fingerprint": dev_fingerprint,
                        "development_query_id": dev_query["id"],
                        "reasons": reasons,
                    })

    blockers = []
    if len(plan["spec"]["repositories"]) < 2:
        blockers.append("validation plan must include at least two repositories")
    if not dev_by_fingerprint:
        blockers.append("at least one development plan is required")
    for item in statuses:
        if item["status"] != "approved":
            blockers.append(f"validation query {item['query_id']} is {item['status']}")
    for item in overlaps:
        blockers.append(
            f"validation query {item['query_id']} overlaps development query "
            f"{item['development_query_id']} in plan {item['development_plan_fingerprint']} "
            f"({', '.join(item['reasons'])})")

    unsigned = {
        "format_version": 1,
        "plan_fingerprint": plan["fingerprint"],
        "review_sha256": ev._digest(review),
        "development_plan_fingerprints": sorted(dev_by_fingerprint),
        "queries": statuses,
        "overlaps": overlaps,
        "blockers": blockers,
        "ready": not blockers,
    }
    unsigned["fingerprint"] = ev._digest(unsigned)
    return unsigned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--development-plan", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(ev._load(args.plan), ev._load(args.review),
                       [ev._load(path) for path in args.development_plan])
        ev._write_new(args.output, result)
        return 0
    except (ev.EvaluationError, OSError, TypeError, ValueError) as e:
        print(f"repository_validation: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
