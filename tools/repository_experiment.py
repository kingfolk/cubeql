"""Freeze declared repository experiment arms and score their complete paired runs.

Offline receipts bind supplied inputs; they do not authenticate people, execution
conditions, or whether a protocol existed before results were observed.
"""
from __future__ import annotations

import argparse
import copy
import sys

import repository_eval as ev
from repository_validation import audit


def _spec(spec):
    ev._object(spec, ("format_version", "name", "arms"), where="experiment spec")
    if type(spec["format_version"]) is not int or spec["format_version"] != 1:
        raise ev.EvaluationError("experiment spec.format_version must be 1")
    ev._string(spec["name"], "experiment name")
    if not isinstance(spec["arms"], list) or len(spec["arms"]) < 2:
        raise ev.EvaluationError("at least two experiment arms are required")
    names, conditions = [], []
    for arm in spec["arms"]:
        ev._object(arm, ("name", "tool", "configuration", "candidate_k"), where="arm")
        ev._string(arm["name"], "arm.name")
        ev._object(arm["tool"], ("name", "revision"), where="arm.tool")
        for key in ("name", "revision"):
            ev._string(arm["tool"][key], "arm.tool." + key)
        ev._integer(arm["candidate_k"], "arm.candidate_k", minimum=10)
        if not isinstance(arm["configuration"], dict):
            raise ev.EvaluationError("arm.configuration must be an object")
        for key in ("model_identity", "corpus_identity", "chunking_identity", "filter_policy"):
            ev._string(arm["configuration"].get(key), "arm.configuration." + key)
        names.append(arm["name"])
        conditions.append(ev._digest({k: v for k, v in arm.items() if k != "name"}))
    ev._unique(names, "arm names")
    ev._unique(conditions, "arm conditions")


def freeze(plan, review, development_plans, spec):
    """Bind verified review, source plans, and ordered arms (first is baseline)."""
    _spec(spec)
    reviewed = audit(plan, review, development_plans)
    if not reviewed["ready"]:
        raise ev.EvaluationError("validation review is not ready: " + str(reviewed["blockers"]))
    receipt = copy.deepcopy({"format_version": 1, "plan": plan, "review": review,
                             "development_plans": development_plans, "spec": spec,
                             "audit": reviewed})
    receipt["fingerprint"] = ev._digest(receipt)
    return receipt


def verify(protocol):
    ev._object(protocol, ("format_version", "plan", "review", "development_plans", "spec", "audit", "fingerprint"), where="protocol")
    if type(protocol["format_version"]) is not int or protocol["format_version"] != 1:
        raise ev.EvaluationError("protocol.format_version must be 1")
    rebuilt = freeze(protocol["plan"], protocol["review"], protocol["development_plans"], protocol["spec"])
    # Canonical JSON comparison distinguishes booleans/integers and catches stale audit fields.
    if ev._canonical(rebuilt) != ev._canonical(protocol):
        raise ev.EvaluationError("protocol fingerprint or embedded audit mismatch")


def score(protocol, runs):
    """Require all declared conditions and retain failed queries in denominators."""
    verify(protocol)
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise ev.EvaluationError("runs must be an array of objects")
    for run in runs:
        ev._string(run.get("name"), "run.name")
    ev._unique([run["name"] for run in runs], "run names")
    indexed = {run["name"]: run for run in runs}
    arms = protocol["spec"]["arms"]
    if set(indexed) != {arm["name"] for arm in arms}:
        raise ev.EvaluationError("runs must cover exactly the frozen experiment arms")
    ordered = []
    for arm in arms:
        run = indexed[arm["name"]]
        for key in ("tool", "configuration", "candidate_k"):
            if key not in run or ev._canonical(run[key]) != ev._canonical(arm[key]):
                raise ev.EvaluationError(f"run {arm['name']} differs from frozen {key}")
        ordered.append(run)
    report = ev.score(protocol["plan"], ordered)
    timings = {}
    for run in ordered:
        observed = [r["elapsed_ms"] for r in run["results"] if "elapsed_ms" in r]
        total = sum(observed)
        ev._finite_number(total, "aggregate elapsed_ms")
        timings[run["name"]] = {
            "observed_query_count": len(observed), "expected_query_count": len(run["results"]),
            "observed_sum_ms": total,
            "complete_sum_ms": total if len(observed) == len(run["results"]) else None,
        }
    return {"format_version": 1, "protocol_fingerprint": protocol["fingerprint"],
            "evaluation": report, "declared_elapsed_ms": timings,
            "limitations": ["Review identities and execution conditions are supplied declarations, not authenticated facts.",
                            "A hash receipt does not prove temporal preregistration or absence of semantic leakage.",
                            "Elapsed times are supplied per-query observations; sums are not process wall time or controlled speed comparisons."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("freeze")
    for key in ("plan", "review", "spec", "output"):
        fp.add_argument("--" + key, required=True)
    fp.add_argument("--development-plan", action="append", default=[])
    sp = sub.add_parser("score")
    sp.add_argument("--protocol", required=True)
    sp.add_argument("--run", action="append", required=True)
    sp.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze(ev._load(args.plan), ev._load(args.review),
                            [ev._load(p) for p in args.development_plan], ev._load(args.spec))
        else:
            result = score(ev._load(args.protocol), [ev._load(p) for p in args.run])
        ev._write_new(args.output, result)
        return 0
    except (ev.EvaluationError, OSError, TypeError, ValueError) as error:
        print(f"repository_experiment: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
