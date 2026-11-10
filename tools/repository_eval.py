"""Offline, bounded repository retrieval evaluation (RQ2 format version 1).

Version 1 freezes one repository per plan. Chunk hits are evaluated in their
native units: an anchor is found only when one hit contains its whole span.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys


class EvaluationError(ValueError):
    """Invalid spec, plan, or run input."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(value, required, optional=(), where="object"):
    if not isinstance(value, dict):
        raise EvaluationError(f"{where} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise EvaluationError(f"{where} keys invalid (missing={sorted(missing)}, extra={sorted(extra)})")


def _string(value, where, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise EvaluationError(f"{where} must be {'a nonempty' if nonempty else 'a'} string")


def _integer(value, where, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationError(f"{where} must be an integer >= {minimum}")


def _finite_number(value, where, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)) or value < minimum:
        raise EvaluationError(f"{where} must be a finite number >= {minimum}")


def _relpath(path, where):
    _string(path, where)
    if "\x00" in path or "\\" in path or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise EvaluationError(f"{where} must be a normalized relative POSIX path")
    pp = PurePosixPath(path)
    if any(part in ("", ".", "..") for part in path.split("/")) or pp.as_posix() != path or path == ".":
        raise EvaluationError(f"{where} must be a normalized relative POSIX path")
    return path


def _filter_prefix(path, where):
    _string(path, where)
    core = path[:-1] if path.endswith("/") else path
    _relpath(core, where)
    if "//" in path:
        raise EvaluationError(f"{where} must be a normalized relative POSIX prefix")
    return path


def _unique(items, where):
    if len(items) != len(set(items)):
        raise EvaluationError(f"duplicate {where}")


def _validate_spec(spec):
    _object(spec, ("format_version", "name", "split", "repositories", "queries"), where="spec")
    if type(spec["format_version"]) is not int or spec["format_version"] != 1:
        raise EvaluationError("spec.format_version must be 1")
    _string(spec["name"], "spec.name")
    if spec["split"] not in ("development", "validation"):
        raise EvaluationError("spec.split must be development or validation")
    repos = spec["repositories"]
    if not isinstance(repos, list) or len(repos) != 1:
        raise EvaluationError("version 1 supports exactly one repository per spec")
    _object(repos[0], ("id", "revision"), where="repository")
    _string(repos[0]["id"], "repository.id")
    _string(repos[0]["revision"], "repository.revision")
    qs = spec["queries"]
    if not isinstance(qs, list) or not qs:
        raise EvaluationError("spec.queries must be a nonempty array")
    for q in qs:
        if isinstance(q, dict):
            _string(q.get("id"), "query.id")
    _unique([q.get("id") if isinstance(q, dict) else None for q in qs], "query IDs")
    for qi, q in enumerate(qs):
        _object(q, ("id", "repository", "category", "text", "filter", "anchors"), where=f"query[{qi}]")
        for k in ("id", "repository", "category", "text"):
            _string(q[k], f"query[{qi}].{k}")
        if q["repository"] != repos[0]["id"]:
            raise EvaluationError(f"query[{qi}].repository does not name the single repository")
        f = q["filter"]
        _object(f, (), ("path_prefix",), f"query[{qi}].filter")
        if "path_prefix" in f:
            prefix = _filter_prefix(f["path_prefix"], f"query[{qi}].filter.path_prefix")
            if prefix == ".git" or prefix.startswith(".git/"):
                raise EvaluationError("filters cannot target .git")
        anchors = q["anchors"]
        if not isinstance(anchors, list) or not anchors:
            raise EvaluationError(f"query[{qi}].anchors must be nonempty")
        for a in anchors:
            if isinstance(a, dict):
                _string(a.get("id"), "anchor.id")
        _unique([a.get("id") if isinstance(a, dict) else None for a in anchors], f"query[{qi}] anchor IDs")
        for ai, a in enumerate(anchors):
            _object(a, ("id", "path", "start_line", "end_line"), where=f"query[{qi}].anchor[{ai}]")
            _string(a["id"], "anchor.id")
            path = _relpath(a["path"], "anchor.path")
            if path == ".git" or path.startswith(".git/"):
                raise EvaluationError("anchors cannot target .git")
            _integer(a["start_line"], "anchor.start_line")
            _integer(a["end_line"], "anchor.end_line")
            if a["end_line"] < a["start_line"]:
                raise EvaluationError("anchor.end_line must be >= start_line")
            if "path_prefix" in f and not path.startswith(f["path_prefix"]):
                raise EvaluationError("anchor path is outside query path_prefix")


def freeze(spec, root):
    """Validate spec and snapshot every regular file under root into a plan."""
    _validate_spec(spec)
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise EvaluationError("root must be a directory")
    files = []
    def walk_error(error):
        raise EvaluationError(f"cannot inventory repository: {error}") from error
    for current, dirs, names in os.walk(root_path, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if d != ".git" and not (Path(current) / d).is_symlink())
        for name in sorted(names):
            p = Path(current) / name
            rel = p.relative_to(root_path).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                continue
            if p.is_symlink():
                raise EvaluationError(f"symlink file entry is not allowed: {rel}")
            if not p.is_file():
                continue
            data = p.read_bytes()
            files.append({"repository": spec["repositories"][0]["id"], "path": rel,
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "line_count": data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)})
    index = {f["path"]: f for f in files}
    for q in spec["queries"]:
        for a in q["anchors"]:
            entry = index.get(a["path"])
            if entry is None:
                raise EvaluationError(f"anchor file does not exist: {a['path']}")
            try:
                (root_path / a["path"]).read_bytes().decode("utf-8")
            except UnicodeDecodeError as e:
                raise EvaluationError(f"anchor file is not UTF-8: {a['path']}") from e
            if a["end_line"] > entry["line_count"]:
                raise EvaluationError(f"anchor exceeds file line count: {a['path']}")
    plan = {"format_version": 1, "spec": spec, "spec_sha256": _digest(spec), "files": files}
    plan["fingerprint"] = _digest(plan)
    return plan


def _verify_plan(plan):
    _object(plan, ("format_version", "spec", "spec_sha256", "files", "fingerprint"), where="plan")
    if type(plan["format_version"]) is not int or plan["format_version"] != 1:
        raise EvaluationError("plan.format_version must be 1")
    _validate_spec(plan["spec"])
    if plan["spec_sha256"] != _digest(plan["spec"]):
        raise EvaluationError("plan spec digest mismatch")
    if not isinstance(plan["files"], list):
        raise EvaluationError("plan.files must be an array")
    repo = plan["spec"]["repositories"][0]["id"]
    file_index = {}
    for i, f in enumerate(plan["files"]):
        _object(f, ("repository", "path", "sha256", "line_count"), where=f"plan.files[{i}]")
        if f["repository"] != repo:
            raise EvaluationError("file repository mismatch")
        path = _relpath(f["path"], "file.path")
        if path == ".git" or path.startswith(".git/"):
            raise EvaluationError("plan cannot inventory .git")
        if not isinstance(f["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
            raise EvaluationError("file sha256 must be lowercase hex SHA-256")
        _integer(f["line_count"], "file.line_count", minimum=0)
        if path in file_index:
            raise EvaluationError("duplicate file path")
        file_index[path] = f
    for q in plan["spec"]["queries"]:
        for a in q["anchors"]:
            if a["path"] not in file_index or a["end_line"] > file_index[a["path"]]["line_count"]:
                raise EvaluationError("anchor exceeds frozen file inventory")
    unsigned = {k: v for k, v in plan.items() if k != "fingerprint"}
    if not isinstance(plan["fingerprint"], str) or plan["fingerprint"] != _digest(unsigned):
        raise EvaluationError("plan fingerprint mismatch")
    return file_index


def _run_scores(run, spec, file_index, fingerprint):
    _object(run, ("format_version", "plan_fingerprint", "name", "tool", "configuration", "candidate_k", "results"), where="run")
    if type(run["format_version"]) is not int or run["format_version"] != 1:
        raise EvaluationError("run.format_version must be 1")
    if run["plan_fingerprint"] != fingerprint:
        raise EvaluationError("run plan_fingerprint mismatch")
    _string(run["name"], "run.name")
    _object(run["tool"], ("name", "revision"), where="run.tool")
    _string(run["tool"]["name"], "tool.name")
    _string(run["tool"]["revision"], "tool.revision")
    if not isinstance(run["configuration"], dict):
        raise EvaluationError("run.configuration must be an object")
    try:
        _canonical(run["configuration"])
    except (TypeError, ValueError) as e:
        raise EvaluationError("configuration must be finite JSON") from e
    _integer(run["candidate_k"], "run.candidate_k", minimum=10)
    if not isinstance(run["results"], list):
        raise EvaluationError("run.results must be an array")
    queries = {q["id"]: q for q in spec["queries"]}
    ids = [r.get("query_id") if isinstance(r, dict) else None for r in run["results"]]
    for qid in ids:
        _string(qid, "result.query_id")
    _unique(ids, "result query IDs")
    if set(ids) != set(queries):
        raise EvaluationError("results must cover each query exactly once")
    scores = {}
    for result in run["results"]:
        _object(result, ("query_id", "status", "hits"), ("error", "elapsed_ms"), "result")
        qid = result["query_id"]
        if qid not in queries:
            raise EvaluationError("unknown query_id")
        status = result["status"]
        if status not in ("ok", "failed", "unsupported", "timeout"):
            raise EvaluationError("invalid result status")
        if not isinstance(result["hits"], list):
            raise EvaluationError("result.hits must be an array")
        if status == "ok":
            if "error" in result:
                raise EvaluationError("ok result must not include error")
        else:
            _string(result.get("error"), "non-ok result.error")
            if result["hits"]:
                raise EvaluationError("non-ok results must have zero hits")
        if len(result["hits"]) > run["candidate_k"]:
            raise EvaluationError("hits exceed candidate_k")
        if "elapsed_ms" in result:
            _finite_number(result["elapsed_ms"], "elapsed_ms")
        q = queries[qid]
        filt = q["filter"].get("path_prefix")
        hit_spans = set()
        anchor_ranks = {a["id"]: None for a in q["anchors"]}
        first_any = None
        for rank, h in enumerate(result["hits"], 1):
            _object(h, ("path", "start_line", "end_line"), where="hit")
            path = _relpath(h["path"], "hit.path")
            _integer(h["start_line"], "hit.start_line")
            _integer(h["end_line"], "hit.end_line")
            if h["end_line"] < h["start_line"]:
                raise EvaluationError("hit.end_line must be >= start_line")
            f = file_index.get(path)
            if f is None or h["end_line"] > f["line_count"]:
                raise EvaluationError("hit path or range outside frozen inventory")
            if filt and not path.startswith(filt):
                raise EvaluationError("hit violates query path_prefix")
            span = (path, h["start_line"], h["end_line"])
            if span in hit_spans:
                raise EvaluationError("duplicate hit span")
            hit_spans.add(span)
            for a in q["anchors"]:
                if path == a["path"] and h["start_line"] <= a["start_line"] and h["end_line"] >= a["end_line"]:
                    if anchor_ranks[a["id"]] is None:
                        anchor_ranks[a["id"]] = rank
                    first_any = rank if first_any is None else first_any
        per_cutoff = {str(k): (sum((anchor_ranks[a["id"]] is not None and anchor_ranks[a["id"]] <= k) for a in q["anchors"]) / len(q["anchors"])) if run["candidate_k"] >= k else None for k in (10, 50, 100)}
        scores[qid] = {"query_id": qid, "category": q["category"], "repository": q["repository"],
                       "status": status, "anchor_ranks": anchor_ranks, "recall_at": per_cutoff,
                       "all_required_at_10": (all(anchor_ranks[a["id"]] is not None and anchor_ranks[a["id"]] <= 10 for a in q["anchors"]) if run["candidate_k"] >= 10 else None),
                       "reciprocal_rank": 1 / first_any if first_any is not None else 0.0,
                       "hits": result["hits"]}
    return scores


def _aggregate(records):
    n = len(records)
    return {"query_count": n,
            "status_counts": {s: sum(r["status"] == s for r in records) for s in ("ok", "failed", "unsupported", "timeout")},
            "coverage": sum(r["status"] == "ok" for r in records) / n if n else None,
            "recall_at_10": sum(r["recall_at"]["10"] for r in records) / n if n else None,
            "recall_at_50": (sum(r["recall_at"]["50"] for r in records) / n if n and all(r["recall_at"]["50"] is not None for r in records) else None),
            "recall_at_100": (sum(r["recall_at"]["100"] for r in records) / n if n and all(r["recall_at"]["100"] is not None for r in records) else None),
            "all_required_at_10": (sum(r["all_required_at_10"] for r in records) / n if n and all(r["all_required_at_10"] is not None for r in records) else None),
            "mrr": sum(r["reciprocal_rank"] for r in records) / n if n else None}


def score(plan, runs):
    """Verify frozen plan, score exact-coverage runs, and compare paired queries."""
    file_index = _verify_plan(plan)
    if not isinstance(runs, list) or not runs:
        raise EvaluationError("at least one run is required")
    if any(not isinstance(r, dict) for r in runs):
        raise EvaluationError("each run must be an object")
    for r in runs:
        _string(r.get("name"), "run.name")
    _unique([r.get("name") for r in runs], "run names")
    spec = plan["spec"]
    run_reports = []
    for run in runs:
        qs = _run_scores(run, spec, file_index, plan["fingerprint"])
        records = [qs[q["id"]] for q in spec["queries"]]
        categories = {}
        repos = {}
        for r in records:
            categories.setdefault(r["category"], []).append(r)
            repos.setdefault(r["repository"], []).append(r)
        run_reports.append({"name": run["name"], "run_sha256": _digest(run), "tool": run["tool"],
                            "configuration": run["configuration"], "candidate_k": run["candidate_k"],
                            "queries": qs, "overall": _aggregate(records),
                            "by_category": {k: _aggregate(v) for k, v in sorted(categories.items())},
                            "by_repository": {k: _aggregate(v) for k, v in sorted(repos.items())}, "raw_run": run})
    baseline = run_reports[0]
    paired = []
    for candidate in run_reports[1:]:
        deltas = {}
        outcomes = {"wins": 0, "ties": 0, "losses": 0}
        for q in spec["queries"]:
            qid = q["id"]
            b, c = baseline["queries"][qid], candidate["queries"][qid]
            br, cr = b["recall_at"]["10"], c["recall_at"]["10"]
            outcomes["wins" if cr > br else "losses" if cr < br else "ties"] += 1
            deltas[qid] = {"recall_at_10": cr - br,
                           "all_required_at_10": c["all_required_at_10"] - b["all_required_at_10"] if b["all_required_at_10"] is not None and c["all_required_at_10"] is not None else None,
                           "mrr": c["reciprocal_rank"] - b["reciprocal_rank"] if candidate["candidate_k"] == baseline["candidate_k"] else None}
        paired.append({"baseline": baseline["name"], "candidate": candidate["name"], "per_query": deltas,
                       "recall_at_10_outcomes": outcomes,
                       "macro_delta": {m: candidate["overall"][m] - baseline["overall"][m] if candidate["overall"][m] is not None and baseline["overall"][m] is not None and (m != "mrr" or candidate["candidate_k"] == baseline["candidate_k"]) else None for m in ("recall_at_10", "all_required_at_10", "mrr")}})
    return {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "spec_sha256": plan["spec_sha256"],
            "runs": run_reports, "paired_comparisons": paired,
            "method": {"anchor_match": "one hit must fully contain one inclusive anchor span; hit unions are not combined",
                       "macro": "unweighted query mean; failures count as zero in retrieval metrics",
                       "cutoffs": "recall@50/@100 are null when candidate_k is below the cutoff",
                       "limitations": ["This offline repository comparison does not establish adoption, market fit, or product superiority.",
                                       "Scores depend on the declared queries, anchors, repository revision, and native hit units.",
                                       "No nDCG is reported because overlapping anchors and chunk units make graded relevance ambiguous."]}}


def _load(path):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise EvaluationError(f"duplicate JSON key: {k}")
            d[k] = v
        return d
    def bad_constant(s):
        raise EvaluationError(f"non-finite JSON number: {s}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise EvaluationError(f"cannot load JSON {path}: {e}") from e


def _write_new(path, value):
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    except FileExistsError as e:
        raise EvaluationError(f"output already exists: {path}") from e


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("freeze")
    fp.add_argument("--spec", required=True); fp.add_argument("--root", required=True); fp.add_argument("--output", required=True)
    sp = sub.add_parser("score")
    sp.add_argument("--plan", required=True); sp.add_argument("--run", required=True, action="append"); sp.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze(_load(args.spec), args.root)
        else:
            result = score(_load(args.plan), [_load(p) for p in args.run])
        _write_new(args.output, result)
        return 0
    except (EvaluationError, OSError, TypeError, ValueError) as e:
        print(f"repository_eval: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
