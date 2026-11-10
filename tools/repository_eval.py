"""Offline, bounded repository retrieval evaluation (RQ2 format versions 1 and 2).

Version 1 freezes one repository per plan; version 2 freezes several distinct
repository roots with repository-scoped paths. Chunk hits are evaluated in their
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
    version = spec["format_version"]
    if type(version) is not int or version not in (1, 2):
        raise EvaluationError("spec.format_version must be 1 or 2")
    _string(spec["name"], "spec.name")
    if spec["split"] not in ("development", "validation"):
        raise EvaluationError("spec.split must be development or validation")
    repos = spec["repositories"]
    if not isinstance(repos, list) or not repos or (version == 1 and len(repos) != 1):
        raise EvaluationError("version 1 supports exactly one repository per spec" if version == 1 else "spec.repositories must be a nonempty array")
    repo_ids = []
    for i, repo in enumerate(repos):
        _object(repo, ("id", "revision"), where=f"repository[{i}]")
        _string(repo["id"], "repository.id")
        _string(repo["revision"], "repository.revision")
        repo_ids.append(repo["id"])
    _unique(repo_ids, "repository IDs")
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
        if version == 1 and q["repository"] != repo_ids[0]:
            raise EvaluationError(f"query[{qi}].repository does not name the single repository")
        if version == 2 and q["repository"] not in repo_ids:
            raise EvaluationError(f"query[{qi}].repository does not name a declared repository")
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
    if version == 2 and {q["repository"] for q in qs} != set(repo_ids):
        raise EvaluationError("every declared repository must have at least one query")


def freeze(spec, root):
    """Validate spec and snapshot every regular file under root into a plan."""
    _validate_spec(spec)
    version = spec["format_version"]
    if version == 1:
        if isinstance(root, dict):
            raise EvaluationError("version 1 root must be a path")
        roots = {spec["repositories"][0]["id"]: root}
    else:
        if not isinstance(root, dict):
            raise EvaluationError("version 2 roots must be a repository-to-path object")
        repo_ids = {r["id"] for r in spec["repositories"]}
        if set(root) != repo_ids:
            raise EvaluationError("roots must name exactly the declared repositories")
        for repo_id, path in root.items():
            _string(repo_id, "root repository id")
            if not isinstance(path, (str, Path)) or not str(path).strip():
                raise EvaluationError(f"root for {repo_id} must be a nonempty path")
        roots = root
    root_paths = {}
    for repo in spec["repositories"]:
        repo_id = repo["id"]
        try:
            root_path = Path(roots[repo_id]).resolve(strict=True)
        except OSError as e:
            raise EvaluationError(f"root for {repo_id} cannot be resolved: {e}") from e
        if not root_path.is_dir():
            raise EvaluationError(f"root for {repo_id} must be a directory")
        root_paths[repo_id] = root_path
    if version == 2:
        identities = {(p.stat().st_dev, p.stat().st_ino) for p in root_paths.values()}
        if len(identities) != len(root_paths):
            raise EvaluationError("repository roots must resolve to distinct directories")
    files = []
    def walk_error(error):
        raise EvaluationError(f"cannot inventory repository: {error}") from error
    for repo in spec["repositories"]:
        repo_id = repo["id"]
        root_path = root_paths[repo_id]
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
                files.append({"repository": repo_id, "path": rel,
                              "sha256": hashlib.sha256(data).hexdigest(),
                              "line_count": data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)})
    index = {(f["repository"], f["path"]): f for f in files}
    for q in spec["queries"]:
        for a in q["anchors"]:
            repo_id = q["repository"]
            entry = index.get((repo_id, a["path"]))
            if entry is None:
                raise EvaluationError(f"anchor file does not exist: {a['path']}")
            try:
                (root_paths[repo_id] / a["path"]).read_bytes().decode("utf-8")
            except UnicodeDecodeError as e:
                raise EvaluationError(f"anchor file is not UTF-8: {a['path']}") from e
            if a["end_line"] > entry["line_count"]:
                raise EvaluationError(f"anchor exceeds file line count: {a['path']}")
    plan = {"format_version": version, "spec": spec, "spec_sha256": _digest(spec), "files": files}
    plan["fingerprint"] = _digest(plan)
    return plan


def _verify_plan(plan):
    _object(plan, ("format_version", "spec", "spec_sha256", "files", "fingerprint"), where="plan")
    if type(plan["format_version"]) is not int or plan["format_version"] not in (1, 2):
        raise EvaluationError("plan.format_version must be 1 or 2")
    _validate_spec(plan["spec"])
    if plan["format_version"] != plan["spec"]["format_version"]:
        raise EvaluationError("plan/spec format_version mismatch")
    if plan["spec_sha256"] != _digest(plan["spec"]):
        raise EvaluationError("plan spec digest mismatch")
    if not isinstance(plan["files"], list):
        raise EvaluationError("plan.files must be an array")
    repo_ids = {r["id"] for r in plan["spec"]["repositories"]}
    file_index = {}
    for i, f in enumerate(plan["files"]):
        _object(f, ("repository", "path", "sha256", "line_count"), where=f"plan.files[{i}]")
        _string(f["repository"], "file.repository")
        if f["repository"] not in repo_ids:
            raise EvaluationError("unknown file repository")
        path = _relpath(f["path"], "file.path")
        if path == ".git" or path.startswith(".git/"):
            raise EvaluationError("plan cannot inventory .git")
        if not isinstance(f["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
            raise EvaluationError("file sha256 must be lowercase hex SHA-256")
        _integer(f["line_count"], "file.line_count", minimum=0)
        key = (f["repository"], path) if plan["format_version"] == 2 else path
        if key in file_index:
            raise EvaluationError("duplicate repository/path pair" if plan["format_version"] == 2 else "duplicate file path")
        file_index[key] = f
    for q in plan["spec"]["queries"]:
        for a in q["anchors"]:
            key = (q["repository"], a["path"]) if plan["format_version"] == 2 else a["path"]
            if key not in file_index or a["end_line"] > file_index[key]["line_count"]:
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
            file_key = (q["repository"], path) if spec["format_version"] == 2 else path
            f = file_index.get(file_key)
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


def _repository_macro(repo_scores):
    """Equal-weight mean across repository-level query macros."""
    n = len(repo_scores)
    metrics = ("coverage", "recall_at_10", "recall_at_50", "recall_at_100", "all_required_at_10", "mrr")
    result = {key: (sum(r[key] for r in repo_scores.values()) / n
                    if n and all(r[key] is not None for r in repo_scores.values()) else None)
              for key in metrics}
    result.update(repository_count=n, query_count=sum(r["query_count"] for r in repo_scores.values()))
    return result


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
        run_report = {"name": run["name"], "run_sha256": _digest(run), "tool": run["tool"],
                            "configuration": run["configuration"], "candidate_k": run["candidate_k"],
                            "queries": qs, "overall": _aggregate(records),
                            "by_category": {k: _aggregate(v) for k, v in sorted(categories.items())},
                            "by_repository": {k: _aggregate(v) for k, v in sorted(repos.items())}, "raw_run": run}
        if spec["format_version"] == 2:
            run_report["repository_macro"] = _repository_macro(run_report["by_repository"])
        run_reports.append(run_report)
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
        paired_report = {"baseline": baseline["name"], "candidate": candidate["name"], "per_query": deltas,
                         "recall_at_10_outcomes": outcomes,
                         "macro_delta": {m: candidate["overall"][m] - baseline["overall"][m] if candidate["overall"][m] is not None and baseline["overall"][m] is not None and (m != "mrr" or candidate["candidate_k"] == baseline["candidate_k"]) else None for m in ("recall_at_10", "all_required_at_10", "mrr")}}
        if spec["format_version"] == 2:
            repo_deltas = {}
            for repo_id in sorted(baseline["by_repository"]):
                base_repo = baseline["by_repository"][repo_id]
                cand_repo = candidate["by_repository"][repo_id]
                repo_outcomes = {"wins": 0, "ties": 0, "losses": 0}
                for q in spec["queries"]:
                    if q["repository"] != repo_id:
                        continue
                    br = baseline["queries"][q["id"]]["recall_at"]["10"]
                    cr = candidate["queries"][q["id"]]["recall_at"]["10"]
                    repo_outcomes["wins" if cr > br else "losses" if cr < br else "ties"] += 1
                repo_deltas[repo_id] = {"macro_delta": {
                    m: cand_repo[m] - base_repo[m] if cand_repo[m] is not None and base_repo[m] is not None and (m != "mrr" or candidate["candidate_k"] == baseline["candidate_k"]) else None
                    for m in ("recall_at_10", "all_required_at_10", "mrr")},
                    "recall_at_10_outcomes": repo_outcomes}
            paired_report["by_repository"] = repo_deltas
            paired_report["repository_macro_delta"] = {
                m: candidate["repository_macro"][m] - baseline["repository_macro"][m]
                if candidate["repository_macro"][m] is not None and baseline["repository_macro"][m] is not None and (m != "mrr" or candidate["candidate_k"] == baseline["candidate_k"]) else None
                for m in ("recall_at_10", "all_required_at_10", "mrr")}
        paired.append(paired_report)
    report = {"format_version": plan["format_version"], "plan_fingerprint": plan["fingerprint"], "spec_sha256": plan["spec_sha256"],
            "runs": run_reports, "paired_comparisons": paired,
            "method": {"anchor_match": "one hit must fully contain one inclusive anchor span; hit unions are not combined",
                       "macro": "unweighted query mean; failures count as zero in retrieval metrics",
                       "cutoffs": "recall@50/@100 are null when candidate_k is below the cutoff",
                       "limitations": ["This offline repository comparison does not establish adoption, market fit, or product superiority.",
                                       "Scores depend on the declared queries, anchors, repository revision, and native hit units.",
                                       "No nDCG is reported because overlapping anchors and chunk units make graded relevance ambiguous."]}}
    if spec["format_version"] == 2:
        report["method"]["repository_macro"] = "unweighted mean of repository-level query macros; each repository has equal weight"
    return report


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
    fp.add_argument("--spec", required=True)
    roots_group = fp.add_mutually_exclusive_group(required=True)
    roots_group.add_argument("--root")
    roots_group.add_argument("--roots")
    fp.add_argument("--output", required=True)
    sp = sub.add_parser("score")
    sp.add_argument("--plan", required=True); sp.add_argument("--run", required=True, action="append"); sp.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            loaded_spec = _load(args.spec)
            if args.roots:
                root_map = _load(args.roots)
                if not isinstance(root_map, dict):
                    raise EvaluationError("roots JSON must be an object")
                for repo_id, path in root_map.items():
                    _string(repo_id, "root repository id")
                    _string(path, f"root for {repo_id}")
                result = freeze(loaded_spec, root_map)
            else:
                result = freeze(loaded_spec, args.root)
        else:
            result = score(_load(args.plan), [_load(p) for p in args.run])
        _write_new(args.output, result)
        return 0
    except (EvaluationError, OSError, TypeError, ValueError) as e:
        print(f"repository_eval: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
