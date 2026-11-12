"""Import raw offline Semble result JSON into the repository evaluator format.

This adapter never invokes Semble. It validates a user-supplied manifest and
preserves each supplied native payload byte-for-byte as evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import repository_eval as ev


class PeerImportError(ev.EvaluationError):
    """Invalid Semble manifest, payload, or output request."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_load_bytes(data: bytes, where: str):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise PeerImportError(f"{where}: duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def bad_constant(token):
        raise PeerImportError(f"{where}: non-finite JSON number: {token}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PeerImportError(f"{where}: invalid strict UTF-8 JSON: {exc}") from exc


def _object(value, required, optional=(), where="object"):
    if not isinstance(value, dict):
        raise PeerImportError(f"{where} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise PeerImportError(f"{where} keys invalid (missing={sorted(missing)}, extra={sorted(extra)})")


def _string(value, where, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise PeerImportError(f"{where} must be {'a nonempty' if nonempty else 'a'} string")


def _safe_payload_path(base: Path, value: str) -> Path:
    _string(value, "result.payload")
    if "\x00" in value or "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise PeerImportError("result.payload must be a normalized relative POSIX path")
    pp = PurePosixPath(value)
    if value == "." or pp.as_posix() != value or any(part in ("", ".", "..") for part in value.split("/")):
        raise PeerImportError("result.payload must be a normalized relative POSIX path")
    path = base / Path(*value.split("/"))
    # Reject symlinks at any level, even if they resolve back inside the manifest tree.
    cursor = base
    for part in value.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise PeerImportError(f"result.payload cannot traverse a symlink: {value}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise PeerImportError(f"result.payload escapes its manifest directory or is missing: {value}") from exc
    if not resolved.is_file():
        raise PeerImportError(f"result.payload must name a regular file: {value}")
    return resolved


def _validate_manifest(manifest, plan, manifest_path: Path):
    _object(manifest, ("format_version", "plan_fingerprint", "name", "tool", "configuration", "candidate_k", "results"), where="manifest")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise PeerImportError("manifest.format_version must be 1")
    if manifest["plan_fingerprint"] != plan["fingerprint"]:
        raise PeerImportError("manifest plan_fingerprint does not match frozen plan")
    _string(manifest["name"], "manifest.name")
    _object(manifest["tool"], ("name", "revision"), where="manifest.tool")
    if manifest["tool"]["name"] != "semble":
        raise PeerImportError("manifest.tool.name must be 'semble'")
    _string(manifest["tool"]["revision"], "manifest.tool.revision")
    if not isinstance(manifest["configuration"], dict):
        raise PeerImportError("manifest.configuration must be an object")
    try:
        ev._canonical(manifest["configuration"])
    except (TypeError, ValueError) as exc:
        raise PeerImportError("manifest.configuration must be finite JSON") from exc
    ev._integer(manifest["candidate_k"], "manifest.candidate_k", minimum=10)
    if not isinstance(manifest["results"], list):
        raise PeerImportError("manifest.results must be an array")
    queries = {q["id"]: q for q in plan["spec"]["queries"]}
    ids = []
    rows = []
    base = manifest_path.parent.resolve(strict=True)
    for i, row in enumerate(manifest["results"]):
        _object(row, ("query_id", "query_text", "applied_filter", "status"), ("payload", "error"), f"manifest.results[{i}]")
        _string(row["query_id"], f"manifest.results[{i}].query_id")
        ids.append(row["query_id"])
        q = queries.get(row["query_id"])
        if q is None:
            raise PeerImportError(f"unknown query_id: {row['query_id']}")
        if row["query_text"] != q["text"]:
            raise PeerImportError(f"query_text mismatch for {row['query_id']}")
        if not isinstance(row["applied_filter"], dict) or row["applied_filter"] != q["filter"]:
            raise PeerImportError(f"applied_filter must exactly match the frozen filter object for {row['query_id']}")
        status = row["status"]
        if status not in ("ok", "failed", "timeout", "unsupported"):
            raise PeerImportError(f"invalid status for {row['query_id']}")
        if status == "ok":
            if "error" in row:
                raise PeerImportError(f"ok result must not contain error: {row['query_id']}")
            if "payload" not in row:
                raise PeerImportError(f"ok result requires payload: {row['query_id']}")
        else:
            _string(row.get("error"), f"manifest.results[{i}].error")
        payload_path = _safe_payload_path(base, row["payload"]) if "payload" in row else None
        payload_bytes = payload_path.read_bytes() if payload_path else None
        rows.append((row, payload_path, payload_bytes))
    ev._unique(ids, "manifest query IDs")
    if set(ids) != set(queries):
        raise PeerImportError("manifest results must cover every frozen query exactly once")
    return rows


def _native_hits(payload_bytes: bytes, query_text: str, candidate_k: int, query_id: str):
    payload = _strict_load_bytes(payload_bytes, f"payload for {query_id}")
    if payload == {"error": "No results found."}:
        return []
    if not isinstance(payload, dict):
        raise PeerImportError(f"payload for {query_id} must be an object")
    if "error" in payload:
        raise PeerImportError(f"unrecognized native error payload for {query_id}")
    _object(payload, ("query", "results"), ("repos",), f"payload for {query_id}")
    if payload["query"] != query_text:
        raise PeerImportError(f"native query echo mismatch for {query_id}")
    if "repos" in payload and (not isinstance(payload["repos"], dict) or payload["repos"]):
        raise PeerImportError(f"native repos map must be empty for {query_id}")
    if not isinstance(payload["results"], list):
        raise PeerImportError(f"native results must be an array for {query_id}")
    if len(payload["results"]) > candidate_k:
        raise PeerImportError(f"native results exceed candidate_k for {query_id}")
    hits = []
    for i, hit in enumerate(payload["results"]):
        _object(hit, ("file_path", "start_line", "end_line", "score"), ("content",), f"native result {i} for {query_id}")
        path = hit["file_path"]
        try:
            ev._relpath(path, f"native result {i}.file_path")
        except ev.EvaluationError as exc:
            raise PeerImportError(f"invalid native file_path for {query_id}: {exc}") from exc
        try:
            ev._integer(hit["start_line"], "native start_line")
            ev._integer(hit["end_line"], "native end_line")
        except ev.EvaluationError as exc:
            raise PeerImportError(f"invalid native line span for {query_id}: {exc}") from exc
        if hit["end_line"] < hit["start_line"]:
            raise PeerImportError(f"native line span is reversed for {query_id}")
        score = hit["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or (isinstance(score, float) and not math.isfinite(score)):
            raise PeerImportError(f"native score must be finite for {query_id}")
        if "content" in hit and not isinstance(hit["content"], str):
            raise PeerImportError(f"native content must be a string when present for {query_id}")
        hits.append({"path": path, "start_line": hit["start_line"], "end_line": hit["end_line"]})
    return hits


def import_semble(plan, manifest_path, output, baselines=()):
    """Validate and import a manifest, writing a new evidence-preserving bundle."""
    ev._verify_plan(plan)
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_load_bytes(manifest_bytes, "manifest")
    rows = _validate_manifest(manifest, plan, manifest_path)
    baseline_runs = []
    baseline_sources = []
    for baseline_path in baselines:
        path = Path(baseline_path).resolve(strict=True)
        raw = path.read_bytes()
        run = _strict_load_bytes(raw, f"baseline {path}")
        baseline_runs.append(run)
        baseline_sources.append({"path": str(path), "sha256": _sha256(raw)})
    results = []
    for row, payload_path, payload_bytes in rows:
        rec = {"query_id": row["query_id"], "status": row["status"], "hits": []}
        if row["status"] == "ok":
            rec["hits"] = _native_hits(payload_bytes, row["query_text"], manifest["candidate_k"], row["query_id"])
        else:
            rec["error"] = row["error"]
        results.append(rec)
    run = {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "name": manifest["name"],
           "tool": manifest["tool"], "configuration": manifest["configuration"],
           "candidate_k": manifest["candidate_k"], "results": results}
    # Score everything before making the destination, so malformed imports never look partial/valid.
    try:
        report = ev.score(plan, baseline_runs + [run])
    except ev.EvaluationError as exc:
        raise PeerImportError(f"run or baseline failed evaluator validation: {exc}") from exc
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise PeerImportError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    # Keep the exact input manifest bytes and a canonical frozen-plan snapshot
    # beside the import so the bundle remains usable without its source paths.
    (output / "manifest.json").write_bytes(manifest_bytes)
    plan_bytes = ev._canonical(plan) + b"\n"
    (output / "plan.json").write_bytes(plan_bytes)
    evidence = []
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    for i, (row, payload_path, payload_bytes) in enumerate(rows, 1):
        if payload_bytes is None:
            continue
        dest_rel = f"evidence/{i:06d}.payload"
        dest = output / dest_rel
        with dest.open("xb") as stream:
            stream.write(payload_bytes)
        evidence.append({"query_id": row["query_id"], "manifest_path": row["payload"],
                         "copied_path": dest_rel, "sha256": _sha256(payload_bytes),
                         "bytes": len(payload_bytes), "parsed_as_native_json": row["status"] == "ok"})
    provenance = {
        "format_version": 1, "plan_fingerprint": plan["fingerprint"], "name": manifest["name"],
        "manifest": {"path": str(manifest_path), "copied_path": "manifest.json", "sha256": _sha256(manifest_bytes)},
        "plan": {"copied_path": "plan.json", "sha256": _sha256(plan_bytes)},
        "tool": manifest["tool"], "configuration": manifest["configuration"],
        "candidate_k": manifest["candidate_k"], "evidence": evidence,
        "baselines": baseline_sources,
        "limits": ["Snapshot/source/model/precision/budgets are declared configuration, not independently attested.",
                   "The manifest's applied_filter declaration is checked against the frozen plan; execution enforcement cannot be attested.",
                   "Scores are preserved in raw payload evidence but are not used by the repository evaluator.",
                   "This is an offline import; it does not execute Semble or establish a real peer-quality result."]}
    ev._write_new(output / "run.json", run)
    ev._write_new(output / "report.json", report)
    ev._write_new(output / "provenance.json", provenance)
    return {"run": run, "report": report, "provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        plan = ev._load(args.plan)
        result = import_semble(plan, args.manifest, args.output, args.baseline)
        print(json.dumps({"run": result["run"]["name"], "output": str(Path(args.output).resolve())}, ensure_ascii=False))
        return 0
    except (ev.EvaluationError, OSError, TypeError, ValueError) as exc:
        print(f"repository_peer: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
