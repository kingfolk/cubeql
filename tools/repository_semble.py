"""Collect native, offline Semble retrieval results for a frozen repository plan."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
import traceback

import repository_collect as rc
import repository_eval as ev
import repository_peer as peer
from repository_mcp import JSONLineSession, REQUEST_TIMING_BOUNDARY


class SembleCollectionError(ValueError):
    """Invalid collection input or unverifiable native result."""


def _failure_status(exc: Exception) -> str:
    message = str(exc).lower()
    return "timeout" if "deadline exceeded" in message or "timed out" in message or "timeout" in message else "failed"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _worker(source: str, root: str, model_dir: str, cache_dir: str) -> int:
    """JSON-RPC stdio worker; all provider output is redirected to stderr."""
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SEMBLE_CACHE_LOCATION"] = str(Path(cache_dir).resolve())
    os.environ["SEMBLE_INDEX_WORKERS"] = "0"
    sys.path.insert(0, str(Path(source).resolve() / "src"))
    index = None
    boot = {}
    def emit(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or not isinstance(req.get("id"), (str, int)):
                raise ValueError("malformed JSON-RPC request")
            method, params, rid = req.get("method"), req.get("params", {}), req["id"]
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
            if method == "initialize":
                if index is not None:
                    raise ValueError("worker is already initialized")
                started = time.monotonic()
                with contextlib.redirect_stdout(sys.stderr):
                    from semble import SembleIndex, ContentType, __version__
                    from semble.utils import format_results
                    index = SembleIndex.from_path(root, model_path=model_dir, content=ContentType.CODE)
                package_versions = {}
                for package in ("semble", "model2vec", "vicinity", "numpy", "pathspec", "orjson", "semble-grammars"):
                    try:
                        package_versions[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError:
                        package_versions[package] = None
                boot.update(format_results=format_results, version=__version__,
                            build_ms=(time.monotonic() - started) * 1000, package_versions=package_versions)
                result = {"protocol_version": 1, "semble_version": __version__,
                          "build_ms": boot["build_ms"], "indexed_files": index.stats.indexed_files,
                          "total_chunks": index.stats.total_chunks, "model_path": str(Path(model_dir).resolve()),
                          "python_version": platform.python_version(), "package_versions": package_versions}
            elif method == "search":
                if index is None:
                    raise ValueError("worker is not initialized")
                query, top_k, prefix = params.get("query"), params.get("top_k"), params.get("path_prefix")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a nonempty string")
                if type(top_k) is not int or top_k < 1:
                    raise ValueError("top_k must be a positive integer")
                if prefix is not None and (not isinstance(prefix, str) or not prefix or "\x00" in prefix):
                    raise ValueError("path_prefix must be null or a nonempty string")
                t0 = time.monotonic()
                with contextlib.redirect_stdout(sys.stderr):
                    if prefix is None:
                        found = index.search(query, top_k=top_k)
                    else:
                        paths = sorted({chunk.file_path for chunk in index.chunks if chunk.file_path.startswith(prefix)})
                        # Empty filter_paths means no filter in Semble. Never search the full corpus here.
                        found = index.search(query, top_k=top_k, filter_paths=paths) if paths else []
                    payload = boot["format_results"](query, found)
                result = {"payload": payload, "search_ms": (time.monotonic() - t0) * 1000,
                          "applied_filter": {"path_prefix": prefix} if prefix is not None else {},
                          "internal_candidate_budget": top_k * 5}
            elif method == "shutdown":
                emit({"jsonrpc": "2.0", "id": rid, "result": {"ok": True}})
                return 0
            else:
                raise ValueError(f"unknown method: {method}")
            emit({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as exc:
            try:
                emit({"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32000,
                      "message": str(exc), "data": traceback.format_exc(limit=8)}})
            except Exception:
                return 1
    return 0


def _validate_inputs(plan, roots, python, source, model_dir, output, top_k, timeout_s, source_revision):
    ev._verify_plan(plan)
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise SembleCollectionError("source_revision must be a nonempty declared revision")
    if type(top_k) is not int or not 10 <= top_k <= 1000:
        raise SembleCollectionError("top_k must be an integer in 10..1000")
    if (not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise SembleCollectionError("timeout_s must be positive")
    if not isinstance(roots, dict) or any(not isinstance(k, str) or not isinstance(v, (str, Path)) for k, v in roots.items()):
        raise SembleCollectionError("roots must map string repository IDs to absolute paths")
    if any(not Path(v).is_absolute() for v in roots.values()):
        raise SembleCollectionError("every repository root must be an absolute path")
    roots = {k: Path(v).resolve(strict=True) for k, v in roots.items()}
    expected = {r["id"] for r in plan["spec"]["repositories"]}
    if set(roots) != expected:
        raise SembleCollectionError("roots must exactly match frozen repository IDs")
    if len(set(roots.values())) != len(roots):
        raise SembleCollectionError("repository roots must resolve to distinct directories")
    root_identities = [(root.stat().st_dev, root.stat().st_ino) for root in roots.values()]
    if len(set(root_identities)) != len(root_identities):
        raise SembleCollectionError("repository roots must have distinct filesystem identities")
    for rid, root in roots.items():
        if not root.is_dir() or rc.source_inventory(root) != {
            f["path"]: {"sha256": f["sha256"], "line_count": f["line_count"]}
            for f in plan["files"] if (f["repository"] == rid if plan["format_version"] == 2 else True)
        }:
            raise SembleCollectionError(f"source inventory does not match frozen plan for {rid}")
    # Keep a virtualenv's executable path in argv; resolving its symlink can bypass pyvenv.cfg.
    python = Path(python).absolute()
    if not python.exists():
        raise SembleCollectionError(f"python interpreter does not exist: {python}")
    source = Path(source).resolve(strict=True)
    model_dir = Path(model_dir).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK) or not source.is_dir() or not model_dir.is_dir():
        raise SembleCollectionError("python must be a file; source and model-dir must be directories")
    source_pkg = source / "src" / "semble"
    if not source_pkg.is_dir():
        raise SembleCollectionError("source must contain src/semble")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise SembleCollectionError(f"output already exists: {output}")
    for protected in [*roots.values(), source, model_dir]:
        if _inside(output, protected):
            raise SembleCollectionError(f"output must be outside protected path: {protected}")
    return roots, python, source, model_dir, output


def collect_semble(plan, roots, python, source, model_dir, output, *, name="semble-native", top_k=10,
                   timeout_s=60, source_revision=""):
    roots, python, source, model_dir, output = _validate_inputs(
        plan, roots, python, source, model_dir, output, top_k, timeout_s, source_revision)
    source_before = rc._fingerprint_path(source)
    model_before = rc._fingerprint_path(model_dir)
    python_before = rc._fingerprint_path(python)
    repo_ids = [r["id"] for r in plan["spec"]["repositories"]]
    q_by_repo = {rid: [q for q in plan["spec"]["queries"] if q["repository"] == rid]
                 for rid in repo_ids}
    base = output.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    payload_dir = output / "payloads"
    payload_dir.mkdir()
    trace_dir = output / "traces"
    trace_dir.mkdir()
    records = {}
    results = []
    manifest_rows = []
    query_records = {}
    for repo_no, rid in enumerate(repo_ids, 1):
        root = roots[rid]
        cache = output / "runtime-cache" / f"repo{repo_no}"
        cache.mkdir(parents=True)
        command = [str(python), "-B", str(Path(__file__).resolve()), "--worker", "--source", str(source),
                   "--root", str(root), "--model-dir", str(model_dir), "--cache-dir", str(cache)]
        status, init, error = "ok", None, None
        try:
            with JSONLineSession(command, output, f"repo{repo_no}", timeout_s=timeout_s,
                                 env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                      "SEMBLE_CACHE_LOCATION": str(cache), "SEMBLE_INDEX_WORKERS": "0"}) as session:
                init = session.request("initialize", {})
                if (not isinstance(init, dict) or type(init.get("protocol_version")) is not int
                        or init["protocol_version"] != 1
                        or isinstance(init.get("build_ms"), bool) or not isinstance(init.get("build_ms"), (int, float))
                        or not math.isfinite(init["build_ms"]) or init["build_ms"] < 0
                        or type(init.get("indexed_files")) is not int or init["indexed_files"] < 0
                        or type(init.get("total_chunks")) is not int or init["total_chunks"] < 0
                        or init.get("model_path") != str(model_dir)
                        or not isinstance(init.get("python_version"), str)
                        or not isinstance(init.get("package_versions"), dict)):
                    raise SembleCollectionError("worker returned invalid initialize metadata")
                records[rid] = {"command": command, "initialize": init}
                for q in q_by_repo[rid]:
                    prefix = q["filter"].get("path_prefix")
                    t0 = time.monotonic()
                    try:
                        response = session.request("search", {"query": q["text"], "top_k": top_k,
                                                               "path_prefix": prefix})
                        wall_ms = (time.monotonic() - t0) * 1000
                        if not isinstance(response, dict):
                            raise SembleCollectionError("worker returned a non-object search response")
                        native_ms = response.get("search_ms")
                        expected_budget = top_k * 5
                        expected_filter = q["filter"]
                        if (isinstance(native_ms, bool) or not isinstance(native_ms, (int, float))
                                or not math.isfinite(native_ms) or native_ms < 0
                                or response.get("applied_filter") != expected_filter
                                or response.get("internal_candidate_budget") != expected_budget):
                            raise SembleCollectionError("worker returned invalid timing, candidate budget, or applied filter")
                        payload = response["payload"]
                        if not isinstance(payload, dict) or payload.get("query") != q["text"] or not isinstance(payload.get("results"), list):
                            raise SembleCollectionError("worker returned malformed native payload")
                        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
                        rel = f"payloads/{len(results) + 1:06d}.json"
                        with (output / rel).open("xb") as stream:
                            stream.write(raw)
                        # Apply the same native payload validation used by the offline peer importer.
                        native_hits = peer._native_hits(raw, q["text"], top_k, q["id"])
                        parsed = [{"path": hit["path"], "start_line": hit["start_line"],
                                   "end_line": hit["end_line"]} for hit in native_hits]
                        one_query_run = {"format_version": 1, "plan_fingerprint": plan["fingerprint"],
                            "name": name, "tool": {"name": "semble", "revision": source_revision},
                            "configuration": {}, "candidate_k": top_k,
                            "results": [{"query_id": q["id"], "status": "ok", "hits": parsed}]}
                        ev._run_scores(one_query_run, {**plan["spec"], "queries": [q]},
                                       ev._verify_plan(plan), plan["fingerprint"])
                        results.append({"query_id": q["id"], "status": "ok", "hits": parsed, "elapsed_ms": wall_ms})
                        manifest_rows.append({"query_id": q["id"], "query_text": q["text"], "applied_filter": q["filter"],
                                              "status": "ok", "payload": rel})
                        query_records[q["id"]] = {"repository": rid, "elapsed_ms": wall_ms,
                            "native_search_ms": native_ms, "internal_candidate_budget": expected_budget,
                            "applied_filter": response["applied_filter"], "status": "ok"}
                    except Exception as exc:
                        results.append({"query_id": q["id"], "status": _failure_status(exc), "hits": [], "error": str(exc)})
                        manifest_rows.append({"query_id": q["id"], "query_text": q["text"], "applied_filter": q["filter"],
                                              "status": _failure_status(exc), "error": str(exc)})
                        query_records[q["id"]] = {"repository": rid, "status": _failure_status(exc), "error": str(exc)}
                try:
                    session.request("shutdown", {})
                except Exception:
                    pass
                records[rid]["session_records"] = getattr(session, "records", [])
        except Exception as exc:
            status, error = _failure_status(exc), str(exc)
            records[rid] = {"command": command, "status": status, "error": error}
            already = {r["query_id"] for r in results}
            for q in q_by_repo[rid]:
                if q["id"] not in already:
                    results.append({"query_id": q["id"], "status": status, "hits": [], "error": error})
                    manifest_rows.append({"query_id": q["id"], "query_text": q["text"], "applied_filter": q["filter"],
                                          "status": status, "error": error})
                    query_records[q["id"]] = {"repository": rid, "status": status, "error": error}
    if (source_before != rc._fingerprint_path(source) or model_before != rc._fingerprint_path(model_dir)
            or python_before != rc._fingerprint_path(python)):
        raise SembleCollectionError("source, model, or interpreter changed during collection")
    for rid, root in roots.items():
        if rc.source_inventory(root) != {f["path"]: {"sha256": f["sha256"], "line_count": f["line_count"]}
                for f in plan["files"] if (f["repository"] == rid if plan["format_version"] == 2 else True)}:
            raise SembleCollectionError(f"source changed during collection: {rid}")
    run = {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "name": name,
           "tool": {"name": "semble", "revision": source_revision},
           "configuration": {"engine": "native-default", "top_k": top_k, "internal_candidate_budget": "5*top_k per branch",
             "timing_boundary": REQUEST_TIMING_BOUNDARY,
             "source_return_mode": "native full chunks in search response",
             "source_revision_is_declared_not_attested": True, "model_path": str(model_dir)},
           "candidate_k": top_k, "results": results}
    report = ev.score(plan, [run])
    manifest = {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "name": name,
                "tool": {"name": "semble", "revision": source_revision},
                "configuration": run["configuration"], "candidate_k": top_k, "results": manifest_rows}
    rc._write_new(output / "plan.json", plan)
    rc._write_new(output / "run.json", run)
    rc._write_new(output / "report.json", report)
    rc._write_new(output / "manifest.json", manifest)
    artifact_hashes = []
    for artifact in sorted(output.rglob("*")):
        if not artifact.is_file() or "runtime-cache" in artifact.relative_to(output).parts:
            continue
        artifact_hashes.append({"path": artifact.relative_to(output).as_posix(),
                                "sha256": rc.sha256_file(artifact), "bytes": artifact.stat().st_size})
    provenance = {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "source_revision_declared": source_revision,
                  "source": source_before, "model": model_before, "interpreter": python_before,
                  "repositories": records, "native_payloads": [
                    {"path": p.relative_to(output).as_posix(), "sha256": rc.sha256_file(p)}
                    for p in sorted(payload_dir.glob("*.json"))],
                  "queries": query_records,
                  "artifact_hashes_excluding_provenance_and_runtime_cache": artifact_hashes,
                  "limits": ["Semble's native search pipeline and 5*top_k per-branch candidate budget are preserved.",
                    "The declared source revision is not cryptographic attestation; source bytes are fingerprinted.",
                    "Model/native retrieval behavior is real, but this collector does not establish independent query labels."]}
    rc._write_new(output / "provenance.json", provenance)
    return {"run": run, "report": report, "manifest": manifest, "provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--root")
    parser.add_argument("--model-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--plan")
    parser.add_argument("--roots", help="JSON object mapping frozen repository IDs to absolute paths")
    parser.add_argument("--python")
    parser.add_argument("--output")
    parser.add_argument("--name", default="semble-native")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args.source, args.root, args.model_dir, args.cache_dir)
    try:
        plan = ev._load(args.plan)
        roots = ev._load(args.roots)
        result = collect_semble(plan, roots, args.python, args.source, args.model_dir, args.output,
                                name=args.name, top_k=args.top_k, timeout_s=args.timeout_seconds,
                                source_revision=args.source_revision or "")
        print(json.dumps({"run": result["run"]["name"], "output": str(Path(args.output).resolve())}))
        return 0
    except Exception as exc:
        parser.exit(2, f"repository_semble: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
