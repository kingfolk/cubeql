"""Collect bounded, fresh-process CubeQL repository retrieval runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

import repository_eval as ev


class CollectionError(ValueError):
    """Invalid collection input or an unverifiable collection."""


STRATEGIES = ("lexical", "symbol", "semantic", "hybrid")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_inventory(root):
    """Return deterministic repository-relative identities for all safe files."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise CollectionError(f"source root is not a directory: {root}")
    files = {}

    def walk_error(error):
        raise CollectionError(f"cannot inventory repository: {error}") from error

    for current, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if d != ".git" and not (Path(current) / d).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            rel = path.relative_to(root).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                continue
            if path.is_symlink():
                raise CollectionError(f"symlink file entry is not allowed: {rel}")
            try:
                mode = path.stat(follow_symlinks=False).st_mode
                if not stat.S_ISREG(mode):
                    continue
                digest = hashlib.sha256()
                line_feeds = 0
                has_bytes = False
                last_byte = None
                with open(path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        line_feeds += block.count(b"\n")
                        has_bytes = True
                        last_byte = block[-1]
            except OSError as e:
                raise CollectionError(f"cannot inventory {rel}: {e}") from e
            files[rel] = {"sha256": digest.hexdigest(),
                          "line_count": line_feeds + (1 if has_bytes and last_byte != 10 else 0)}
    return dict(sorted(files.items()))


def _session_groups(session_id):
    """Return process groups that are members of our private launched session."""
    try:
        result = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,pgid="], capture_output=True,
                                text=True, timeout=2, check=False)
        if result.returncode:
            return None
        groups = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                pid, pgid = map(int, parts)
                try:
                    if os.getsid(pid) == session_id:
                        groups.add(pgid)
                except ProcessLookupError:
                    continue
                except (PermissionError, OSError):
                    return None
        return groups
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _terminate_tree(proc, groups, session_owned=False):
    cleanup_error = None
    if os.name == "posix":
        if session_owned:
            session_groups = _session_groups(proc.pid)
            if session_groups is None:
                cleanup_error = "could not enumerate private session; terminated known process group only"
            else:
                groups.update(session_groups)
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Allow group members (including detached encoder workers) time to handle
        # SIGTERM; waiting only on the CLI would not wait for its orphaned children.
        pending = set(groups)
        grace_deadline = time.monotonic() + 0.5
        while pending and time.monotonic() < grace_deadline:
            for pgid in tuple(pending):
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    pending.discard(pgid)
                except (PermissionError, OSError):
                    pending.discard(pgid)
            if pending:
                time.sleep(0.02)
        for pgid in pending:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    else:  # execute rejects non-POSIX before reaching here.
        proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        cleanup_error = cleanup_error or "launched process did not exit after process-group termination"
    return cleanup_error


def execute(command, output_dir, stem, timeout_s=300, max_output_bytes=16777216):
    """Run argv without a shell, bounding runtime and aggregate captured bytes."""
    if os.name != "posix":
        raise CollectionError("repository collection currently supports POSIX only")
    if (not isinstance(command, (list, tuple)) or not command or
            any(not isinstance(x, str) or not x or "\x00" in x for x in command)):
        raise CollectionError("command must be a nonempty argv of nonempty strings")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or
            not math.isfinite(timeout_s) or timeout_s <= 0):
        raise CollectionError("timeout_s must be positive")
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 0:
        raise CollectionError("max_output_bytes must be a nonnegative integer")
    if not isinstance(stem, str) or not stem or Path(stem).name != stem or stem in (".", ".."):
        raise CollectionError("stem must be a simple filename stem")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stdout_rel, stderr_rel = f"{stem}.stdout", f"{stem}.stderr"
    stdout_path, stderr_path = out / stdout_rel, out / stderr_rel
    started = time.monotonic()
    record = {"status": "failed", "returncode": None, "elapsed_ms": 0,
              "stdout": stdout_rel, "stderr": stderr_rel, "command": list(command)}
    # Exclusive creation protects earlier diagnostics even if a caller reuses a stem.
    try:
        with open(stdout_path, "xb") as stdout, open(stderr_path, "xb") as stderr:
            if "--encoder-config" in command and _session_groups(os.getpid()) is None:
                record["error"] = "cannot enumerate POSIX processes; configured encoder was not launched"
                return _elapsed(record, started)
            try:
                proc = subprocess.Popen(list(command), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        start_new_session=True, close_fds=True)
            except OSError as e:
                record["error"] = f"invocation failed: {e}"
                return _elapsed(record, started)
            import selectors
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ, (stdout, "stdout"))
            selector.register(proc.stderr, selectors.EVENT_READ, (stderr, "stderr"))
            groups = {proc.pid}
            session_owned = "--encoder-config" in command
            total = 0
            deadline = started + timeout_s
            status = None
            error = None
            last_session_scan = 0.0
            while selector.get_map() or proc.poll() is None:
                if time.monotonic() >= deadline:
                    status, error = "timeout", f"command exceeded {timeout_s:g}s deadline"
                    break
                # Configured resident encoders can use a separate process group. Keep their
                # private session identity observable while the CLI is alive.
                if session_owned and time.monotonic() - last_session_scan >= 0.2:
                    session_groups = _session_groups(proc.pid)
                    last_session_scan = time.monotonic()
                    if session_groups:
                        groups.update(session_groups)
                events = selector.select(min(0.05, max(0, deadline - time.monotonic())))
                for key, _ in events:
                    stream, _label = key.data
                    try:
                        data = os.read(key.fileobj.fileno(), 65536)
                    except (OSError, ValueError):
                        try:
                            selector.unregister(key.fileobj)
                        except Exception:
                            pass
                        continue
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    remaining = max_output_bytes - total
                    if remaining > 0:
                        stream.write(data[:remaining])
                        total += min(len(data), remaining)
                    if len(data) > remaining:
                        status, error = "failed", f"captured output exceeded {max_output_bytes} byte limit"
                        break
                if status:
                    break
            if status:
                cleanup_error = _terminate_tree(proc, groups, session_owned)
                if cleanup_error:
                    error = f"{error}; cleanup: {cleanup_error}"
            else:
                record["returncode"] = proc.wait()
                status = "ok" if record["returncode"] == 0 else "failed"
                if status == "failed":
                    error = f"command exited with status {record['returncode']}"
                # A CLI can exit before a configured encoder worker. The launched session
                # belongs solely to this invocation, so terminate any workers left in it.
                if session_owned:
                    session_groups = _session_groups(proc.pid)
                    if session_groups is None:
                        error = "could not verify configured encoder session cleanup"
                        status = "failed"
                    else:
                        orphan_groups = session_groups - {proc.pid}
                        if orphan_groups:
                            cleanup = _terminate_tree(proc, orphan_groups, False)
                            if cleanup:
                                error = f"encoder cleanup failed: {cleanup}"
                            if status == "ok":
                                status = "failed"
                                error = error or "configured encoder worker survived CLI exit"
            selector.close()
            proc.stdout.close()
            proc.stderr.close()
            stdout.flush(); stderr.flush()
            record["status"] = status
            if error:
                record["error"] = error
            if status != "timeout" and proc.poll() is not None:
                record["returncode"] = proc.returncode
    except FileExistsError as e:
        raise CollectionError(f"diagnostic output already exists: {e.filename}") from e
    return _elapsed(record, started)


def _elapsed(record, started):
    record["elapsed_ms"] = max(0, int((time.monotonic() - started) * 1000))
    return record


def _load_json(path):
    return ev._load(path)


def _write_new(path, value):
    try:
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as e:
        raise CollectionError(f"output already exists: {path}") from e


def _fingerprint_path(path):
    original = Path(path).absolute()
    resolved = original.resolve(strict=True)
    if resolved.is_file():
        return {"kind": "file", "path": str(original), "resolved_path": str(resolved),
                "sha256": sha256_file(resolved)}
    if not resolved.is_dir():
        raise CollectionError(f"fingerprint target is not a regular file or directory: {resolved}")
    entries = {}
    def walk_error(error):
        raise CollectionError(f"cannot fingerprint directory {resolved}: {error}") from error
    for current, dirs, names in os.walk(resolved, followlinks=False, onerror=walk_error):
        for d in dirs:
            if (Path(current) / d).is_symlink():
                raise CollectionError(f"symlink in fingerprint directory: {Path(current) / d}")
        dirs[:] = sorted(dirs)
        for name in sorted(names):
            p = Path(current) / name
            if p.is_symlink():
                raise CollectionError(f"symlink in fingerprint directory: {p}")
            if p.is_file():
                entries[p.relative_to(resolved).as_posix()] = sha256_file(p)
    return {"kind": "directory", "path": str(original), "resolved_path": str(resolved), "files": entries}


def _profile_fingerprints(binary, encoder_config, model_dir):
    result = {"binary": _fingerprint_path(binary)}
    if encoder_config:
        config = Path(encoder_config).resolve(strict=True)
        result["encoder_config"] = _fingerprint_path(config)
        try:
            obj = _load_json(config)
        except Exception as e:
            raise CollectionError(f"cannot parse encoder config: {e}") from e
        if (not isinstance(obj, dict) or
                set(obj) - {"format_version", "command", "args", "startup_timeout_ms", "request_timeout_ms"} or
                not {"format_version", "command"}.issubset(obj) or
                type(obj.get("format_version")) is not int or obj.get("format_version") != 1 or
                not isinstance(obj.get("command"), str) or not Path(obj["command"]).is_absolute() or
                not isinstance(obj.get("args", []), list) or
                any(not isinstance(arg, str) for arg in obj.get("args", []))):
            raise CollectionError("encoder config must contain format_version 1, absolute command, and string args")
        for key, default in (("startup_timeout_ms", 120000), ("request_timeout_ms", 30000)):
            value = obj.get(key, default)
            if type(value) is not int or not 1 <= value <= 600000:
                raise CollectionError(f"encoder config {key} must be an integer in 1..600000")
        result["encoder_executable"] = _fingerprint_path(obj["command"])
        arg_files = {}
        for i, arg in enumerate(obj.get("args", [])):
            candidate = Path(arg)
            if candidate.is_absolute():
                if not candidate.exists():
                    raise CollectionError(f"absolute encoder argument does not exist and cannot be fingerprinted: {arg}")
                arg_files[str(i)] = _fingerprint_path(candidate)
        result["encoder_argument_paths"] = arg_files
    if model_dir:
        result["model_dir"] = _fingerprint_path(model_dir)
    return result


def _check_config(embedding, encoder_config, model_dir):
    if embedding not in ("hash", "minilm", "process"):
        raise CollectionError("embedding must be hash, minilm, or process")
    if embedding == "hash" and (encoder_config or model_dir):
        raise CollectionError("hash embedding disallows encoder config and model directory")
    if embedding == "minilm" and (not model_dir or encoder_config):
        raise CollectionError("minilm requires model directory and disallows encoder config")
    if embedding == "process" and (not encoder_config or model_dir):
        raise CollectionError("process requires encoder config and disallows model directory")


def _inclusive_source_bytes(data, start_line, end_line):
    """Extract one-based inclusive LF lines while retaining original bytes/endings."""
    raw_lines = data.split(b"\n")
    if data.endswith(b"\n"):
        raw_lines.pop()
        lines = [line + b"\n" for line in raw_lines]
    else:
        lines = [line + b"\n" for line in raw_lines[:-1]] + [raw_lines[-1]]
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise CollectionError("native hit range is outside source bytes")
    return b"".join(lines[start_line - 1:end_line])


def collect(plan, roots, binary, output, *, name="cubeql", strategies=("lexical", "symbol", "semantic", "hybrid"),
            candidate_k=100, top_k=100, embedding="hash", encoder_config=None, model_dir=None,
            timeout_s=300, fallback_lines=80, max_file_bytes=2097152):
    """Collect one fresh CLI invocation per query/strategy and score exact coverage."""
    file_index = ev._verify_plan(plan)
    spec = plan["spec"]
    ids = {r["id"] for r in spec["repositories"]}
    if not isinstance(roots, dict) or set(roots) != ids:
        raise CollectionError("roots must name exactly the plan repository IDs")
    if not isinstance(name, str) or not name.strip():
        raise CollectionError("name must be a nonempty string")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 10:
        raise CollectionError("candidate_k must be an integer >= 10")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 10 <= top_k <= candidate_k:
        raise CollectionError("top_k must be between 10 and candidate_k")
    strategies = tuple(strategies)
    if not strategies or len(set(strategies)) != len(strategies) or any(s not in STRATEGIES for s in strategies):
        raise CollectionError("strategies must be unique known strategy names")
    if isinstance(fallback_lines, bool) or not isinstance(fallback_lines, int) or fallback_lines < 1:
        raise CollectionError("fallback_lines must be a positive integer")
    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes < 1:
        raise CollectionError("max_file_bytes must be a positive integer")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or
            not math.isfinite(timeout_s) or timeout_s <= 0):
        raise CollectionError("timeout_s must be positive")
    _check_config(embedding, encoder_config, model_dir)
    binary = Path(binary).resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise CollectionError("binary must be an absolute resolved executable regular file")
    root_paths = {rid: Path(p).resolve(strict=True) for rid, p in roots.items()}
    if any(not p.is_dir() for p in root_paths.values()):
        raise CollectionError("all repository roots must be directories")
    if embedding == "minilm" and not Path(model_dir).is_dir():
        raise CollectionError("model_dir must be an existing directory")
    if embedding == "process" and not Path(encoder_config).is_file():
        raise CollectionError("encoder_config must be an existing file")
    output = Path(output).absolute()
    if output.exists():
        raise CollectionError(f"output already exists: {output}")
    output_resolved = output.resolve(strict=False)
    for root in root_paths.values():
        if output_resolved == root or root in output_resolved.parents:
            raise CollectionError("output directory cannot be inside a frozen repository root")
    source_before = {rid: source_inventory(root_paths[rid]) for rid in sorted(ids)}
    expected = {rid: {f["path"]: {"sha256": f["sha256"], "line_count": f["line_count"]}
                      for f in plan["files"] if (plan["format_version"] == 1 or f["repository"] == rid)}
                for rid in sorted(ids)}
    # v1 plans omit repository on files; v2 maps per repository.
    if source_before != expected:
        raise CollectionError("repository sources do not match frozen plan")
    profile_before = _profile_fingerprints(binary, encoder_config, model_dir)

    output.mkdir(parents=True, exist_ok=False)
    records = []
    commands = []
    snapshots = {}
    root_map = {rid: str(root_paths[rid]) for rid in sorted(ids)}
    _write_new(output / "plan.json", plan)
    _write_new(output / "roots.json", root_map)
    query_by_repo = {rid: [q for q in spec["queries"] if q["repository"] == rid] for rid in sorted(ids)}
    failed_repos = {}
    for ri, rid in enumerate(sorted(ids)):
        snap_rel = f"snapshot.{ri}.arrow"
        cmd = [str(binary), "index", str(root_paths[rid]), "--output", str(output / snap_rel),
               "--embedding", embedding, "--fallback-lines", str(fallback_lines),
               "--max-file-bytes", str(max_file_bytes), "--format", "json"]
        if encoder_config:
            cmd += ["--encoder-config", str(Path(encoder_config).resolve())]
        if model_dir:
            cmd += ["--model-dir", str(Path(model_dir).resolve())]
        record = execute(cmd, output, f"command.{len(commands)}", timeout_s)
        commands.append(record)
        _write_new(output / f"execution.{len(commands)-1}.json", record)
        snapshots[rid] = {"path": snap_rel, "execution": record}
        if record["status"] == "ok":
            try:
                metadata = _load_json(output / record["stdout"])
                if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
                    raise CollectionError("index response must be format_version 1 JSON")
                snapshot_info = metadata.get("snapshot")
                if snapshot_info is not None:
                    if not isinstance(snapshot_info, dict):
                        raise CollectionError("index snapshot metadata must be an object")
                    if snapshot_info.get("root") not in (None, str(root_paths[rid])):
                        raise CollectionError("index snapshot root does not match frozen repository root")
                    if snapshot_info.get("fallback_lines") not in (None, fallback_lines):
                        raise CollectionError("index snapshot fallback_lines differs from requested value")
                    if snapshot_info.get("max_file_bytes") not in (None, max_file_bytes):
                        raise CollectionError("index snapshot max_file_bytes differs from requested value")
                    if not isinstance(snapshot_info.get("corpus_fingerprint"), str):
                        raise CollectionError("index snapshot is missing corpus fingerprint metadata")
                    if not isinstance(snapshot_info.get("chunks"), int):
                        raise CollectionError("index snapshot is missing chunk count metadata")
                snap = output / snap_rel
                if not snap.is_file():
                    raise CollectionError("index command succeeded without creating its snapshot")
                snapshots[rid]["sha256"] = sha256_file(snap)
                snapshots[rid]["metadata"] = metadata
            except Exception as e:
                failed_repos[rid] = ("failed", f"invalid index output: {e}")
        else:
            failed_repos[rid] = (record["status"], record.get("error", "index failed"))

    for strategy in strategies:
        results = []
        for rid in sorted(ids):
            for q in query_by_repo[rid]:
                if rid in failed_repos:
                    status, error = failed_repos[rid]
                    results.append({"query_id": q["id"], "status": status, "error": error, "hits": []})
                    continue
                cmd = [str(binary), "search", str(output / snapshots[rid]["path"]),
                       "--top-k", str(top_k), "--candidate-k", str(candidate_k),
                       "--strategy", strategy, "--include-source", "--format", "json"]
                prefix = q["filter"].get("path_prefix")
                if prefix is not None:
                    cmd += ["--path-prefix", prefix]
                if strategy in ("semantic", "hybrid"):
                    if encoder_config:
                        cmd += ["--encoder-config", str(Path(encoder_config).resolve())]
                    if model_dir:
                        cmd += ["--model-dir", str(Path(model_dir).resolve())]
                cmd += ["--", q["text"]]
                record = execute(cmd, output, f"command.{len(commands)}", timeout_s)
                commands.append(record)
                _write_new(output / f"execution.{len(commands)-1}.json", record)
                if record["status"] != "ok":
                    results.append({"query_id": q["id"], "status": record["status"],
                                    "error": record.get("error", "search failed"), "hits": [],
                                    "elapsed_ms": record["elapsed_ms"]})
                    continue
                try:
                    response = _load_json(output / record["stdout"])
                    if (not isinstance(response, dict) or response.get("format_version") != 2 or
                            response.get("query") != q["text"] or response.get("strategy") != strategy or
                            not isinstance(response.get("hits"), list)):
                        raise CollectionError("native response query, strategy, format, or hits is invalid")
                    if len(response["hits"]) > top_k:
                        raise CollectionError("native response exceeds requested top_k")
                    hits = []
                    for native in response["hits"]:
                        if not isinstance(native, dict) or not isinstance(native.get("source"), str) or not native["source"]:
                            raise CollectionError("native hit is missing nonempty included source")
                        hit = {k: native[k] for k in ("path", "start_line", "end_line") if k in native}
                        # First let the common scorer enforce the frozen path/range/filter contract.
                        hits.append(hit)
                        plan_file = next((f for f in plan["files"] if f["path"] == hit.get("path") and
                                          (plan["format_version"] == 1 or f["repository"] == rid)), None)
                        if plan_file is None:
                            raise CollectionError("native hit path is outside frozen inventory")
                        source_bytes = native["source"].encode("utf-8")
                        # Preserve LF/CRLF bytes and permit a native chunk to be a substring
                        # of the original inclusive source span.
                        source_data = (root_paths[rid] / hit["path"]).read_bytes()
                        span_bytes = _inclusive_source_bytes(source_data, hit["start_line"], hit["end_line"])
                        if not source_bytes.strip() or source_bytes not in span_bytes:
                            raise CollectionError("native hit source is inconsistent with frozen source span")
                    probe = {"format_version": 1, "plan_fingerprint": plan["fingerprint"],
                             "name": f"{name}-{strategy}-probe",
                             "tool": {"name": "cubeql", "revision": "probe"},
                             "configuration": {}, "candidate_k": top_k,
                             "results": [{"query_id": q["id"], "status": "ok", "hits": hits}]}
                    ev._run_scores(probe, {"format_version": plan["format_version"], "queries": [q]},
                                   file_index, plan["fingerprint"])
                    results.append({"query_id": q["id"], "status": "ok", "hits": hits,
                                    "elapsed_ms": record["elapsed_ms"]})
                except Exception as e:
                    results.append({"query_id": q["id"], "status": "failed",
                                    "error": f"invalid native response: {e}", "hits": [],
                                    "elapsed_ms": record["elapsed_ms"]})
        run = {"format_version": 1, "plan_fingerprint": plan["fingerprint"],
               "name": f"{name}-{strategy}",
               "tool": {"name": "cubeql", "revision": profile_before["binary"]["sha256"]},
               "configuration": {"strategy": strategy, "embedding": embedding,
                                  "internal_candidate_k": candidate_k, "top_k": top_k,
                                  "fresh_process_per_query": True,
                                  "timing_boundary": "fresh CLI end-to-end wall time",
                                  "fallback_lines": fallback_lines, "max_file_bytes": max_file_bytes},
               "candidate_k": top_k, "results": results}
        records.append(run)

    # Recheck all inputs before scoring/valid result writes.
    source_after = {rid: source_inventory(root_paths[rid]) for rid in sorted(ids)}
    profile_after = _profile_fingerprints(binary, encoder_config, model_dir)
    changed = source_after != source_before or profile_after != profile_before
    if changed:
        raise CollectionError("source, executable, model, or encoder configuration changed during collection; runs retained only in command diagnostics")
    report = ev.score(plan, records)
    for run in records:
        _write_new(output / f"run.{run['configuration']['strategy']}.json", run)
    _write_new(output / "report.json", report)
    artifacts = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            artifacts[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    manifest = {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "name": name,
                "binary": profile_after["binary"], "profile_fingerprints": profile_after,
                "source_inventories": source_after, "snapshots": snapshots,
                "commands": commands, "artifacts": artifacts,
                "timing_boundary": "fresh CLI end-to-end wall time per index/search command"}
    _write_new(output / "manifest.json", manifest)
    return {"runs": records, "report": report, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True); parser.add_argument("--roots", required=True)
    parser.add_argument("--binary", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="cubeql"); parser.add_argument("--strategy", action="append", dest="strategies")
    parser.add_argument("--embedding", choices=("hash", "minilm", "process"), default="hash")
    parser.add_argument("--encoder-config"); parser.add_argument("--model-dir")
    parser.add_argument("--candidate-k", type=int, default=100); parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--fallback-lines", type=int, default=80); parser.add_argument("--max-file-bytes", type=int, default=2097152)
    args = parser.parse_args(argv)
    try:
        result = collect(_load_json(args.plan), _load_json(args.roots), args.binary, args.output,
                         name=args.name, strategies=args.strategies or STRATEGIES,
                         candidate_k=args.candidate_k, top_k=args.top_k, embedding=args.embedding,
                         encoder_config=args.encoder_config, model_dir=args.model_dir,
                         timeout_s=args.timeout_seconds, fallback_lines=args.fallback_lines,
                         max_file_bytes=args.max_file_bytes)
        print(json.dumps({"runs": [r["name"] for r in result["runs"]],
                          "report": str(Path(args.output) / "report.json"),
                          "manifest": str(Path(args.output) / "manifest.json")}, ensure_ascii=False))
        return 0
    except (CollectionError, ev.EvaluationError, OSError, TypeError, ValueError) as e:
        print(f"repository_collect: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
