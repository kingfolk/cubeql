"""Persistent, bounded JSON-RPC transport and snapshot-backed MCP collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time

import repository_collect as rc
import repository_eval as ev

REQUEST_TIMING_BOUNDARY = "resident request wall ms; excludes indexing/startup and source reads"
FRAME_LIMIT = 2 * 1024 * 1024
SESSION_CAPTURE_LIMIT = 16 * 1024 * 1024


class MCPError(ValueError):
    pass


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise MCPError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad_constant(token): raise MCPError(f"non-finite JSON number: {token}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def _positive_timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise MCPError("timeout_s must be a finite positive number")


class JSONLineSession:
    """One private MCP process with serialized requests and byte-preserving traces."""
    def __init__(self, command, output_dir, stem, timeout_s=30, env=None):
        _positive_timeout(timeout_s)
        if not isinstance(command, (list, tuple)) or not command or any(not isinstance(x, str) or not x or "\0" in x for x in command):
            raise MCPError("command must be a nonempty argv of nonempty strings")
        if not isinstance(stem, str) or not stem or Path(stem).name != stem or stem in (".", ".."):
            raise MCPError("stem must be a simple filename stem")
        self.command, self.timeout_s = list(command), float(timeout_s)
        self.output_dir, self.stem = Path(output_dir), stem
        self.env = env
        self.records = []
        self.next_id = 1
        self._request_lock = threading.Lock()
        self.poisoned = False
        self.closed = False
        self._buffer = bytearray()
        self._captured = 0
        self._stdout_path = self.output_dir / f"{stem}.stdout"
        self._stderr_path = self.output_dir / f"{stem}.stderr"
        self._stdin_path = self.output_dir / f"{stem}.stdin"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Preflight all paths before launch; x mode prevents overwriting evidence.
        try:
            self._stdout = open(self._stdout_path, "xb")
            self._stderr = open(self._stderr_path, "xb")
            self._stdin = open(self._stdin_path, "xb")
        except FileExistsError as e:
            for f in (getattr(self, "_stdout", None), getattr(self, "_stderr", None), getattr(self, "_stdin", None)):
                if f: f.close()
            raise MCPError(f"diagnostic output already exists: {e.filename}") from e
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, env=env, start_new_session=True, close_fds=True)
        except OSError as e:
            self._close_files()
            raise MCPError(f"MCP invocation failed: {e}") from e
        os.set_blocking(self.proc.stdin.fileno(), False)
        os.set_blocking(self.proc.stdout.fileno(), False)
        os.set_blocking(self.proc.stderr.fileno(), False)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self.proc.stdout, selectors.EVENT_READ, "out")
        self._sel.register(self.proc.stderr, selectors.EVENT_READ, "err")
        self._sel.register(self.proc.stdin, selectors.EVENT_WRITE, "in")
        self._stdin_enabled = True
        self._groups = {self.proc.pid}
        self.started = time.monotonic()

    def __enter__(self): return self

    def __exit__(self, *_): self.close()

    def _close_files(self):
        for f in (getattr(self, "_stdout", None), getattr(self, "_stderr", None), getattr(self, "_stdin", None)):
            if f and not f.closed: f.close()

    def _poison(self, message):
        self.poisoned = True
        raise MCPError(message)

    def _capture(self, label, data):
        if self._captured + len(data) > SESSION_CAPTURE_LIMIT:
            self._poison("MCP session exceeded 16 MiB combined stdout/stderr capture limit")
        (self._stdout if label == "out" else self._stderr).write(data)
        self._captured += len(data)

    def _pump(self, deadline, write_data=None):
        offset = 0
        while True:
            if time.monotonic() >= deadline: self._poison("MCP request deadline exceeded")
            if self.proc.poll() is not None and not self._sel.get_map(): self._poison("MCP process exited")
            for key, _ in self._sel.select(min(.05, max(0, deadline-time.monotonic()))):
                kind = key.data
                if kind in ("out", "err"):
                    try: data = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError: continue
                    if not data:
                        self._sel.unregister(key.fileobj); key.fileobj.close(); continue
                    self._capture(kind, data)
                    if kind == "out":
                        self._buffer.extend(data)
                        if len(self._buffer) > FRAME_LIMIT and b"\n" not in self._buffer:
                            self._poison("MCP frame exceeds 2 MiB limit")
                elif write_data is not None and offset < len(write_data):
                    try: n = os.write(key.fileobj.fileno(), write_data[offset:])
                    except (BrokenPipeError, OSError): self._poison("MCP stdin closed during request")
                    offset += n
                    if offset >= len(write_data):
                        self._sel.unregister(key.fileobj); self._stdin_enabled = False
                if write_data is None or offset >= len(write_data):
                    return

    def _write(self, payload, deadline):
        if not self._stdin_enabled:
            self._sel.register(self.proc.stdin, selectors.EVENT_WRITE, "in"); self._stdin_enabled = True
        self._stdin.write(payload); self._stdin.flush()
        self._pump(deadline, payload)

    def _next_message(self, deadline):
        while True:
            pos = self._buffer.find(b"\n")
            if pos >= 0:
                frame = bytes(self._buffer[:pos+1]); del self._buffer[:pos+1]
                if pos > FRAME_LIMIT: self._poison("MCP frame exceeds 2 MiB limit")
                try: msg = _strict_json(frame.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, MCPError) as e: self._poison(f"malformed MCP JSON frame: {e}")
                if not isinstance(msg, dict): self._poison("MCP JSON-RPC frame must be an object")
                return msg
            self._pump(deadline)

    def _rpc(self, method, params, want_result):
        if not self._request_lock.acquire(timeout=self.timeout_s):
            self._poison("MCP request serialization deadline exceeded")
        try:
            return self._rpc_one(method, params, want_result)
        finally:
            self._request_lock.release()

    def _rpc_one(self, method, params, want_result):
        if self.closed or self.poisoned: raise MCPError("MCP session is closed or poisoned")
        if not isinstance(method, str) or not method or not isinstance(params, dict): raise MCPError("method/params invalid")
        started = time.monotonic(); deadline = started + self.timeout_s
        try:
            if not want_result:
                payload = (json.dumps({"jsonrpc":"2.0", "method":method, "params":params}, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
            else:
                rid = self.next_id
                payload = (json.dumps({"jsonrpc":"2.0", "id":rid, "method":method, "params":params}, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
        except (TypeError, ValueError) as e: raise MCPError(f"request params are not finite JSON: {e}") from e
        if len(payload) > FRAME_LIMIT: raise MCPError("outgoing MCP frame exceeds 2 MiB limit")
        if not want_result:
            self._write(payload, deadline)
            rec = {"method":method, "elapsed_ms":(time.monotonic()-started)*1000, "notification":True}
            self.records.append(rec); return None
        self.next_id += 1
        self._write(payload, deadline)
        while True:
            msg = self._next_message(deadline)
            if msg.get("jsonrpc") != "2.0": self._poison("response is not JSON-RPC 2.0")
            if "method" in msg and "id" not in msg: continue  # server notification; same deadline
            if type(msg.get("id")) is not int or msg["id"] != rid: self._poison(f"unexpected JSON-RPC response id {msg.get('id')!r}; expected {rid}")
            if ("result" in msg) == ("error" in msg): self._poison("response must contain exactly one of result/error")
            self.records.append({"id":rid,"method":method,"elapsed_ms":(time.monotonic()-started)*1000})
            if "error" in msg: raise MCPError(f"JSON-RPC error: {msg['error']!r}")
            return msg["result"]

    def request(self, method, params): return self._rpc(method, params, True)
    def notify(self, method, params): return self._rpc(method, params, False)

    def close(self):
        if self.closed: return
        self.closed = True
        cleanup_error = None
        try:
            # The session ID is the child PID and remains usable to find orphaned
            # members after the JSON-RPC parent has exited.
            found = rc._session_groups(self.proc.pid)
            if found is not None: self._groups.update(found)
            else: cleanup_error = "could not enumerate private session before close"
            try: self._sel.unregister(self.proc.stdin)
            except Exception: pass
            if self.proc.stdin and not self.proc.stdin.closed: self.proc.stdin.close()
            try: self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                cleanup_error = rc._terminate_tree(self.proc, self._groups, session_owned=True) or cleanup_error
            else:
                # A clean parent exit can still leave private-session workers behind.
                cleanup_error = rc._terminate_tree(self.proc, self._groups, session_owned=False)
            # Preserve trailing stdout/stderr without allowing an unbounded drain.
            drain_deadline=time.monotonic()+0.5
            while self._sel.get_map() and time.monotonic()<drain_deadline:
                events=self._sel.select(min(.05,max(0,drain_deadline-time.monotonic())))
                if not events: break
                for key,_ in events:
                    kind=key.data
                    try: data=os.read(key.fileobj.fileno(),65536)
                    except (OSError,ValueError): data=b""
                    if not data:
                        try: self._sel.unregister(key.fileobj)
                        except Exception: pass
                        try: key.fileobj.close()
                        except Exception: pass
                        continue
                    try: self._capture(kind,data)
                    except MCPError as e:
                        cleanup_error=cleanup_error or str(e)
                        try: self._sel.unregister(key.fileobj)
                        except Exception: pass
                        key.fileobj.close()
            # Any descendants still holding pipes are outside the bounded drain window.
            for stream in (self.proc.stdout, self.proc.stderr):
                if stream and not stream.closed: stream.close()
            try: self._sel.close()
            except Exception: pass
        finally:
            self._close_files()
        if cleanup_error: self.records.append({"cleanup_error":cleanup_error})


def _write_new(path, value):
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    except FileExistsError as e:
        raise MCPError(f"output already exists: {path}") from e


def _tool_result(result):
    if not isinstance(result, dict) or result.get("isError") is True:
        raise MCPError("tools/call returned isError or malformed result")
    structured = result.get("structuredContent")
    texts = [x.get("text") for x in result.get("content", []) if isinstance(x, dict) and x.get("type") == "text"]
    parsed = []
    for text in texts:
        try: parsed.append(_strict_json(text))
        except (TypeError, json.JSONDecodeError, MCPError) as e: raise MCPError(f"malformed JSON text content: {e}") from e
    if structured is None:
        if len(parsed) != 1 or not isinstance(parsed[0], dict): raise MCPError("tool result lacks structuredContent and JSON text")
        structured = parsed[0]
    if not isinstance(structured, dict): raise MCPError("structuredContent must be an object")
    if any(p != structured for p in parsed): raise MCPError("structuredContent disagrees with JSON text content")
    if "error" in structured: raise MCPError(f"structured tool error: {structured['error']!r}")
    return structured


def collect_mcp(plan, roots, binary, mcp_binary, output, *, name="cubeql-mcp",
                strategies=("lexical", "symbol", "semantic", "hybrid"), candidate_k=100,
                top_k=10, embedding="hash", encoder_config=None, model_dir=None,
                timeout_s=30, read_budget_bytes=262144):
    """Index each exact frozen repository once, then reuse one MCP process per repository."""
    file_index = ev._verify_plan(plan); spec = plan["spec"]
    ids = {r["id"] for r in spec["repositories"]}
    if not isinstance(roots, dict) or set(roots) != ids: raise MCPError("roots must name exactly the plan repository IDs")
    if not isinstance(name, str) or not name.strip(): raise MCPError("name must be nonempty")
    strategies = tuple(strategies)
    if not strategies or len(set(strategies)) != len(strategies) or any(s not in rc.STRATEGIES for s in strategies): raise MCPError("invalid strategies")
    if isinstance(top_k,bool) or not isinstance(top_k,int) or not 10 <= top_k <= 100: raise MCPError("top_k must be 10..100")
    if isinstance(candidate_k,bool) or not isinstance(candidate_k,int) or not top_k <= candidate_k <= 1000: raise MCPError("candidate_k must be top_k..1000")
    if isinstance(read_budget_bytes,bool) or not isinstance(read_budget_bytes,int) or read_budget_bytes < 0: raise MCPError("read_budget_bytes must be nonnegative")
    _positive_timeout(timeout_s)
    rc._check_config(embedding, encoder_config, model_dir)
    binary=Path(binary).resolve(strict=True); mcp_binary=Path(mcp_binary).absolute()
    if not binary.is_file() or not os.access(binary,os.X_OK): raise MCPError("binary must be executable")
    if not mcp_binary.is_file() or not os.access(mcp_binary,os.X_OK): raise MCPError("mcp_binary must be an executable regular file")
    mcp_profile=rc._fingerprint_path(mcp_binary)
    roots={rid:Path(p).resolve(strict=True) for rid,p in roots.items()}
    if any(not p.is_dir() for p in roots.values()): raise MCPError("repository roots must be directories")
    output=Path(output).absolute()
    if output.exists(): raise MCPError(f"output already exists: {output}")
    for root in roots.values():
        if root == output.resolve(strict=False) or root in output.resolve(strict=False).parents: raise MCPError("output cannot be inside frozen repository")
    before={rid:rc.source_inventory(root) for rid,root in roots.items()}
    expected={rid:{f["path"]:{"sha256":f["sha256"],"line_count":f["line_count"]} for f in plan["files"] if plan["format_version"]==1 or f["repository"]==rid} for rid in ids}
    if before != expected: raise MCPError("repository sources do not match frozen plan")
    profile=rc._profile_fingerprints(binary,encoder_config,model_dir); profile["mcp_binary"]=mcp_profile
    output.mkdir(parents=True,exist_ok=False); _write_new(output/"plan.json",plan); _write_new(output/"roots.json",{k:str(v) for k,v in roots.items()})
    results={strategy:[] for strategy in strategies}; sessions={}; snapshots={}; traces=[]; query_by_repo={rid:[q for q in spec["queries"] if q["repository"]==rid] for rid in ids}
    text_only=all(s in ("lexical","symbol") for s in strategies)
    try:
      for ri,rid in enumerate(sorted(ids)):
        snap=output/f"snapshot.{ri}.arrow"; cmd=[str(binary),"index",str(roots[rid]),"--output",str(snap),"--embedding",embedding,"--format","json"]
        if encoder_config: cmd += ["--encoder-config",str(Path(encoder_config).resolve())]
        if model_dir: cmd += ["--model-dir",str(Path(model_dir).resolve())]
        ix=rc.execute(cmd,output,f"index.{ri}",timeout_s)
        snapshots[rid]={"execution":ix,"path":snap.name}
        if ix["status"]!="ok" or not snap.is_file():
            sessions[rid]={"error":ix.get("error","index did not create snapshot"),"status":ix.get("status","failed")}; continue
        try:
          metadata=rc._load_json(output/ix["stdout"])
          if not isinstance(metadata,dict) or metadata.get("format_version")!=1: raise MCPError("index response must be format_version 1 JSON")
          si=metadata.get("snapshot")
          if si is not None and (not isinstance(si,dict) or si.get("root") not in (None,str(roots[rid])) or not isinstance(si.get("corpus_fingerprint"),str) or not isinstance(si.get("chunks"),int)):
            raise MCPError("index snapshot metadata invalid or for another root")
        except Exception as e:
          sessions[rid]={"error":f"invalid index output: {e}"}; continue
        snapshots[rid]["sha256"]=rc.sha256_file(snap)
        argv=[str(mcp_binary),str(snap)] + (["--text-only"] if text_only else [])
        if not text_only and embedding == "minilm": argv += ["--model-dir",str(Path(model_dir).resolve())]
        if not text_only and embedding == "process": argv += ["--encoder-config",str(Path(encoder_config).resolve())]
        # Preserve executable symlinks as argv text; never resolve the launch path.
        session=None; startup=time.monotonic()
        try:
          session=JSONLineSession(argv,output,f"mcp.{ri}",timeout_s)
          sessions[rid]=session
          init=session.request("initialize",{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"rq2-collector","version":"1"}})
          if not isinstance(init,dict) or init.get("protocolVersion")!="2025-11-25": raise MCPError("initialize returned unexpected protocolVersion")
          session.notify("notifications/initialized",{})
          tool_list=session.request("tools/list",{})
          if not isinstance(tool_list,dict) or not isinstance(tool_list.get("tools"),list) or not {"search_code","read_code"}.issubset({t.get("name") for t in tool_list["tools"] if isinstance(t,dict)}): raise MCPError("server lacks search_code/read_code")
          traces.append({"repository":rid,"startup_ms":(time.monotonic()-startup)*1000,"tools":tool_list,"rpc":session.records})
        except Exception as e:
          if session: session.close()
          sessions[rid]={"error":f"MCP startup failed: {e}","status":"timeout" if "deadline exceeded" in str(e).lower() else "failed","argv":argv,"startup_ms":(time.monotonic()-startup)*1000}
          continue
        snapshot_id=None
        for strategy in strategies:
          for q in query_by_repo[rid]:
            search_started=time.monotonic(); payload={"query":q["text"],"strategy":strategy,"top_k":top_k,"candidate_k":candidate_k}
            if "path_prefix" in q["filter"]: payload["path_prefix"]=q["filter"]["path_prefix"]
            try:
              native=_tool_result(session.request("tools/call",{"name":"search_code","arguments":payload}))
              elapsed=(time.monotonic()-search_started)*1000
              if native.get("format_version")!=1 or native.get("source")!="snapshot" or native.get("strategy")!=strategy or native.get("candidate_k")!=candidate_k or not isinstance(native.get("snapshot_id"),str) or not native["snapshot_id"] or not isinstance(native.get("hits"),list) or len(native["hits"])>top_k: raise MCPError("search response fields/budget invalid")
              if snapshot_id is not None and native["snapshot_id"]!=snapshot_id: raise MCPError("snapshot_id changed within session")
              snapshot_id=native["snapshot_id"]
              for key in ("embedding_ms","retrieval_ms"):
                if isinstance(native.get(key),bool) or not isinstance(native.get(key),(int,float)) or not math.isfinite(native[key]) or native[key]<0: raise MCPError(f"invalid native {key}")
              hits=[]; handles=set(); read_bytes=0; reads=[]; truncated=False
              for h in native["hits"]:
                if not isinstance(h,dict) or isinstance(h.get("chunk_id"),bool) or not isinstance(h.get("chunk_id"),int) or h["chunk_id"]<0: raise MCPError("invalid chunk handle")
                loc={k:h.get(k) for k in ("path","start_line","end_line")}
                if h["chunk_id"] in handles: raise MCPError("duplicate chunk handle")
                handles.add(h["chunk_id"])
                hits.append(loc)
              # Validate all paths/ranges/filters/duplicate spans before using any returned path.
              probe={"format_version":1,"plan_fingerprint":plan["fingerprint"],"name":"probe","tool":{"name":"mcp","revision":"probe"},"configuration":{},"candidate_k":top_k,"results":[{"query_id":q["id"],"status":"ok","hits":hits}]}
              ev._run_scores(probe,{"format_version":plan["format_version"],"queries":[q]},file_index,plan["fingerprint"])
              for h in native["hits"]:
                loc={k:h[k] for k in ("path","start_line","end_line")}
                remaining=read_budget_bytes-read_bytes
                if remaining<=0: truncated=True; continue
                offset=0; chunks=[]
                total_bytes=None
                while remaining>0:
                  if remaining < 4: truncated=True; break
                  requested=min(65536,remaining)
                  page=_tool_result(session.request("tools/call",{"name":"read_code","arguments":{"snapshot_id":snapshot_id,"chunk_id":h["chunk_id"],"offset":offset,"max_bytes":requested}}))
                  if (page.get("format_version")!=1 or page.get("source")!="snapshot" or page.get("snapshot_id")!=snapshot_id or page.get("chunk_id")!=h["chunk_id"] or
                      page.get("path")!=loc["path"] or page.get("start_line")!=loc["start_line"] or page.get("end_line")!=loc["end_line"] or
                      type(page.get("offset")) is not int or page.get("offset")!=offset or type(page.get("total_bytes")) is not int or page["total_bytes"]<0 or not isinstance(page.get("text"),str)):
                    raise MCPError("read_code handle/location/offset mismatch")
                  if total_bytes is None: total_bytes=page["total_bytes"]
                  elif total_bytes!=page["total_bytes"]: raise MCPError("read_code total_bytes changed between pages")
                  b=page["text"].encode("utf-8")
                  if len(b)>requested: raise MCPError("read_code exceeded requested byte count")
                  chunks.append(b); read_bytes+=len(b); remaining-=len(b)
                  nxt=page.get("next_offset")
                  if nxt is None:
                    if offset+len(b)!=total_bytes: raise MCPError("read_code ended before total_bytes")
                    break
                  if type(nxt) is not int or nxt<=offset or nxt!=offset+len(b) or nxt>=total_bytes: raise MCPError("read_code next_offset did not progress by UTF-8 bytes")
                  offset=nxt
                source=b"".join(chunks)
                expected_span=rc._inclusive_source_bytes((roots[rid]/loc["path"]).read_bytes(),loc["start_line"],loc["end_line"])
                if source and source not in expected_span: raise MCPError("read_code bytes differ from frozen source span")
                reads.append({"chunk_id":h["chunk_id"],"bytes":len(source),"sha256":hashlib.sha256(source).hexdigest()})
                if remaining<=0 and page.get("next_offset") is not None: truncated=True
              results[strategy].append({"query_id":q["id"],"status":"ok","hits":hits,"elapsed_ms":elapsed})
              traces.append({"repository":rid,"query_id":q["id"],"strategy":strategy,"embedding_ms":native["embedding_ms"],"retrieval_ms":native["retrieval_ms"],"reads":reads,"read_bytes":read_bytes,"read_truncated":truncated})
            except Exception as e:
              # Preserve the frozen denominator; transport failures poison the session.
              status="timeout" if "deadline exceeded" in str(e).lower() else "failed"
              results[strategy].append({"query_id":q["id"],"status":status,"error":str(e),"hits":[],"elapsed_ms":(time.monotonic()-search_started)*1000})
    finally:
      for s in sessions.values():
        if isinstance(s,JSONLineSession): s.close()
    runs=[]
    for strategy in strategies:
      run={"format_version":1,"plan_fingerprint":plan["fingerprint"],"name":f"{name}-{strategy}","tool":{"name":"cubeql-mcp","revision":profile["mcp_binary"]["sha256"]},"configuration":{"strategy":strategy,"embedding":embedding,"candidate_k":candidate_k,"top_k":top_k,"persistent_session_per_repository":True,"timing_boundary":REQUEST_TIMING_BOUNDARY},"candidate_k":top_k,"results":results[strategy]}
      # Fail all outstanding queries when startup/index failed.
      present={x["query_id"] for x in run["results"]}
      for q in spec["queries"]:
        if q["id"] not in present:
          rid=q["repository"]; failure=sessions.get(rid,{})
          reason=failure.get("error","index/MCP session startup failed") if isinstance(failure,dict) else "query not executed"
          status=failure.get("status","failed") if isinstance(failure,dict) else "failed"
          run["results"].append({"query_id":q["id"],"status":status,"error":reason,"hits":[]})
      runs.append(run)
    after={rid:rc.source_inventory(root) for rid,root in roots.items()}; profile_after=rc._profile_fingerprints(binary,encoder_config,model_dir); profile_after["mcp_binary"]=rc._fingerprint_path(mcp_binary)
    if after!=before or profile_after!=profile: raise MCPError("source, executable, model, or encoder changed during collection")
    report=ev.score(plan,runs)
    for run in runs: _write_new(output/f"run.{run['configuration']['strategy']}.json",run)
    _write_new(output/"report.json",report)
    # Records are copied before handles are closed; include their raw trace paths and RPC rows.
    traces.append({"sessions":{rid:dict(v) if isinstance(v,dict) else {"records":v.records,"files":[f"{v.stem}.stdin",f"{v.stem}.stdout",f"{v.stem}.stderr"]} for rid,v in sessions.items()},"index":snapshots})
    artifacts={p.name:{"bytes":p.stat().st_size,"sha256":rc.sha256_file(p)} for p in sorted(output.iterdir()) if p.is_file() and p.name!="manifest.json"}
    manifest={"format_version":1,"plan_fingerprint":plan["fingerprint"],"name":name,"binary":profile_after,"source_inventories":after,"snapshots":snapshots,"traces":traces,"artifacts":artifacts,"timing_boundary":REQUEST_TIMING_BOUNDARY}
    _write_new(output/"manifest.json",manifest)
    return {"runs":runs,"report":report,"manifest":manifest}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True); p.add_argument("--roots",required=True,help="JSON object mapping repository IDs to roots")
    p.add_argument("--binary",required=True); p.add_argument("--mcp-binary",required=True); p.add_argument("--output",required=True)
    p.add_argument("--name",default="cubeql-mcp"); p.add_argument("--strategy",action="append",dest="strategies")
    p.add_argument("--candidate-k",type=int,default=100); p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--embedding",choices=("hash","minilm","process"),default="hash"); p.add_argument("--encoder-config"); p.add_argument("--model-dir")
    p.add_argument("--timeout-seconds",type=float,default=30); p.add_argument("--read-budget-bytes",type=int,default=262144)
    a=p.parse_args(argv)
    try:
        plan=rc._load_json(Path(a.plan)); roots=rc._load_json(Path(a.roots))
        result=collect_mcp(plan,roots,a.binary,a.mcp_binary,a.output,name=a.name,strategies=a.strategies or rc.STRATEGIES,
                           candidate_k=a.candidate_k,top_k=a.top_k,embedding=a.embedding,encoder_config=a.encoder_config,
                           model_dir=a.model_dir,timeout_s=a.timeout_seconds,read_budget_bytes=a.read_budget_bytes)
        print(json.dumps({"output":str(Path(a.output).absolute()),"strategies":len(result["runs"]),"queries":len(plan["spec"]["queries"])},indent=2))
        return 0
    except (MCPError,rc.CollectionError,ev.EvaluationError,OSError) as e:
        print(f"repository_mcp: {e}",file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
