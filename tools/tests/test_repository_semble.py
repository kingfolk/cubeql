from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_collect as rc
import repository_eval as ev
import repository_semble as rs


def make_plan(root):
    inventory = rc.source_inventory(root)
    spec = {"format_version": 1, "name": "fixture", "split": "development",
            "repositories": [{"id": "repo", "revision": "fixture"}], "queries": [
                {"id": "q1", "repository": "repo", "category": "fixture", "text": "find target",
                 "filter": {"path_prefix": "src/"},
                 "anchors": [{"id": "a1", "path": "src/a.py", "start_line": 1, "end_line": 1}]},
                {"id": "q2", "repository": "repo", "category": "fixture", "text": "find target again",
                 "filter": {}, "anchors": [{"id": "a2", "path": "src/a.py", "start_line": 1, "end_line": 1}]},
            ]}
    files = [{"repository": rid, "path": path, **meta} for rid in ("repo",)
             for path, meta in inventory.items()]
    unsigned = {"format_version": 1, "spec": spec, "spec_sha256": ev._digest(spec), "files": files}
    return {**unsigned, "fingerprint": ev._digest(unsigned)}


class FakeSession:
    constructed = 0
    calls = []
    mode = "ok"
    commands = []
    model_path = "/local/model"
    mutation = None

    def __init__(self, command, output_dir, stem, timeout_s=30, env=None):
        type(self).constructed += 1
        type(self).commands.append(command)
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def request(self, method, params):
        self.calls.append((method, params))
        if method == "initialize":
            if type(self).mode == "badinit":
                return {"protocol_version": 2, "build_ms": float("nan"), "indexed_files": 1,
                        "total_chunks": 1, "model_path": str(type(self).model_path),
                        "python_version": "test", "package_versions": {}}
            return {"protocol_version": 1, "semble_version": "test", "build_ms": 4.5,
                    "indexed_files": 1, "total_chunks": 1, "model_path": str(type(self).model_path),
                    "python_version": "test", "package_versions": {}}
        if method == "shutdown":
            return {"ok": True}
        if type(self).mode == "timeout":
            raise RuntimeError("JSON-RPC request deadline exceeded")
        if type(self).mutation:
            path, data = type(self).mutation
            Path(path).write_bytes(data)
            type(self).mutation = None
        if type(self).mode in ("corrupt", "badspan", "badfilter"):
            path, start, end = "../escape.py", 1, 1
            if type(self).mode == "badspan":
                path, start, end = "src/a.py", 2, 1
            return {"payload": {"query": params["query"], "results": [
                {"file_path": path, "start_line": start, "end_line": end, "score": 0.5}]},
                "search_ms": 1.2, "applied_filter": {"wrong": "filter"} if type(self).mode == "badfilter" else ({"path_prefix": params["path_prefix"]} if params["path_prefix"] is not None else {}),
                "internal_candidate_budget": params["top_k"] * 5}
        return {"payload": {"query": params["query"], "results": [
            {"file_path": "src/a.py", "start_line": 1, "end_line": 1, "score": 0.5}]},
            "search_ms": 1.2, "applied_filter": {"path_prefix": params["path_prefix"]} if params["path_prefix"] is not None else {},
            "internal_candidate_budget": params["top_k"] * 5}


class RepositorySembleTests(unittest.TestCase):
    def test_worker_filters_prefix_before_topk_and_skips_empty_match(self):
        calls = []

        class FakeIndex:
            chunks = [types.SimpleNamespace(file_path="pkg/a.py"), types.SimpleNamespace(file_path="src/a.py")]
            stats = types.SimpleNamespace(indexed_files=2, total_chunks=2)

            @classmethod
            def from_path(cls, root, model_path, content):
                assert os.environ["HF_HUB_OFFLINE"] == "1"
                assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
                assert os.environ["SEMBLE_INDEX_WORKERS"] == "0"
                assert os.environ["SEMBLE_CACHE_LOCATION"] == str(Path("/tmp/cache").resolve())
                return cls()

            def search(self, query, top_k=10, filter_paths=None):
                calls.append((query, top_k, filter_paths))
                return []

        fake_semble = types.ModuleType("semble")
        fake_semble.SembleIndex, fake_semble.ContentType = FakeIndex, types.SimpleNamespace(CODE="code")
        fake_semble.__version__ = "test"
        fake_utils = types.ModuleType("semble.utils")
        fake_utils.format_results = lambda q, found: {"query": q, "results": []}
        old = {k: sys.modules.get(k) for k in ("semble", "semble.utils")}
        old_bytecode = sys.dont_write_bytecode
        sys.modules["semble"], sys.modules["semble.utils"] = fake_semble, fake_utils
        stdin = io.StringIO("\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "search", "params": {"query": "x", "top_k": 10, "path_prefix": "none/"}}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "search", "params": {"query": "x", "top_k": 10, "path_prefix": "src/"}}),
            json.dumps({"jsonrpc": "2.0", "id": 4, "method": "search", "params": {"query": "x", "top_k": 10, "path_prefix": None}}),
            json.dumps({"jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": {}}), ""]))
        stdout = io.StringIO()
        try:
            with mock.patch.object(sys, "stdin", stdin), mock.patch.object(sys, "stdout", stdout):
                rs._worker("/tmp/source", "/tmp/root", "/tmp/model", "/tmp/cache")
        finally:
            sys.dont_write_bytecode = old_bytecode
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(messages[1]["result"]["payload"]["results"], [])
        self.assertEqual([c for c in calls if c[0] == "x"], [("x", 10, ["src/a.py"]), ("x", 10, None)])

    def test_collector_reuses_one_session_and_scores_native_payloads(self):
        FakeSession.constructed, FakeSession.calls = 0, []
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            (root / "src").mkdir(parents=True)
            (root / "src" / "a.py").write_text("target\n", encoding="utf-8")
            source = base / "source"
            (source / "src" / "semble").mkdir(parents=True)
            model = base / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"local fixture")
            plan = make_plan(root)
            FakeSession.model_path = model.resolve()
            FakeSession.commands = []
            out = base / "result"
            with mock.patch.object(rs, "JSONLineSession", FakeSession):
                result = rs.collect_semble(plan, {"repo": root}, sys.executable, source, model, out,
                                           top_k=10, source_revision="fixture-commit")
            self.assertEqual(FakeSession.constructed, 1)
            self.assertEqual([c[0] for c in FakeSession.calls], ["initialize", "search", "search", "shutdown"])
            self.assertEqual([r["status"] for r in result["run"]["results"]], ["ok", "ok"])
            self.assertEqual(result["run"]["candidate_k"], 10)
            self.assertTrue((out / "payloads" / "000001.json").exists())
            self.assertTrue(all("-B" in command for command in FakeSession.commands))

    def test_query_ids_cannot_escape_payload_directory(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            (root / "src").mkdir(parents=True)
            (root / "src" / "a.py").write_text("target\n", encoding="utf-8")
            source = base / "source"
            (source / "src" / "semble").mkdir(parents=True)
            model = base / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"local fixture")
            plan = make_plan(root)
            plan["spec"]["queries"][0]["id"] = "../../outside"
            plan["spec_sha256"] = ev._digest(plan["spec"])
            plan["fingerprint"] = ev._digest({k: v for k, v in plan.items() if k != "fingerprint"})
            FakeSession.model_path, FakeSession.mode = model.resolve(), "ok"
            out = base / "result"
            with mock.patch.object(rs, "JSONLineSession", FakeSession):
                rs.collect_semble(plan, {"repo": root}, sys.executable, source, model, out,
                                  top_k=10, source_revision="fixture-commit")
            self.assertTrue((out / "payloads" / "000001.json").is_file())
            self.assertFalse((base / "outside.json").exists())

    def test_timeout_and_corrupt_native_payloads_receive_no_partial_credit(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            (root / "src").mkdir(parents=True)
            (root / "src" / "a.py").write_text("target\n", encoding="utf-8")
            source = base / "source"
            (source / "src" / "semble").mkdir(parents=True)
            model = base / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"local fixture")
            plan = make_plan(root)
            FakeSession.model_path = model.resolve()
            for mode, expected in (("timeout", "timeout"), ("corrupt", "failed"),
                                   ("badspan", "failed"), ("badfilter", "failed")):
                FakeSession.constructed, FakeSession.calls, FakeSession.mode = 0, [], mode
                out = base / f"result-{mode}"
                with mock.patch.object(rs, "JSONLineSession", FakeSession):
                    result = rs.collect_semble(plan, {"repo": root}, sys.executable, source, model, out,
                                               top_k=10, source_revision="fixture-commit")
                self.assertTrue(all(row["status"] == expected and not row["hits"]
                                    for row in result["run"]["results"]))
            FakeSession.mode = "ok"

    def test_corrupt_native_json_is_rejected(self):
        with self.assertRaises(Exception):
            rs.peer._native_hits(b'{"query":', "find target", 10, "q1")

    def test_invalid_initialize_metadata_fails_every_query(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            (root / "src").mkdir(parents=True)
            (root / "src" / "a.py").write_text("target\n", encoding="utf-8")
            source = base / "source"
            (source / "src" / "semble").mkdir(parents=True)
            model = base / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"local fixture")
            plan = make_plan(root)
            FakeSession.model_path, FakeSession.mode = model.resolve(), "badinit"
            out = base / "result"
            with mock.patch.object(rs, "JSONLineSession", FakeSession):
                result = rs.collect_semble(plan, {"repo": root}, sys.executable, source, model, out,
                                           top_k=10, source_revision="fixture-commit")
            self.assertEqual([r["status"] for r in result["run"]["results"]], ["failed", "failed"])

    def test_source_and_model_mutations_prevent_report_creation(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "root"
            (root / "src").mkdir(parents=True)
            source_file = root / "src" / "a.py"
            source_file.write_text("target\n", encoding="utf-8")
            source = base / "source"
            (source / "src" / "semble").mkdir(parents=True)
            model = base / "model"
            model.mkdir()
            model_file = model / "weights.bin"
            model_file.write_bytes(b"local fixture")
            FakeSession.model_path, FakeSession.mode = model.resolve(), "ok"
            for kind, changed in (("source", source_file), ("model", model_file)):
                plan = make_plan(root)
                before = changed.read_bytes()
                FakeSession.mutation = (changed, b"changed bytes\n")
                out = base / f"result-{kind}"
                with mock.patch.object(rs, "JSONLineSession", FakeSession):
                    with self.assertRaises(rs.SembleCollectionError):
                        rs.collect_semble(plan, {"repo": root}, sys.executable, source, model, out,
                                          top_k=10, source_revision="fixture-commit")
                self.assertFalse((out / "report.json").exists())
                changed.write_bytes(before)


if __name__ == "__main__":
    unittest.main()
