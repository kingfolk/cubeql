"""Manual offline importer fixtures; none of these fixtures are Semble executions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_eval as ev
import repository_peer as peer


class RepositoryPeerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "src" / "answer.py").write_text("".join(f"line {i}\n" for i in range(1, 21)), encoding="utf-8")
        spec = {"format_version": 1, "name": "manual fixture", "split": "validation",
                "repositories": [{"id": "sample", "revision": "fixture"}],
                "queries": [{"id": "q1", "repository": "sample", "category": "fixture", "text": "find answer",
                             "filter": {"path_prefix": "src/"},
                             "anchors": [{"id": "a1", "path": "src/answer.py", "start_line": 2, "end_line": 3}]}]}
        self.plan = ev.freeze(spec, self.root)
        self.manifest_path = self.base / "manifest.json"
        self.payload_path = self.base / "payload.json"
        self.output = self.base / "imported"
        self.valid_payload({"query": "find answer", "results": [
            {"file_path": "src/answer.py", "start_line": 1, "end_line": 4, "score": 0.75, "content": "line 1\nline 2"}]})
        self.manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def valid_payload(self, value):
        self.payload_path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")

    def manifest(self, row_update=None, manifest_update=None):
        row = {"query_id": "q1", "query_text": "find answer", "applied_filter": {"path_prefix": "src/"}, "status": "ok", "payload": "payload.json"}
        if row_update:
            row.update(row_update)
        manifest = {"format_version": 1, "plan_fingerprint": self.plan["fingerprint"], "name": "semble-import",
                    "tool": {"name": "semble", "revision": "manual-source-pin"},
                    "configuration": {"snapshot": "declared only"}, "candidate_k": 10, "results": [row]}
        if manifest_update:
            manifest.update(manifest_update)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_import_preserves_raw_evidence_and_scores_inclusive_span(self):
        original = self.payload_path.read_bytes()
        result = peer.import_semble(self.plan, self.manifest_path, self.output)
        self.assertEqual(result["run"]["results"][0]["hits"], [{"path": "src/answer.py", "start_line": 1, "end_line": 4}])
        self.assertEqual(result["report"]["runs"][0]["overall"]["recall_at_10"], 1.0)
        evidence = self.output / result["provenance"]["evidence"][0]["copied_path"]
        self.assertEqual(evidence.read_bytes(), original)
        self.assertEqual((self.output / "manifest.json").read_bytes(), self.manifest_path.read_bytes())
        self.assertEqual(json.loads((self.output / "plan.json").read_text(encoding="utf-8")), self.plan)
        self.assertEqual(result["provenance"]["evidence"][0]["sha256"], __import__("hashlib").sha256(original).hexdigest())
        self.assertTrue(self.manifest_path.exists())
        self.assertTrue(self.payload_path.exists())

    def test_complete_denominator_failed_result_retained_and_paired_baseline(self):
        self.manifest({"status": "failed", "error": "fixture failure", "payload": "diagnostic.bin"})
        (self.base / "diagnostic.bin").write_bytes(b"not parsed as JSON\x00")
        baseline = {"format_version": 1, "plan_fingerprint": self.plan["fingerprint"], "name": "baseline",
                    "tool": {"name": "fixture", "revision": "manual"}, "configuration": {}, "candidate_k": 10,
                    "results": [{"query_id": "q1", "status": "ok", "hits": []}]}
        baseline_path = self.base / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        result = peer.import_semble(self.plan, self.manifest_path, self.output, [baseline_path])
        self.assertEqual(result["run"]["results"][0]["status"], "failed")
        self.assertEqual(result["report"]["runs"][1]["overall"]["query_count"], 1)
        self.assertEqual(result["report"]["paired_comparisons"][0]["baseline"], "baseline")
        self.assertEqual((self.output / "evidence/000001.payload").read_bytes(), b"not parsed as JSON\x00")

    def test_duplicate_and_missing_manifest_query_ids_fail(self):
        self.manifest(manifest_update={"results": []})
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)
        self.manifest(manifest_update={"results": [
            {"query_id": "q1", "query_text": "find answer", "applied_filter": {"path_prefix": "src/"}, "status": "failed", "error": "x"},
            {"query_id": "q1", "query_text": "find answer", "applied_filter": {"path_prefix": "src/"}, "status": "failed", "error": "y"}]})
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)

    def test_query_and_filter_must_match(self):
        for update in ({"query_text": "changed"}, {"applied_filter": None}, {"applied_filter": "src/"}, {"applied_filter": {"other": "src/"}}):
            self.manifest(update)
            with self.assertRaises(peer.PeerImportError):
                peer.import_semble(self.plan, self.manifest_path, self.output)

    def test_bad_native_echo_paths_spans_scores_and_duplicate_json_fail(self):
        bad_payloads = [
            {"query": "other", "results": []},
            {"query": "find answer", "results": [{"file_path": "src/answer.py", "start_line": 2, "end_line": 3, "score": float("nan")}]},
            {"query": "find answer", "results": [{"file_path": "src/answer.py", "start_line": True, "end_line": 3, "score": 1}]},
            {"query": "find answer", "results": [{"file_path": "src/answer.py", "start_line": 3, "end_line": 2, "score": 1}]},
            {"query": "find answer", "results": [{"file_path": "src/answer.py", "start_line": 1, "end_line": 4, "score": 1, "content": "ok", "extra": 0}]},
            {"query": "find answer", "results": [{"file_path": "repo/src/answer.py", "start_line": 1, "end_line": 4, "score": 1}]},
            {"query": "find answer", "repos": {"repo": []}, "results": []},
            {"query": "find answer", "results": [{"file_path": "src/answer.py", "start_line": 1, "end_line": 4, "score": 1}, {"file_path": "src/answer.py", "start_line": 1, "end_line": 4, "score": 2}]},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                self.valid_payload(payload)
                with self.assertRaises(peer.PeerImportError):
                    peer.import_semble(self.plan, self.manifest_path, self.output)
        self.payload_path.write_text('{"query":"find answer","query":"find answer","results":[]}', encoding="utf-8")
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)

    def test_legitimate_repo_directory_name_is_accepted_without_prefix_stripping(self):
        (self.root / "repo").mkdir()
        (self.root / "repo" / "answer.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
        spec = {"format_version": 1, "name": "repo path fixture", "split": "validation",
                "repositories": [{"id": "sample", "revision": "fixture"}],
                "queries": [{"id": "q1", "repository": "sample", "category": "fixture", "text": "find answer",
                             "filter": {},
                             "anchors": [{"id": "a1", "path": "repo/answer.py", "start_line": 2, "end_line": 2}]}]}
        plan = ev.freeze(spec, self.root)
        self.valid_payload({"query": "find answer", "results": [
            {"file_path": "repo/answer.py", "start_line": 1, "end_line": 3, "score": 0.5}]})
        self.manifest_path.write_text(json.dumps({"format_version": 1, "plan_fingerprint": plan["fingerprint"],
            "name": "repo-dir", "tool": {"name": "semble", "revision": "manual-source-pin"},
            "configuration": {}, "candidate_k": 10, "results": [{"query_id": "q1", "query_text": "find answer",
            "applied_filter": {}, "status": "ok", "payload": "payload.json"}]}), encoding="utf-8")
        imported = peer.import_semble(plan, self.manifest_path, self.base / "repo-import")
        self.assertEqual(imported["run"]["results"][0]["hits"][0]["path"], "repo/answer.py")

    def test_payload_escape_and_symlink_rejected(self):
        self.manifest({"payload": "../outside-peer.json"})
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)
        (self.base / "linked.json").symlink_to(self.payload_path)
        self.manifest({"payload": "linked.json"})
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)

    def test_native_empty_no_result_is_success_but_other_error_is_rejected(self):
        self.payload_path.write_text('{"error":"No results found."}', encoding="utf-8")
        imported = peer.import_semble(self.plan, self.manifest_path, self.output)
        self.assertEqual(imported["run"]["results"][0]["status"], "ok")
        self.assertEqual(imported["run"]["results"][0]["hits"], [])
        self.output = self.base / "other"
        self.payload_path.write_text('{"error":"some other error"}', encoding="utf-8")
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)

    def test_output_create_new_and_no_overwrite(self):
        self.output.mkdir()
        with self.assertRaises(ev.EvaluationError):
            peer.import_semble(self.plan, self.manifest_path, self.output)


if __name__ == "__main__":
    unittest.main()
