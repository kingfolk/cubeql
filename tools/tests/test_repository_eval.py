import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "repository_eval.py"
spec = importlib.util.spec_from_file_location("repository_eval", MODULE)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def make_spec():
    return {"format_version": 1, "name": "tiny", "split": "development",
            "repositories": [{"id": "repo", "revision": "commit-declared"}],
            "queries": [
                {"id": "q1", "repository": "repo", "category": "api", "text": "find alpha", "filter": {},
                 "anchors": [{"id": "a1", "path": "src/alpha.rs", "start_line": 2, "end_line": 3},
                             {"id": "a2", "path": "docs/guide.md", "start_line": 1, "end_line": 1}]},
                {"id": "q2", "repository": "repo", "category": "api", "text": "find beta", "filter": {"path_prefix": "src/"},
                 "anchors": [{"id": "b1", "path": "src/beta.rs", "start_line": 1, "end_line": 1}]},
                {"id": "q3", "repository": "repo", "category": "docs", "text": "find gamma", "filter": {"path_prefix": "docs/g"},
                 "anchors": [{"id": "g1", "path": "docs/guide.md", "start_line": 3, "end_line": 3}]},
            ]}


def make_run(plan, name="baseline", candidate_k=10, results=None):
    if results is None:
        results = [
            {"query_id": "q1", "status": "ok", "hits": [
                {"path": "src/alpha.rs", "start_line": 1, "end_line": 2},  # partial overlap: not a match
                {"path": "src/alpha.rs", "start_line": 2, "end_line": 4},  # contains anchor a1
                {"path": "docs/guide.md", "start_line": 1, "end_line": 1}]},
            {"query_id": "q2", "status": "ok", "hits": []},
            {"query_id": "q3", "status": "failed", "error": "unsupported filter", "hits": []},
        ]
    return {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "name": name,
            "tool": {"name": "test", "revision": "1"}, "configuration": {"mode": "small"},
            "candidate_k": candidate_k, "results": results}


class RepositoryEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "src/alpha.rs").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
        (self.root / "src/beta.rs").write_text("beta\n", encoding="utf-8")
        (self.root / "docs/guide.md").write_text("one\ntwo\ngamma\n", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git/config").write_text("private", encoding="utf-8")
        self.spec = make_spec()
        self.plan = ev.freeze(self.spec, self.root)

    def test_freeze_inventory_and_lf_line_count(self):
        paths = {f["path"] for f in self.plan["files"]}
        self.assertEqual(paths, {"src/alpha.rs", "src/beta.rs", "docs/guide.md"})
        self.assertEqual(next(f["line_count"] for f in self.plan["files"] if f["path"] == "src/alpha.rs"), 4)

    def test_line_count_uses_lf_not_unicode_or_control_separators(self):
        (self.root / "odd.txt").write_text("a\vb\u2028c\r\nd", encoding="utf-8")
        plan = ev.freeze(self.spec, self.root)
        self.assertEqual(next(f["line_count"] for f in plan["files"] if f["path"] == "odd.txt"), 2)

    def test_exact_containment_cross_file_recall_and_failure_denominator(self):
        report = ev.score(self.plan, [make_run(self.plan)])
        overall = report["runs"][0]["overall"]
        self.assertAlmostEqual(report["runs"][0]["queries"]["q1"]["recall_at"]["10"], 1.0)
        self.assertEqual(report["runs"][0]["queries"]["q1"]["anchor_ranks"], {"a1": 2, "a2": 3})
        self.assertAlmostEqual(report["runs"][0]["queries"]["q1"]["reciprocal_rank"], 0.5)
        # q1 recall=1, q2=0, failed q3=0; failed queries remain in the macro denominator.
        self.assertAlmostEqual(overall["recall_at_10"], 1 / 3)
        self.assertAlmostEqual(overall["all_required_at_10"], 1 / 3)
        self.assertEqual(overall["status_counts"], {"ok": 2, "failed": 1, "unsupported": 0, "timeout": 0})
        self.assertAlmostEqual(overall["coverage"], 2 / 3)
        self.assertAlmostEqual(overall["mrr"], 1 / 6)
        partial = make_run(self.plan)
        partial["results"][0]["hits"] = [{"path": "src/alpha.rs", "start_line": 1, "end_line": 2}]
        self.assertEqual(ev.score(self.plan, [partial])["runs"][0]["queries"]["q1"]["recall_at"]["10"], 0.0)

    def test_cutoffs_null_when_candidate_window_too_small(self):
        report = ev.score(self.plan, [make_run(self.plan, candidate_k=10)])
        q = report["runs"][0]["queries"]["q1"]
        self.assertIsNone(q["recall_at"]["50"])
        self.assertIsNone(report["runs"][0]["overall"]["recall_at_100"])

    def test_paired_tie_and_regression_summary(self):
        baseline = make_run(self.plan)
        changed = copy.deepcopy(baseline)
        changed["name"] = "candidate"
        changed["results"][0]["hits"] = []
        report = ev.score(self.plan, [baseline, changed])
        paired = report["paired_comparisons"][0]
        self.assertEqual(paired["recall_at_10_outcomes"], {"wins": 0, "ties": 2, "losses": 1})
        self.assertAlmostEqual(paired["macro_delta"]["recall_at_10"], -1 / 3)
        changed["candidate_k"] = 20
        report = ev.score(self.plan, [baseline, changed])
        self.assertIsNone(report["paired_comparisons"][0]["per_query"]["q1"]["mrr"])

    def test_filter_uses_literal_startswith_for_slash_and_partial_filename(self):
        run = make_run(self.plan)
        run["results"][1]["hits"] = [{"path": "src/beta.rs", "start_line": 1, "end_line": 1}]
        run["results"][1]["status"] = "ok"
        run["results"][2] = {"query_id": "q3", "status": "ok", "hits": [
            {"path": "docs/guide.md", "start_line": 3, "end_line": 3}]}
        self.assertEqual(ev.score(self.plan, [run])["runs"][0]["queries"]["q2"]["recall_at"]["10"], 1.0)
        self.assertEqual(ev.score(self.plan, [run])["runs"][0]["queries"]["q3"]["recall_at"]["10"], 1.0)

    def test_cutoff_boundary_and_multiple_hits_do_not_union_partial_evidence(self):
        (self.root / "long.rs").write_text("x\n" * 100)
        spec = copy.deepcopy(self.spec)
        spec["queries"] = [spec["queries"][0]]
        spec["queries"][0]["anchors"] = [
            {"id": "last", "path": "long.rs", "start_line": 50, "end_line": 51}]
        plan = ev.freeze(spec, self.root)
        hits = [{"path": "long.rs", "start_line": n, "end_line": n} for n in range(1, 50)]
        hits.append({"path": "long.rs", "start_line": 50, "end_line": 51})
        run = make_run(plan, candidate_k=50, results=[{"query_id": "q1", "status": "ok", "hits": hits}])
        metrics = ev.score(plan, [run])["runs"][0]["overall"]
        self.assertEqual(metrics["recall_at_10"], 0)
        self.assertEqual(metrics["recall_at_50"], 1)
        self.assertAlmostEqual(metrics["mrr"], 1 / 50)
        run["results"][0]["hits"] = [
            {"path": "long.rs", "start_line": 50, "end_line": 50},
            {"path": "long.rs", "start_line": 51, "end_line": 51}]
        self.assertEqual(ev.score(plan, [run])["runs"][0]["overall"]["recall_at_50"], 0)

    def test_all_failure_statuses_keep_full_denominator_and_order_is_stable(self):
        run = make_run(self.plan)
        for result, status in zip(run["results"], ["failed", "timeout", "unsupported"]):
            result.update(status=status, hits=[], error="retained failure")
        report = ev.score(self.plan, [run])["runs"][0]
        self.assertEqual(report["overall"]["recall_at_10"], 0)
        self.assertEqual(report["overall"]["query_count"], 3)
        self.assertEqual(report["overall"]["coverage"], 0)
        run["results"].reverse()
        self.assertEqual(ev.score(self.plan, [run])["runs"][0]["overall"], report["overall"])

    def test_strict_schema_and_identity_fail_closed(self):
        mutations = [
            lambda r: r.update(format_version=1.0),
            lambda r: r.update(candidate_k=True),
            lambda r: r.update(plan_fingerprint="0" * 64),
            lambda r: r.update(unknown=True),
            lambda r: r["results"][0].update(query_id=[]),
            lambda r: r["results"].append(copy.deepcopy(r["results"][0])),
            lambda r: r["results"][0]["hits"].append(copy.deepcopy(r["results"][0]["hits"][0])),
            lambda r: r["results"][0]["hits"][0].update(end_line=999),
            lambda r: r.update(configuration={"value": float("inf")}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                run = make_run(self.plan)
                mutate(run)
                with self.assertRaises(ev.EvaluationError):
                    ev.score(self.plan, [run])

    def test_cli_roundtrip_and_no_overwrite(self):
        folder = Path(self.tmp.name)
        spec_path, plan_path = folder / "spec.json", folder / "plan.json"
        run_path, output = folder / "run.json", folder / "report.json"
        spec_path.write_text(json.dumps(self.spec))
        self.assertEqual(ev.main(["freeze", "--spec", str(spec_path), "--root", str(self.root), "--output", str(plan_path)]), 0)
        run_path.write_text(json.dumps(make_run(ev._load(plan_path))))
        args = ["score", "--plan", str(plan_path), "--run", str(run_path), "--output", str(output)]
        self.assertEqual(ev.main(args), 0)
        original = output.read_bytes()
        with patch("sys.stderr"):
            self.assertEqual(ev.main(args), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_unreadable_inventory_fails_instead_of_silently_omitting_files(self):
        def failed_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("denied fixture"))
            return iter(())
        with patch.object(ev.os, "walk", failed_walk):
            with self.assertRaises(ev.EvaluationError):
                ev.freeze(self.spec, self.root)

    def test_rejects_plan_tampering_and_bad_inputs(self):
        bad = copy.deepcopy(self.plan)
        bad["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ev.EvaluationError):
            ev.score(bad, [make_run(self.plan)])
        run = make_run(self.plan)
        run["results"].pop()
        with self.assertRaises(ev.EvaluationError):
            ev.score(self.plan, [run])
        run = make_run(self.plan)
        run["results"][0]["hits"].append({"path": "../etc/passwd", "start_line": 1, "end_line": 1})
        with self.assertRaises(ev.EvaluationError):
            ev.score(self.plan, [run])
        run = make_run(self.plan)
        run["results"][1]["status"] = "ok"
        run["results"][1]["hits"] = [{"path": "docs/guide.md", "start_line": 1, "end_line": 1}]
        with self.assertRaises(ev.EvaluationError):
            ev.score(self.plan, [run])
        with self.assertRaises(ev.EvaluationError):
            ev.score(self.plan, [make_run(self.plan, name="same"), make_run(self.plan, name="same")])

    def test_rejects_bool_integer_duplicate_ids_and_nonfinite_values(self):
        bad = copy.deepcopy(self.spec)
        bad["queries"][0]["anchors"][0]["start_line"] = True
        with self.assertRaises(ev.EvaluationError):
            ev.freeze(bad, self.root)
        bad = copy.deepcopy(self.spec)
        bad["queries"][1]["id"] = "q1"
        with self.assertRaises(ev.EvaluationError):
            ev.freeze(bad, self.root)
        run = make_run(self.plan)
        run["results"][0]["elapsed_ms"] = float("nan")
        with self.assertRaises(ev.EvaluationError):
            ev.score(self.plan, [run])

    def test_cli_loader_rejects_duplicate_json_keys(self):
        path = Path(self.tmp.name) / "duplicate.json"
        path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        with self.assertRaises(ev.EvaluationError):
            ev._load(path)

    def test_freeze_rejects_symlink_file_and_multi_repository_spec(self):
        (self.root / "linked").symlink_to(self.root / "src/alpha.rs")
        with self.assertRaises(ev.EvaluationError):
            ev.freeze(self.spec, self.root)
        (self.root / "linked").unlink()
        bad = copy.deepcopy(self.spec)
        bad["repositories"].append({"id": "other", "revision": "x"})
        with self.assertRaises(ev.EvaluationError):
            ev.freeze(bad, self.root)


if __name__ == "__main__":
    unittest.main()
