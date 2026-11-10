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


def make_v2_spec(alpha_count=3):
    repos = [{"id": "alpha", "revision": "a"}, {"id": "beta", "revision": "b"}]
    queries = []
    for repo_id, count in (("alpha", alpha_count), ("beta", 1)):
        for i in range(count):
            queries.append({"id": f"{repo_id}-{i}", "repository": repo_id, "category": "api",
                            "text": "find symbol", "filter": {},
                            "anchors": [{"id": "same-anchor", "path": "src/lib.rs", "start_line": 1, "end_line": 1}]})
    return {"format_version": 2, "name": "multi", "split": "development", "repositories": repos, "queries": queries}


def make_v2_run(plan, name="baseline", successes=(), candidate_k=10):
    results = []
    for q in plan["spec"]["queries"]:
        if q["id"] in successes:
            results.append({"query_id": q["id"], "status": "ok",
                            "hits": [{"path": "src/lib.rs", "start_line": 1, "end_line": 1}]})
        else:
            results.append({"query_id": q["id"], "status": "failed", "error": "fixture miss", "hits": []})
    return make_run(plan, name=name, candidate_k=candidate_k, results=results)


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

    def _v2_roots(self):
        alpha, beta = Path(self.tmp.name) / "alpha", Path(self.tmp.name) / "beta"
        alpha.mkdir(); beta.mkdir()
        (alpha / "src").mkdir(); (beta / "src").mkdir()
        (alpha / "src/lib.rs").write_text("alpha\n", encoding="utf-8")
        (beta / "src/lib.rs").write_text("beta\n", encoding="utf-8")
        return {"alpha": alpha, "beta": beta}

    def test_v2_same_paths_and_anchor_ids_are_repository_scoped(self):
        roots = self._v2_roots()
        plan = ev.freeze(make_v2_spec(), roots)
        entries = {(f["repository"], f["path"]): f for f in plan["files"]}
        self.assertEqual(entries[("alpha", "src/lib.rs")]["sha256"], ev.hashlib.sha256(b"alpha\n").hexdigest())
        self.assertEqual(entries[("beta", "src/lib.rs")]["sha256"], ev.hashlib.sha256(b"beta\n").hexdigest())
        self.assertEqual(plan["files"], sorted(plan["files"], key=lambda f: (0 if f["repository"] == "alpha" else 1, f["path"])))

    def test_v2_repository_macro_equal_weights_and_query_macro(self):
        roots = self._v2_roots()
        plan = ev.freeze(make_v2_spec(), roots)
        success_ids = {q["id"] for q in plan["spec"]["queries"] if q["repository"] == "alpha"}
        report = ev.score(plan, [make_v2_run(plan, successes=success_ids)])
        run = report["runs"][0]
        self.assertEqual(report["format_version"], 2)
        self.assertEqual(run["overall"]["recall_at_10"], 0.75)
        self.assertEqual(run["overall"]["coverage"], 0.75)
        self.assertEqual(run["repository_macro"]["recall_at_10"], 0.5)
        self.assertEqual(run["repository_macro"]["coverage"], 0.5)
        self.assertEqual(run["repository_macro"]["repository_count"], 2)
        self.assertEqual(run["repository_macro"]["query_count"], 4)

    def test_v2_paired_repository_gains_regressions_and_null_mrr(self):
        roots = self._v2_roots()
        plan = ev.freeze(make_v2_spec(), roots)
        alpha_ids = {q["id"] for q in plan["spec"]["queries"] if q["repository"] == "alpha"}
        base = make_v2_run(plan, "base", successes=alpha_ids)
        candidate_successes = {"alpha-0", "alpha-1", "beta-0"}
        candidate = make_v2_run(plan, "candidate", successes=candidate_successes, candidate_k=20)
        paired = ev.score(plan, [base, candidate])["paired_comparisons"][0]
        self.assertEqual(paired["by_repository"]["alpha"]["recall_at_10_outcomes"],
                         {"wins": 0, "ties": 2, "losses": 1})
        self.assertEqual(paired["by_repository"]["beta"]["recall_at_10_outcomes"],
                         {"wins": 1, "ties": 0, "losses": 0})
        self.assertAlmostEqual(paired["by_repository"]["alpha"]["macro_delta"]["recall_at_10"], -1 / 3)
        self.assertEqual(paired["by_repository"]["beta"]["macro_delta"]["recall_at_10"], 1.0)
        self.assertAlmostEqual(paired["repository_macro_delta"]["recall_at_10"], 1 / 3)
        self.assertAlmostEqual(paired["repository_macro_delta"]["all_required_at_10"], 1 / 3)
        self.assertEqual(paired["macro_delta"]["recall_at_10"], 0.0)
        self.assertEqual(paired["macro_delta"]["all_required_at_10"], 0.0)
        self.assertIsNone(paired["by_repository"]["alpha"]["macro_delta"]["mrr"])
        self.assertIsNone(paired["repository_macro_delta"]["mrr"])

    def test_v2_rejects_missing_extra_duplicate_empty_and_unknown_repositories(self):
        roots = self._v2_roots()
        spec = make_v2_spec()
        for bad_roots in ({"alpha": roots["alpha"]}, {**roots, "extra": roots["alpha"]}):
            with self.subTest(roots=bad_roots), self.assertRaises(ev.EvaluationError):
                ev.freeze(spec, bad_roots)
        for mutate in (
            lambda s: s.update(repositories=[]),
            lambda s: s["repositories"].append(copy.deepcopy(s["repositories"][0])),
            lambda s: s["queries"].pop(),
            lambda s: s["queries"][0].update(repository="unknown"),
        ):
            bad = copy.deepcopy(spec); mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ev.EvaluationError):
                ev.freeze(bad, roots)

    def test_v2_rejects_two_root_aliases_for_same_checkout(self):
        roots = self._v2_roots()
        with self.assertRaises(ev.EvaluationError):
            ev.freeze(make_v2_spec(), {"alpha": roots["alpha"], "beta": roots["alpha"] / ".." / "alpha"})

    def test_v2_anchor_range_is_checked_only_against_its_own_repository(self):
        roots = self._v2_roots()
        (roots["alpha"] / "src/lib.rs").write_text("one\ntwo\nthree\n", encoding="utf-8")
        spec = make_v2_spec()
        beta_query = next(q for q in spec["queries"] if q["repository"] == "beta")
        beta_query["anchors"][0]["end_line"] = 2
        with self.assertRaisesRegex(ev.EvaluationError, "exceeds file line count"):
            ev.freeze(spec, roots)

    def test_v2_hit_range_is_checked_only_against_its_own_repository(self):
        roots = self._v2_roots()
        (roots["alpha"] / "src/lib.rs").write_text("one\ntwo\nthree\n", encoding="utf-8")
        plan = ev.freeze(make_v2_spec(), roots)
        run = make_v2_run(plan)
        beta_result = next(r for r in run["results"] if r["query_id"] == "beta-0")
        beta_result.pop("error")
        beta_result.update(status="ok", hits=[{"path": "src/lib.rs", "start_line": 1, "end_line": 2}])
        with self.assertRaisesRegex(ev.EvaluationError, "outside frozen inventory"):
            ev.score(plan, [run])

    def test_v2_plan_schema_rejects_repo_membership_tampering_after_rehash(self):
        roots = self._v2_roots()
        plan = ev.freeze(make_v2_spec(), roots)
        bad = copy.deepcopy(plan)
        bad["files"][0]["repository"] = "not-declared"
        unsigned = {k: v for k, v in bad.items() if k != "fingerprint"}
        bad["fingerprint"] = ev._digest(unsigned)
        with self.assertRaisesRegex(ev.EvaluationError, "unknown file repository"):
            ev._verify_plan(bad)
        bad = copy.deepcopy(plan)
        bad["files"][1]["repository"] = bad["files"][0]["repository"]
        bad["files"][1]["path"] = bad["files"][0]["path"]
        unsigned = {k: v for k, v in bad.items() if k != "fingerprint"}
        bad["fingerprint"] = ev._digest(unsigned)
        with self.assertRaisesRegex(ev.EvaluationError, "duplicate repository/path"):
            ev._verify_plan(bad)

    def test_v2_cli_roundtrip_and_strict_roots_json(self):
        roots = self._v2_roots()
        folder = Path(self.tmp.name)
        spec_path, roots_path = folder / "spec2.json", folder / "roots.json"
        plan_path, run_path, output = folder / "plan2.json", folder / "run2.json", folder / "report2.json"
        spec = make_v2_spec()
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        roots_path.write_text(json.dumps({k: str(v) for k, v in roots.items()}), encoding="utf-8")
        self.assertEqual(ev.main(["freeze", "--spec", str(spec_path), "--roots", str(roots_path), "--output", str(plan_path)]), 0)
        plan = ev._load(plan_path)
        self.assertNotIn("roots", plan)
        run_path.write_text(json.dumps(make_v2_run(plan, successes={q["id"] for q in spec["queries"] if q["repository"] == "alpha"})), encoding="utf-8")
        self.assertEqual(ev.main(["score", "--plan", str(plan_path), "--run", str(run_path), "--output", str(output)]), 0)
        self.assertEqual(ev._load(output)["runs"][0]["repository_macro"]["recall_at_10"], 0.5)
        roots_path.write_text('{"alpha": "x", "alpha": "y"}', encoding="utf-8")
        with patch("sys.stderr"):
            self.assertEqual(ev.main(["freeze", "--spec", str(spec_path), "--roots", str(roots_path), "--output", str(folder / "bad.json")]), 2)

    def test_v1_frozen_plan_keeps_original_shape(self):
        # Golden digests captured from the shipped v1 implementation (e5f8074).
        self.assertEqual(set(self.plan), {"format_version", "spec", "spec_sha256", "files", "fingerprint"})
        self.assertEqual(self.plan["format_version"], 1)
        self.assertTrue(all(set(f) == {"repository", "path", "sha256", "line_count"} for f in self.plan["files"]))
        self.assertEqual(ev._digest(self.plan), "001d677852db87358b1068446b311f9f1a20cc3138f1e02408d073dea35b8672")
        self.assertEqual(ev._digest(ev.score(self.plan, [make_run(self.plan)])),
                         "e1c0031b1be94f16dbaff8e8f572378558e778ac9a28164b8889cc8e634500ad")


if __name__ == "__main__":
    unittest.main()
