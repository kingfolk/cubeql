"""Synthetic declarations exercise gates; no actual human review is asserted."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_eval as ev
import repository_experiment as ex


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        roots = {}
        for name in ("one", "two", "dev"):
            root = folder / name
            root.mkdir()
            (root / "lib.rs").write_text(name + "\nsecond\n")
            roots[name] = root
        def spec(ids, split):
            return {"format_version": 2, "name": split, "split": split,
                    "repositories": [{"id": i, "revision": i + "-rev"} for i in ids],
                    "queries": [{"id": i, "repository": i, "category": "api", "text": "find " + i,
                                 "filter": {}, "anchors": [{"id": "a", "path": "lib.rs", "start_line": 1, "end_line": 1}]} for i in ids]}
        self.plan = ev.freeze(spec(["one", "two"], "validation"), {k: roots[k] for k in ("one", "two")})
        self.devs = [ev.freeze(spec(["dev"], "development"), {"dev": roots["dev"]})]
        self.review = {"format_version": 1, "plan_fingerprint": self.plan["fingerprint"],
                       "queries": [{"query_id": i, "author": {"identity": "fixture-agent", "kind": "agent"},
                                    "reviews": [{"identity": "fixture-human", "kind": "human", "decision": "approved", "notes": "Synthetic test declaration only"}]} for i in ("one", "two")]}
        config = {"model_identity": "fixture-model", "corpus_identity": "fixture-sources", "chunking_identity": "fixture-lines", "filter_policy": "literal-prefix"}
        self.spec = {"format_version": 1, "name": "synthetic", "arms": [
            {"name": "base", "tool": {"name": "fixture", "revision": "1"}, "configuration": config, "candidate_k": 10},
            {"name": "candidate", "tool": {"name": "fixture", "revision": "2"}, "configuration": config, "candidate_k": 10}]}
        self.protocol = ex.freeze(self.plan, self.review, self.devs, self.spec)

    def runs(self):
        runs = []
        for arm in self.spec["arms"]:
            run = copy.deepcopy(arm)
            run.update(format_version=1, plan_fingerprint=self.plan["fingerprint"], results=[])
            for q in self.plan["spec"]["queries"]:
                ok = (q["id"] == "one") == (arm["name"] == "base")
                result = {"query_id": q["id"], "status": "ok" if ok else "timeout",
                          "hits": [{"path": "lib.rs", "start_line": 1, "end_line": 1}] if ok else [], "elapsed_ms": 2}
                if not ok:
                    result["error"] = "retained fixture timeout"
                run["results"].append(result)
            runs.append(run)
        return runs

    def test_hand_computed_pair_retains_failures_and_baseline_order(self):
        result = ex.score(self.protocol, list(reversed(self.runs())))
        report = result["evaluation"]
        self.assertEqual([r["name"] for r in report["runs"]], ["base", "candidate"])
        self.assertEqual(report["runs"][0]["overall"]["recall_at_10"], .5)
        self.assertEqual(report["runs"][0]["overall"]["status_counts"]["timeout"], 1)
        pair = report["paired_comparisons"][0]
        self.assertEqual(pair["recall_at_10_outcomes"], {"wins": 1, "ties": 0, "losses": 1})
        self.assertEqual(pair["per_query"]["one"]["recall_at_10"], -1)
        self.assertEqual(pair["per_query"]["two"]["recall_at_10"], 1)
        self.assertEqual(pair["repository_macro_delta"]["recall_at_10"], 0)
        self.assertEqual(result["declared_elapsed_ms"]["base"]["complete_sum_ms"], 4)

    def test_requires_exact_full_arm_set_and_conditions(self):
        for mutate in (lambda rs: rs.pop(), lambda rs: rs.append(copy.deepcopy(rs[0])),
                       lambda rs: rs[1].update(name="other"),
                       lambda rs: rs[1].update(candidate_k=20),
                       lambda rs: rs[1]["tool"].update(revision="3"),
                       lambda rs: rs[1]["configuration"].update(weight=2),
                       lambda rs: rs[1]["results"].pop()):
            runs = self.runs()
            mutate(runs)
            with self.subTest(mutate=mutate), self.assertRaises(ev.EvaluationError):
                ex.score(self.protocol, runs)

    def test_frozen_input_changes_fail_even_if_outer_receipt_rehashed(self):
        for mutate in (lambda p: p["audit"].update(ready=False),
                       lambda p: p["review"]["queries"][0].update(reviews=[]),
                       lambda p: p["plan"]["spec"]["queries"][0].update(text="changed"),
                       lambda p: p.update(format_version=True)):
            p = copy.deepcopy(self.protocol)
            mutate(p)
            p["fingerprint"] = ev._digest({k: v for k, v in p.items() if k != "fingerprint"})
            with self.subTest(mutate=mutate), self.assertRaises(ev.EvaluationError):
                ex.verify(p)

    def test_unready_review_cannot_freeze(self):
        self.review["queries"][0]["reviews"] = []
        with self.assertRaises(ev.EvaluationError):
            ex.freeze(self.plan, self.review, self.devs, self.spec)
        with self.assertRaises(ev.EvaluationError):
            ex.freeze(self.plan, self.protocol["review"], [], self.spec)

    def test_strict_experiment_schema_and_no_duplicate_conditions(self):
        for mutate in (lambda s: s.update(format_version=True), lambda s: s.update(extra=1),
                       lambda s: s.update(arms=s["arms"][:1]),
                       lambda s: s["arms"][1].update(name="base"),
                       lambda s: s["arms"][1].update(tool=s["arms"][0]["tool"]),
                       lambda s: s["arms"][0].update(candidate_k=True),
                       lambda s: s["arms"][0]["configuration"].pop("model_identity"),
                       lambda s: s["arms"][0]["configuration"].update(weight=float("nan"))):
            spec = copy.deepcopy(self.spec)
            mutate(spec)
            with self.subTest(mutate=mutate), self.assertRaises((ev.EvaluationError, ValueError)):
                ex.freeze(self.plan, self.review, self.devs, spec)

    def test_unequal_windows_null_mrr_and_missing_timing_explicit(self):
        self.spec["arms"][1]["candidate_k"] = 20
        protocol = ex.freeze(self.plan, self.review, self.devs, self.spec)
        runs = self.runs()
        runs[1]["results"][0].pop("elapsed_ms")
        report = ex.score(protocol, runs)
        self.assertIsNone(report["evaluation"]["paired_comparisons"][0]["macro_delta"]["mrr"])
        self.assertIsNone(report["declared_elapsed_ms"]["candidate"]["complete_sum_ms"])
        self.assertEqual(report["declared_elapsed_ms"]["candidate"]["observed_query_count"], 1)

    def test_inputs_not_mutated_and_receipt_repeatable(self):
        before = ev._canonical([self.plan, self.review, self.devs, self.spec])
        self.assertEqual(self.protocol, ex.freeze(self.plan, self.review, self.devs, self.spec))
        runs = self.runs()
        run_bytes = ev._canonical(runs)
        ex.score(self.protocol, runs)
        self.assertEqual(before, ev._canonical([self.plan, self.review, self.devs, self.spec]))
        self.assertEqual(run_bytes, ev._canonical(runs))

    def test_exact_json_types_and_finite_aggregate_timing(self):
        self.spec["arms"][1]["configuration"]["weight"] = 1
        protocol = ex.freeze(self.plan, self.review, self.devs, self.spec)
        runs = self.runs()
        runs[1]["configuration"]["weight"] = True
        with self.assertRaises(ev.EvaluationError):
            ex.score(protocol, runs)
        runs = self.runs()
        for result in runs[0]["results"]:
            result["elapsed_ms"] = 1e308
        with self.assertRaises(ev.EvaluationError):
            ex.score(protocol, runs)

    def test_cli_roundtrip_strict_json_and_create_new(self):
        folder = Path(self.tmp.name)
        for name, value in (("plan", self.plan), ("review", self.review), ("dev", self.devs[0]), ("spec", self.spec)):
            (folder / (name + ".json")).write_text(json.dumps(value))
        p = lambda name: str(folder / (name + ".json"))
        args = ["freeze", "--plan", p("plan"), "--review", p("review"), "--development-plan", p("dev"), "--spec", p("spec"), "--output", p("protocol")]
        self.assertEqual(ex.main(args), 0)
        self.assertEqual(ev._load(p("protocol")), self.protocol)
        for i, run in enumerate(self.runs()):
            (folder / (str(i) + ".json")).write_text(json.dumps(run))
        score = ["score", "--protocol", p("protocol"), "--run", p("1"), "--run", p("0"), "--output", p("report")]
        self.assertEqual(ex.main(score), 0)
        old = Path(p("report")).read_bytes()
        with patch("sys.stderr"):
            self.assertEqual(ex.main(score), 2)
            Path(p("spec")).write_text('{"name":1,"name":2}')
            args[-1] = p("bad")
            self.assertEqual(ex.main(args), 2)
        self.assertEqual(old, Path(p("report")).read_bytes())
        self.assertFalse(Path(p("bad")).exists())


if __name__ == "__main__":
    unittest.main()
