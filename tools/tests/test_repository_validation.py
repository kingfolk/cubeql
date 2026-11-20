import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
ev = importlib.import_module("repository_eval")
rv = importlib.import_module("repository_validation")


def spec(split, repos, queries, version=2):
    return {"format_version": version, "name": f"{split}-fixture", "split": split,
            "repositories": [{"id": repo, "revision": f"rev-{repo}"} for repo in repos],
            "queries": queries}


def query(query_id, repo, text, path, start=1, end=1):
    return {"id": query_id, "repository": repo, "category": "api", "text": text,
            "filter": {}, "anchors": [{"id": "anchor", "path": path,
                                         "start_line": start, "end_line": end}]}


def review_for(plan, reviews_by_id=None, author_kind="agent"):
    reviews_by_id = reviews_by_id or {}
    items = []
    for q in plan["spec"]["queries"]:
        items.append({"query_id": q["id"],
                      "author": {"identity": f"author-{q['id']}", "kind": author_kind},
                      "reviews": copy.deepcopy(reviews_by_id.get(q["id"], []))})
    return {"format_version": 1, "plan_fingerprint": plan["fingerprint"], "queries": items}


class RepositoryValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "v1").mkdir()
        (root / "v2").mkdir()
        (root / "dev").mkdir()
        (root / "v1/renamed.rs").write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
        (root / "v2/other.rs").write_text("v2 only\n", encoding="utf-8")
        # Identical complete bytes at a different path/repository alias.
        (root / "dev/copy.rs").write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
        self.validation = ev.freeze(spec("validation", ["alias-a", "alias-b"], [
            query("q-human", "alias-a", "  ＦＩＮＤ   Stra\u00dfe ", "renamed.rs", 2, 3),
            query("q-agent", "alias-a", "brand new search", "renamed.rs", 4, 4),
            query("q-change", "alias-b", "unique third question", "other.rs", 1, 1),
        ]), {"alias-a": root / "v1", "alias-b": root / "v2"})
        self.development = ev.freeze(spec("development", ["dev-alias"], [
            query("d-anchor", "dev-alias", "nothing alike", "copy.rs", 1, 2),
            query("d-text", "dev-alias", "find strasse", "copy.rs", 3, 3),
            query("d-disjoint", "dev-alias", "different wording", "copy.rs", 4, 4),
        ], version=1), root / "dev")

    def test_hand_computable_text_anchor_overlap_and_readiness(self):
        reviews = {
            "q-human": [{"identity": "reviewer-human", "kind": "human", "decision": "approved", "notes": "checked"},
                        {"identity": "reviewer-agent-veto", "kind": "agent", "decision": "changes_requested", "notes": "request revision"}],
            "q-agent": [{"identity": "reviewer-agent", "kind": "agent", "decision": "approved", "notes": "machine review"},
                        {"identity": "second-agent", "kind": "agent", "decision": "approved", "notes": "second pass"}],
            "q-change": [{"identity": "reviewer-change", "kind": "human", "decision": "changes_requested", "notes": "revise"}],
        }
        result = rv.audit(self.validation, review_for(self.validation, reviews), [self.development])
        self.assertEqual(result["queries"], [
            {"query_id": "q-human", "status": "changes_requested"},
            {"query_id": "q-agent", "status": "pending"},
            {"query_id": "q-change", "status": "changes_requested"},
        ])
        self.assertEqual(result["overlaps"], [
            {"query_id": "q-human", "development_plan_fingerprint": self.development["fingerprint"],
             "development_query_id": "d-anchor", "reasons": ["anchor"]},
            {"query_id": "q-human", "development_plan_fingerprint": self.development["fingerprint"],
             "development_query_id": "d-text", "reasons": ["text", "anchor"]},
            {"query_id": "q-agent", "development_plan_fingerprint": self.development["fingerprint"],
             "development_query_id": "d-disjoint", "reasons": ["anchor"]},
        ])
        self.assertFalse(result["ready"])
        self.assertIn("validation query q-human is changes_requested", result["blockers"])
        self.assertIn("validation query q-agent is pending", result["blockers"])
        self.assertIn("validation query q-change is changes_requested", result["blockers"])

    def test_pending_empty_reviews_and_agent_approval_do_not_count_as_human(self):
        review = review_for(self.validation, {
            "q-agent": [{"identity": "agent-reviewer", "kind": "agent", "decision": "approved", "notes": "looks fine"}],
        })
        result = rv.audit(self.validation, review, [])
        self.assertEqual([q["status"] for q in result["queries"]], ["pending"] * 3)
        self.assertFalse(result["ready"])
        self.assertIn("at least one development plan is required", result["blockers"])

    def test_same_identity_as_author_and_duplicate_reviewer_and_kind_consistency_rejected(self):
        mutations = [
            lambda r: r["queries"][0]["reviews"].append({"identity": "author-q-human", "kind": "agent", "decision": "approved", "notes": "self"}),
            lambda r: r["queries"][0]["reviews"].extend([
                {"identity": "reviewer", "kind": "human", "decision": "approved", "notes": "one"},
                {"identity": "reviewer", "kind": "human", "decision": "approved", "notes": "two"}]),
            lambda r: (r["queries"][0]["reviews"].append({"identity": "global-reviewer", "kind": "human", "decision": "approved", "notes": "first kind"}),
                      r["queries"][1]["reviews"].append({"identity": "global-reviewer", "kind": "agent", "decision": "approved", "notes": "kind changed"})),
        ]
        for mutate in mutations:
            review = review_for(self.validation)
            mutate(review)
            with self.subTest(mutate=mutate), self.assertRaises(ev.EvaluationError):
                rv.audit(self.validation, review, [self.development])

    def test_strict_schema_coverage_decisions_and_plan_provenance(self):
        mutations = [
            lambda r: r.update(extra=True),
            lambda r: r.update(format_version=True),
            lambda r: r.update(plan_fingerprint="0" * 64),
            lambda r: r["queries"].pop(),
            lambda r: r["queries"].append(copy.deepcopy(r["queries"][0])),
            lambda r: r["queries"][0].update(query_id="unknown"),
            lambda r: r["queries"][0]["author"].update(kind="model"),
            lambda r: r["queries"][0]["reviews"].append({"identity": "x", "kind": "agent", "decision": "maybe", "notes": "x"}),
            lambda r: r["queries"][0]["reviews"].append({"identity": "x", "kind": "agent", "decision": "approved", "notes": "  "}),
        ]
        for mutate in mutations:
            review = review_for(self.validation)
            mutate(review)
            with self.subTest(mutate=mutate), self.assertRaises(ev.EvaluationError):
                rv.audit(self.validation, review, [self.development])
        tampered = copy.deepcopy(self.validation)
        tampered["spec"]["queries"][0]["text"] = "changed after freeze"
        with self.assertRaises(ev.EvaluationError):
            rv.audit(tampered, review_for(self.validation), [self.development])
        wrong_split = copy.deepcopy(self.development)
        with self.assertRaisesRegex(ev.EvaluationError, "validation"):
            rv.audit(wrong_split, review_for(wrong_split), [self.development])
        with self.assertRaisesRegex(ev.EvaluationError, "development"):
            rv.audit(self.validation, review_for(self.validation), [self.validation])
        with self.assertRaisesRegex(ev.EvaluationError, "duplicate development"):
            rv.audit(self.validation, review_for(self.validation), [self.development, self.development])
        with self.assertRaises(ev.EvaluationError):
            rv.audit(self.validation, review_for(self.validation), {})

    def test_anchor_comparison_uses_whole_file_hash_and_inclusive_ranges(self):
        # Same full file hash and overlapping inclusive range counts despite aliases/path changes.
        self.assertEqual(rv._overlap_reasons(
            self.validation["spec"]["queries"][0], {"renamed.rs": self.validation["files"][0]["sha256"]},
            self.development["spec"]["queries"][0], {"copy.rs": self.development["files"][0]["sha256"]}), ["anchor"])
        # Same bytes but disjoint line intervals do not overlap.
        vq = copy.deepcopy(self.validation["spec"]["queries"][0])
        dq = copy.deepcopy(self.development["spec"]["queries"][0])
        vq["anchors"][0].update(start_line=3, end_line=3)
        dq["anchors"][0].update(start_line=1, end_line=2)
        hashes = {"renamed.rs": "a" * 64}
        other_hashes = {"copy.rs": "a" * 64}
        self.assertEqual(rv._overlap_reasons(vq, hashes, dq, other_hashes), [])
        # A shifted file with different contents does not qualify by path/range alone.
        self.assertEqual(rv._overlap_reasons(vq, hashes, dq, {"copy.rs": "b" * 64}), [])
        # Shared line numbers alone are irrelevant when whole-file digests differ.
        vq["anchors"][0].update(start_line=1, end_line=2)
        self.assertEqual(rv._overlap_reasons(vq, hashes, dq, {"copy.rs": "b" * 64}), [])

    def test_readiness_requires_multiple_validation_repositories(self):
        root = Path(self.tmp.name)
        single = ev.freeze(spec("validation", ["only"], [
            query("q", "only", "clean", "renamed.rs")], version=1), root / "v1")
        reviews = review_for(single, {"q": [{"identity": "human-review", "kind": "human", "decision": "approved", "notes": "reviewed"}]})
        # Development text/anchor are deliberately unrelated.
        (root / "dev/clean.rs").write_text("unrelated file content\n", encoding="utf-8")
        dev = ev.freeze(spec("development", ["dev"], [query("d", "dev", "other", "clean.rs")], version=1), root / "dev")
        result = rv.audit(single, reviews, [dev])
        self.assertFalse(result["ready"])
        self.assertEqual(result["blockers"], ["validation plan must include at least two repositories"])

    def test_clean_declared_human_review_is_ready_and_missing_dev_blocks_it(self):
        # Fictional declarations only; no real reviewer participated in this fixture.
        root = Path(self.tmp.name)
        (root / "dev/clean.rs").write_text("unrelated development content\n", encoding="utf-8")
        dev = ev.freeze(spec("development", ["dev"], [
            query("clean-dev", "dev", "unrelated development question", "clean.rs")], version=1), root / "dev")
        review = review_for(self.validation, {
            q["id"]: [{"identity": "fictional-human", "kind": "human", "decision": "approved", "notes": "Synthetic declaration"}]
            for q in self.validation["spec"]["queries"]
        })
        result = rv.audit(self.validation, review, [dev])
        self.assertTrue(result["ready"])
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["overlaps"], [])
        self.assertEqual([q["status"] for q in result["queries"]], ["approved"] * 3)
        unsigned = {k: v for k, v in result.items() if k != "fingerprint"}
        self.assertEqual(result["fingerprint"], ev._digest(unsigned))
        self.assertEqual(rv.audit(self.validation, review, [])["blockers"], ["at least one development plan is required"])

    def test_output_is_deterministic_and_cli_will_not_overwrite(self):
        review = review_for(self.validation, {
            qid: [{"identity": f"human-{qid}", "kind": "human", "decision": "approved", "notes": "checked"}]
            for qid in ("q-human", "q-agent", "q-change")
        })
        first = rv.audit(self.validation, review, [self.development])
        self.assertEqual(first, rv.audit(copy.deepcopy(self.validation), copy.deepcopy(review), [copy.deepcopy(self.development)]))
        folder = Path(self.tmp.name)
        plan_path, review_path = folder / "plan.json", folder / "review.json"
        dev_path, output_path = folder / "dev.json", folder / "audit.json"
        for path, value in ((plan_path, self.validation), (review_path, review), (dev_path, self.development)):
            path.write_text(json.dumps(value), encoding="utf-8")
        args = ["--plan", str(plan_path), "--review", str(review_path), "--development-plan", str(dev_path), "--output", str(output_path)]
        self.assertEqual(rv.main(args), 0)
        original = output_path.read_bytes()
        with patch("sys.stderr"):
            self.assertEqual(rv.main(args), 2)
        self.assertEqual(output_path.read_bytes(), original)

    def test_cli_rejects_duplicate_json_keys(self):
        folder = Path(self.tmp.name)
        plan_path, review_path, output_path = folder / "plan.json", folder / "review.json", folder / "out.json"
        plan_path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        review_path.write_text("{}", encoding="utf-8")
        with patch("sys.stderr"):
            self.assertEqual(rv.main(["--plan", str(plan_path), "--review", str(review_path), "--output", str(output_path)]), 2)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
