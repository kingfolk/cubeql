import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness_eval as runner


class AnalysisTests(unittest.TestCase):
    def events(self, values):
        return "\n".join(json.dumps(v) for v in values)

    def test_missing_usage_is_null_not_zero(self):
        report = runner.analyze(self.events([{"type":"turn.completed"}]), "baseline")
        self.assertIsNone(report["tokens"]["input_tokens"])
        self.assertIsNone(report["tokens"]["output_tokens"])

    def test_items_are_deduplicated_by_identity_and_reads_count_utf8_once(self):
        item = {"id":"x", "type":"mcp_tool_call", "server":"cubeql", "tool":"read_code"}
        value = {"snapshot_id":"saved", "chunk_id":3, "text":"配置🦀"}
        completed = dict(item, result={"structuredContent":value, "content":[{"type":"text","text":json.dumps(value)}]})
        report = runner.analyze(self.events([{"type":"item.started","item":item},
            {"type":"item.completed","item":completed}, {"type":"turn.completed","usage":{"input_tokens":0}}]), "lexical")
        self.assertEqual(report["tool_calls"]["mcp_tool_call"], 1)
        self.assertEqual(report["read_bytes_returned"], len("配置🦀".encode()))
        self.assertEqual(report["distinct_snapshot_chunks_read"], 1)
        self.assertEqual(report["tokens"]["input_tokens"], 0)

    def test_baseline_or_foreign_mcp_is_not_a_valid_control(self):
        item = {"id":"x", "type":"mcp_tool_call", "server":"cubeql", "tool":"search_code"}
        for arm, server in [("baseline", "cubeql"), ("lexical", "other")]:
            result = runner.analyze(self.events([{"type":"item.completed","item":dict(item, server=server)},
                                                {"type":"turn.completed"}]), arm)
            self.assertIn("unexpected_mcp_tool", result["problems"])

    def test_wrong_source_strategy_and_broken_streams_are_visible(self):
        item = {"id":"x", "type":"mcp_tool_call", "server":"cubeql", "tool":"search_code",
                "result":{"structured_content":{"strategy":"hybrid","hits":[]}}}
        self.assertIn("wrong_retrieval_strategy", runner.analyze(self.events([
            {"type":"item.completed","item":item},{"type":"turn.completed"}]), "lexical")["problems"])
        self.assertIn("invalid_jsonl", runner.analyze("not json", "baseline")["problems"])
        self.assertIn("missing_or_ambiguous_completion", runner.analyze("", "baseline")["problems"])

    def test_grading_requires_all_labels_and_actual_function_line_ranges(self):
        task = {"relevant":[{"path":"a.rs","symbol":"a","start_line":10,"end_line":20},
                            {"path":"b.rs","symbol":"b","start_line":1,"end_line":2}]}
        answer = {"locations":[{"path":"a.rs","symbol":"a","start_line":12,"end_line":14}],"explanation":"not graded"}
        self.assertEqual(runner.grade(answer, task)["location_recall"], 0.5)
        self.assertFalse(runner.grade(answer, task)["location_success"])
        answer["locations"][0]["start_line"] = 1
        self.assertEqual(runner.grade(answer, task)["location_recall"], 0)

    def test_unsafe_paths_and_duplicate_labels_are_rejected(self):
        for path in ["../a", "/tmp/a", "C:/a", "a\\b", "a//b"]:
            self.assertFalse(runner.relative(path))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tasks.json"
            label = {"path":"a.rs","symbol":"a"}
            path.write_text(json.dumps({"version":1,"name":"x","tasks":[{"id":"x","question":"q","relevant":[label,label]}]}))
            with self.assertRaises(ValueError):
                runner.load_tasks(path)

    def test_execute_opt_in_is_required_before_any_process(self):
        with self.assertRaisesRegex(ValueError, "--execute"):
            runner.run(argparse.Namespace(execute=False, max_runs=1))

    def test_unknown_tools_and_unobserved_context_are_not_reported_as_known_zero(self):
        result = runner.analyze(self.events([
            {"type":"item.completed","item":{"id":"r","type":"mcp_tool_call","server":"cubeql","tool":"read_code"}},
            {"type":"item.completed","item":{"id":"u","type":"new_external_tool"}},
            {"type":"turn.completed"}]), "lexical")
        self.assertIsNone(result["read_bytes_returned"])
        self.assertIn("unrecognized_item_type:new_external_tool", result["problems"])

    def test_stdout_overflow_is_a_failed_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = runner.MAX_LOG
            try:
                runner.MAX_LOG = 64
                result = runner.process([sys.executable, "-c", "print('x'*100000)"], root, root, 5)
                self.assertTrue(result["failure"].startswith("output_limit"))
                self.assertEqual((root / "events.jsonl").stat().st_size, 64)
            finally:
                runner.MAX_LOG = original

    def test_failed_runs_remain_in_denominator_and_missing_usage_remains_null(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"version":1,"harness":"codex-exec-jsonl","model":"fixture","reasoning_effort":"medium",
                    "timeout_seconds":30,"embedding":"hash","mcp_executable":sys.executable,
                    "dataset":{"tasks":[{"id":"t"}]},"trials":[{"id":"t-0-baseline","task_id":"t","arm":"baseline","repeat":0}],
                    "commit":"fixed","source_sha256":"source","snapshot_sha256":"snapshot"}
            runner.write_json(root / "plan.json", plan)
            directory = root / "runs/t-0-baseline"
            directory.mkdir(parents=True)
            runner.write_json(directory / "result.json", {"arm":"baseline","valid":False,
                "grading":{"location_success":True},"elapsed_seconds":1,"tool_calls":{"mcp_tool_call":0},
                "tokens":{key:None for key in ("input_tokens","output_tokens","cached_input_tokens","reasoning_output_tokens")}})
            result = runner.report(root / "plan.json")
            self.assertEqual(result["arms"]["baseline"]["location_success_rate_completed"], 0)
            self.assertEqual(result["arms"]["baseline"]["invalid"], 1)
            self.assertIsNone(result["arms"]["baseline"]["tokens"]["input_tokens"]["total"])
            plan["trials"][0]["id"] = "../outside"
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                runner.load_plan(root / "plan.json")

    def test_timeout_and_log_caps_preserve_failed_run_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = runner.process([sys.executable, "-c", "import time;time.sleep(5)"], root, root, .1)
            self.assertTrue(result["failure"].startswith("timeout"))
            self.assertTrue((root / "stderr.log").exists())


@unittest.skipUnless(os.getenv("CUBEQL_EVAL_REAL_BINARIES") == "1", "build Rust binaries and opt into offline integration")
class OfflineIntegrationTests(unittest.TestCase):
    def test_pinned_three_arm_run_uses_real_cubeql_mcp_without_a_model_request(self):
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            source = "pub fn target() {}\npub fn unrelated() {}\n"
            (repo / "lib.rs").write_text(source)
            (repo / "evals").mkdir()
            (repo / "evals" / "answers.json").write_text('{"secret":"must not enter workspace"}')
            subprocess.run(["git","init","--quiet",repo], check=True)
            subprocess.run(["git","-C",repo,"add","--all"], check=True)
            subprocess.run(["git","-C",repo,"-c","user.name=Fixture","-c","user.email=fixture@invalid",
                            "-c","commit.gpgsign=false","-c","core.hooksPath=/dev/null","commit","-qm","fixture"], check=True)
            # An uncommitted source change must not enter the pinned export.
            (repo / "lib.rs").write_text("pub fn dirty_worktree() {}\n")
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps({"version":1,"name":"OFFLINE","tasks":[
                {"id":"target","question":"Find target","relevant":[{"path":"lib.rs","symbol":"target"}]}]}))
            fixture = root / "fake-codex"
            shutil.copyfile(Path(__file__).parent / "fixtures" / "fake_codex.py", fixture)
            fixture.chmod(0o755)
            output = root / "prepared"
            result = runner.prepare(argparse.Namespace(tasks=tasks, repeats=1, timeout=30, model="offline-fixture",
                model_dir=None, embedding="hash", cubeql_code=str(repository / "target/debug/cubeql-code"),
                cubeql_mcp=str(repository / "target/debug/cubeql-mcp"), repo=repo, revision="HEAD",
                output=output, reasoning_effort="medium"))
            self.assertEqual(result["model_requests_started"], 0)
            self.assertEqual((output / "source/lib.rs").read_text(), source)
            self.assertFalse((output / "source/evals").exists())
            missing = runner.report(output / "plan.json")
            self.assertFalse(missing["complete_matrix"])
            self.assertIsNone(missing["arms"]["baseline"]["tokens"]["input_tokens"]["total"])
            report = runner.run(argparse.Namespace(plan=output / "plan.json", execute=True, max_runs=3, codex=str(fixture)))
            self.assertTrue(report["complete_matrix"])
            for arm in runner.ARMS:
                self.assertEqual(report["arms"][arm]["valid"], 1, report)
                self.assertEqual(report["arms"][arm]["location_successes"], 1)
            self.assertEqual(report["arms"]["baseline"]["cubeql_calls"], 0)
            self.assertEqual(report["arms"]["lexical"]["cubeql_calls"], 2)
            self.assertEqual(report["arms"]["hybrid"]["cubeql_calls"], 2)
            self.assertEqual((repo / "lib.rs").read_text(), "pub fn dirty_worktree() {}\n")
            with self.assertRaises(FileExistsError):
                output.mkdir(exist_ok=False)
            # Changing artifacts must fail before executing another harness process.
            (output / "source/lib.rs").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.run(argparse.Namespace(plan=output / "plan.json", execute=True, max_runs=1, codex=str(fixture)))


if __name__ == "__main__":
    unittest.main()
