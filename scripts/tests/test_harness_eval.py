import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness_eval as runner
import mcp_audit
import io


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

    def test_rust_aliases_preserve_crate_module_impl_path_and_range_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Cargo.toml").write_text('[package]\nname="cubeql-code"\nversion="0.1.0"\n')
            (root / "src").mkdir()
            (root / "src/retrieval.rs").write_text("pub fn target() {}\n")
            label = {"path":"src/retrieval.rs","symbol":"Collector::target","start_line":10,"end_line":20}
            label["symbol_aliases"] = runner.rust_symbol_aliases(root, label)
            task = {"relevant":[label]}
            for name in ["Collector::target", "retrieval::Collector::target",
                         "crate::retrieval::Collector::target", "cubeql_code::retrieval::Collector::target"]:
                answer = {"locations":[dict(path=label["path"], symbol=name, start_line=12, end_line=14)],"explanation":""}
                self.assertTrue(runner.grade(answer, task)["location_success"], name)
            for name in ["target", "Other::target", "other::Collector::target",
                         "wrong_crate::retrieval::Collector::target", "cubeql_code::wrong::Collector::target"]:
                answer["locations"][0]["symbol"] = name
                self.assertFalse(runner.grade(answer, task)["location_success"], name)
            answer["locations"][0] = dict(path="src/other.rs", symbol="Collector::target", start_line=12, end_line=14)
            self.assertFalse(runner.grade(answer, task)["location_success"])
            answer["locations"][0] = dict(path=label["path"], symbol="Collector::target", start_line=1, end_line=14)
            self.assertFalse(runner.grade(answer, task)["location_success"])

    def test_aliases_keep_inline_scopes_and_fail_closed_for_custom_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "Cargo.toml"
            manifest.write_text('[package]\nname="fixture"\nversion="0.1.0"\n')
            (root / "src").mkdir()
            source = root / "src/lib.rs"
            source.write_text("mod a {}\nmod b {}\n")
            label = {"path":"src/lib.rs","symbol":"a::target","start_line":1,"end_line":1}
            aliases = runner.rust_symbol_aliases(root, label)
            self.assertIn("fixture::a::target", aliases)
            self.assertNotIn("fixture::b::target", aliases)
            self.assertNotIn("target", aliases)
            source.write_text('#[path="elsewhere.rs"] mod custom;')
            self.assertEqual(runner.rust_symbol_aliases(root, label), ["a::target"])
            source.write_text("include!(\"elsewhere.rs\");")
            self.assertEqual(runner.rust_symbol_aliases(root, label), ["a::target"])
            source.write_text("pub fn target() {}")
            manifest.write_text('[package]\nname="fixture"\n[lib]\npath="custom.rs"\n')
            self.assertEqual(runner.rust_symbol_aliases(root, label), ["a::target"])

    def test_cargo_library_names_and_nested_module_filenames_are_not_confused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Cargo.toml").write_text('[package]\nname="package-name"\n[lib]\nname="library_name"\n')
            (root / "src/nested").mkdir(parents=True)
            for filename, qualified in [("lib.rs","library_name::target"),
                                        ("main.rs","package_name::target"),
                                        ("nested/mod.rs","library_name::nested::target"),
                                        ("nested/lib.rs","library_name::nested::lib::target")]:
                (root / "src" / filename).write_text("pub fn target() {}")
                label = {"path":"src/"+filename,"symbol":"target"}
                self.assertIn(qualified, runner.rust_symbol_aliases(root, label))

    def test_failed_search_attempt_is_not_successful_retrieval_adoption(self):
        item = {"id":"x","type":"mcp_tool_call","server":"cubeql","tool":"search_code",
                "result":{"isError":True,"structuredContent":{"strategy":"lexical","hits":[]}}}
        values = [{"type":"item.completed","item":item},{"type":"turn.completed"}]
        result = runner.analyze(self.events(values), "lexical")
        self.assertEqual(result["tool_calls"]["search_code"], 1)
        self.assertEqual(result["successful_search_calls"], 0)

    def test_same_client_discovery_requires_initialize_and_complete_expected_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.jsonl"
            self.assertEqual(runner.discovery(path, "lexical")["status"], "unobserved")
            output = io.BytesIO()
            audit = mcp_audit.Discovery(output)
            def observe(value, incoming):
                audit.observe(json.dumps(value).encode(), incoming)
            observe({"id":1,"method":"initialize"}, True)
            observe({"id":1,"result":{}}, False)
            observe({"id":2,"method":"tools/list"}, True)
            tools = [{"name":name,"inputSchema":{"type":"object"},"annotations":{"readOnlyHint":True}}
                     for name in runner.MCP_TOOLS]
            observe({"id":2,"result":{"tools":tools}}, False)
            path.write_bytes(output.getvalue())
            self.assertEqual(runner.discovery(path, "lexical")["status"], "client_discovered")
            record = json.loads(path.read_text())
            for key, value in [("initialized",False), ("next_cursor","more"), ("tools",tools[:1])]:
                path.write_text(json.dumps(dict(record, **{key:value})))
                with self.assertRaises(ValueError):
                    runner.discovery(path, "hybrid")
            path.write_text("invalid")
            with self.assertRaises(ValueError):
                runner.discovery(path, "lexical")

    def test_legacy_plans_cannot_resume_with_changed_grading(self):
        with mock.patch.object(runner, "load_plan", return_value={"version":1}):
            with self.assertRaisesRegex(ValueError, "legacy plans"):
                runner.run(argparse.Namespace(execute=True, max_runs=1, plan=Path("plan.json")))

    def test_regrade_is_offline_versioned_and_never_overwrites_original_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            (source / "src").mkdir(parents=True)
            (source / "Cargo.toml").write_text('[package]\nname="fixture"\nversion="0.1.0"\n')
            (source / "src/lib.rs").write_text("pub fn target() {}\n")
            (root / "snapshot.arrow").write_bytes(b"fixture snapshot")
            plan = {"version":1,"harness":"codex-exec-jsonl","model":"fixture","reasoning_effort":"medium",
                    "timeout_seconds":30,"embedding":"hash","mcp_executable":sys.executable,
                    "dataset":{"tasks":[{"id":"t","relevant":[{"path":"src/lib.rs","symbol":"target",
                                                                              "start_line":1,"end_line":1}]}]},
                    "trials":[{"id":"t-0-baseline","task_id":"t","arm":"baseline","repeat":0}],
                    "commit":"fixed","source_sha256":runner.fingerprint(source),
                    "snapshot_sha256":runner.digest(root / "snapshot.arrow")}
            runner.write_json(root / "plan.json", plan)
            directory = root / "runs/t-0-baseline"
            directory.mkdir(parents=True)
            runner.write_json(directory / "answer.json", {"locations":[{"path":"src/lib.rs",
                "symbol":"fixture::target","start_line":1,"end_line":1}],"explanation":""})
            runner.write_json(directory / "result.json", {"arm":"baseline","valid":True,
                "grading":{"location_success":False,"location_recall":0},"elapsed_seconds":1,
                "tool_calls":{"mcp_tool_call":0},
                "tokens":{key:None for key in ("input_tokens","output_tokens","cached_input_tokens","reasoning_output_tokens")}})
            originals = {p:runner.digest(p) for p in root.rglob("*") if p.is_file()}
            args = argparse.Namespace(plan=root / "plan.json", output=root / "regraded.json")
            with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must be offline")):
                report = runner.regrade(args)
            self.assertEqual(report["model_requests_started"], 0)
            self.assertEqual(report["arms"]["baseline"]["location_successes"], 1)
            self.assertTrue(runner.read_json(args.output)["post_hoc_regrade"])
            self.assertEqual(runner.report(args.plan)["arms"]["baseline"]["location_successes"], 0)
            self.assertEqual(originals, {p:runner.digest(p) for p in originals})
            with self.assertRaises(FileExistsError):
                runner.regrade(args)

    def test_unrequested_tool_catalog_is_not_discovery_evidence(self):
        output = io.BytesIO()
        audit = mcp_audit.Discovery(output)
        audit.observe(json.dumps({"id":99,"result":{"tools":[]}}).encode(), False)
        self.assertEqual(output.getvalue(), b"")

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
            for arm in ("lexical", "hybrid"):
                self.assertEqual(report["arms"][arm]["client_discovery_observed_runs"], 1)
                self.assertEqual(report["arms"][arm]["search_adoption_runs"], 1)
                self.assertTrue((output / f"runs/target-0-{arm}/mcp-discovery.jsonl").is_file())
            self.assertEqual(report["comparison_warnings"], [])
            # A callable server without tools/list by the actual CLI is NOT proven visible.
            # The runner must stop after the first augmented trial rather than pay for more.
            no_discovery = root / "no-discovery"
            shutil.copytree(output, no_discovery, ignore=shutil.ignore_patterns("runs", "preflight-*"))
            with mock.patch.dict(os.environ, {"CUBEQL_FIXTURE_SKIP_DISCOVERY":"1"}):
                missing = runner.run(argparse.Namespace(plan=no_discovery / "plan.json", execute=True,
                                                       max_runs=3, codex=str(fixture)))
            self.assertEqual(missing["arms"]["lexical"]["invalid"], 1)
            self.assertEqual(missing["arms"]["hybrid"]["missing"], 1)
            self.assertIn("mcp_discovery_unobserved", missing["runs"][1]["problems"])
            # An unavailable MCP server is caught in the offline preflight, before exec.
            failure = root / "preflight-failure"
            shutil.copytree(output, failure, ignore=shutil.ignore_patterns("runs", "preflight-*"))
            broken = runner.read_json(failure / "plan.json")
            broken["mcp_executable"] = sys.executable  # Python cannot speak MCP for an Arrow file.
            broken["mcp_sha256"] = runner.digest(Path(sys.executable))
            (failure / "plan.json").write_text(json.dumps(broken))
            real_process = runner.process
            def offline_only(command, *args):
                self.assertNotIn("exec", command)
                return real_process(command, *args)
            with mock.patch.object(runner, "process", side_effect=offline_only):
                with self.assertRaisesRegex(ValueError, "preflight failed"):
                    runner.run(argparse.Namespace(plan=failure / "plan.json", execute=True,
                                                 max_runs=3, codex=str(fixture)))
            self.assertEqual(list((failure / "runs").iterdir()), [])
            self.assertTrue(list(failure.glob("preflight-*/lexical/stderr.log")))
            self.assertEqual((repo / "lib.rs").read_text(), "pub fn dirty_worktree() {}\n")
            with self.assertRaises(FileExistsError):
                output.mkdir(exist_ok=False)
            # Changing artifacts must fail before executing another harness process.
            (output / "source/lib.rs").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.run(argparse.Namespace(plan=output / "plan.json", execute=True, max_runs=1, codex=str(fixture)))


if __name__ == "__main__":
    unittest.main()
