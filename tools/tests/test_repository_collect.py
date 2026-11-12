import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import repository_collect as rc
import repository_eval as ev


def plan_for(root):
    spec = {"format_version": 1, "name": "fixture", "split": "validation",
            "repositories": [{"id": "repo", "revision": "fixture"}],
            "queries": [{"id": "q1", "repository": "repo", "category": "test",
                         "text": "find anchor", "filter": {},
                         "anchors": [{"id": "a", "path": "src.py", "start_line": 1, "end_line": 2}]}]}
    return ev.freeze(spec, root)


def fake_binary(directory, *, index_fail=False, corrupt=False, slow=False, bad_range=False, mutate=None):
    path = Path(directory) / "fake cubeql"
    script = r'''#!/usr/bin/env python3
import json, os, sys, time
args = sys.argv[1:]
if args[0] == "index":
    if SLOW:
        time.sleep(3)
    if FAIL:
        print("bad index", file=sys.stderr)
        sys.exit(7)
    if MUTATE:
        open(MUTATE, "w").write("mutated\n")
    output = args[args.index("--output") + 1]
    open(output, "xb").write(b"snapshot")
    print(json.dumps({"format_version": 1, "report": {"embedding": "hash", "chunks": 1}}))
elif args[0] == "search":
    q = args[-1]
    strategy = args[args.index("--strategy") + 1]
    hit = {"path":"src.py", "start_line":(99 if BAD_RANGE else 1), "end_line":2, "score":0.8, "source":("wrong" if CORRUPT else "alpha\nbeta")}
    print(json.dumps({"format_version":2, "query":q, "strategy":strategy,
                      "hits":[hit], "timings":{"retrieval_ms":0.1}}))
'''
    script = (script.replace("FAIL", repr(index_fail)).replace("CORRUPT", repr(corrupt))
              .replace("SLOW", repr(slow)).replace("BAD_RANGE", repr(bad_range))
              .replace("MUTATE", repr(str(mutate) if mutate else None)))
    path.write_text(script)
    path.chmod(0o755)
    return path


class RepositoryCollectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        (self.root / ".git").mkdir(parents=True)
        (self.root / "src.py").write_text("alpha\nbeta\n")
        (self.root / ".git" / "ignored").write_text("ignored")
        self.plan = plan_for(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_success_has_full_coverage_and_create_new_artifacts(self):
        binary = fake_binary(self.base)
        output = self.base / "out"
        result = rc.collect(self.plan, {"repo": self.root}, binary, output,
                            strategies=("lexical", "symbol"), candidate_k=20, top_k=10)
        self.assertEqual([r["name"] for r in result["runs"]], ["cubeql-lexical", "cubeql-symbol"])
        self.assertEqual(result["report"]["runs"][0]["overall"]["coverage"], 1)
        self.assertTrue((output / "manifest.json").is_file())
        self.assertEqual(result["runs"][0]["results"][0]["hits"][0]["path"], "src.py")
        with self.assertRaises(rc.CollectionError):
            rc.collect(self.plan, {"repo": self.root}, binary, output,
                       strategies=("lexical",), candidate_k=10, top_k=10)

    def test_inventory_skips_git_and_rejects_symlink_files(self):
        self.assertEqual(set(rc.source_inventory(self.root)), {"src.py"})
        if hasattr(os, "symlink"):
            (self.root / "link").symlink_to(self.root / "src.py")
            with self.assertRaises(rc.CollectionError):
                rc.source_inventory(self.root)

    def test_inventory_lf_count_and_timeout_finite_validation(self):
        (self.root / "src.py").write_bytes(b"a\r\nb\r\nlast")
        self.assertEqual(rc.source_inventory(self.root)["src.py"]["line_count"], 3)
        for timeout in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(rc.CollectionError):
                rc.execute(["true"], self.base / "timeouts", f"t{len(list((self.base / 'timeouts').glob('*')))}", timeout_s=timeout)

    def test_output_inside_source_is_rejected_before_creation(self):
        output = self.root / "collection"
        with self.assertRaises(rc.CollectionError):
            rc.collect(self.plan, {"repo": self.root}, fake_binary(self.base), output,
                       strategies=("lexical",), candidate_k=10, top_k=10)
        self.assertFalse(output.exists())

    def test_index_failure_becomes_failure_for_every_query_and_strategy(self):
        binary = fake_binary(self.base, index_fail=True)
        result = rc.collect(self.plan, {"repo": self.root}, binary, self.base / "failed",
                            strategies=("lexical", "semantic"), candidate_k=10, top_k=10)
        for run in result["runs"]:
            self.assertEqual(run["results"][0]["status"], "failed")
            self.assertEqual(run["results"][0]["query_id"], "q1")
        self.assertEqual(result["report"]["runs"][0]["overall"]["status_counts"]["failed"], 1)

    def test_corrupt_native_source_is_failed_not_partial_credit(self):
        binary = fake_binary(self.base, corrupt=True)
        result = rc.collect(self.plan, {"repo": self.root}, binary, self.base / "corrupt",
                            strategies=("lexical",), candidate_k=10, top_k=10)
        self.assertEqual(result["runs"][0]["results"][0]["status"], "failed")
        self.assertEqual(result["runs"][0]["results"][0]["hits"], [])

    def test_invalid_native_span_is_recorded_as_failed_query(self):
        binary = fake_binary(self.base, bad_range=True)
        result = rc.collect(self.plan, {"repo": self.root}, binary, self.base / "bad-range",
                            strategies=("lexical",), candidate_k=10, top_k=10)
        self.assertEqual(result["runs"][0]["results"][0]["status"], "failed")
        self.assertIn("invalid native response", result["runs"][0]["results"][0]["error"])

    def test_source_mutation_prevents_any_scoreable_run_or_report(self):
        binary = fake_binary(self.base, mutate=self.root / "src.py")
        output = self.base / "mutated"
        with self.assertRaises(rc.CollectionError):
            rc.collect(self.plan, {"repo": self.root}, binary, output,
                       strategies=("lexical",), candidate_k=10, top_k=10)
        self.assertFalse((output / "run.lexical.json").exists())
        self.assertFalse((output / "report.json").exists())
        self.assertTrue((output / "execution.0.json").exists())

    def test_timeout_and_output_limit_retain_diagnostics(self):
        binary = fake_binary(self.base, slow=True)
        result = rc.execute([str(binary), "index"], self.base / "timeout", "slow", timeout_s=0.1)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue((self.base / "timeout" / "slow.stderr").exists())
        noisy = self.base / "noisy"
        noisy.write_text("#!/bin/sh\nprintf '0123456789'\n")
        noisy.chmod(0o755)
        result = rc.execute([str(noisy)], self.base / "limited", "noise", max_output_bytes=4)
        self.assertEqual(result["status"], "failed")
        self.assertIn("byte limit", result["error"])
        self.assertLessEqual((self.base / "limited" / "noise.stdout").stat().st_size, 4)

    def test_spawn_failure_is_a_record_with_logs(self):
        result = rc.execute([str(self.base / "missing")], self.base / "spawn", "x")
        self.assertEqual(result["status"], "failed")
        self.assertIn("invocation failed", result["error"])
        self.assertTrue((self.base / "spawn" / "x.stdout").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX process groups are required")
    def test_configured_encoder_orphan_in_separate_pgid_is_cleaned(self):
        config = self.base / "encoder.json"
        config.write_text("{}")
        child_pid_file = self.base / "child.pid"
        cleaned_file = self.base / "cleaned"
        launcher = self.base / "launcher"
        ready_file = self.base / "child.ready"
        launcher.write_text("""#!/usr/bin/env python3
import os, subprocess, sys
pidfile, ready, cleaned = sys.argv[1:4]
code = 'import os,signal,time; os.setpgid(0,0); open(%r,"w").write("ready"); signal.signal(signal.SIGTERM, lambda *_: (open(%r,"w").write("yes"), exit(0))); time.sleep(30)' % (ready, cleaned)
p = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not os.path.exists(ready):
    pass
open(pidfile, 'w').write(str(p.pid))
""" )
        launcher.chmod(0o755)

        result = rc.execute([str(launcher), str(child_pid_file), str(ready_file), str(cleaned_file),
                             "--encoder-config", str(config)], self.base / "orphan", "launch")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(cleaned_file.exists(), f"execution={result}, stderr={(self.base / 'orphan' / 'launch.stderr').read_text()}, child={child_pid_file.read_text() if child_pid_file.exists() else None}")


if __name__ == "__main__":
    unittest.main()
