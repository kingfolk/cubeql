import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcp_reload_smoke as smoke


class ReloadSmokeTests(unittest.TestCase):
    def test_invalid_resource_limits_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            for value in [True, 0, -1, float('nan'), float('inf')]:
                out = Path(d)/'not-created'
                with self.subTest(value=value), self.assertRaises(ValueError):
                    smoke.run(sys.executable, sys.executable, out, timeout_s=value)
                self.assertFalse(out.exists())

    def test_existing_output_is_never_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'existing'
            out.mkdir()
            sentinel = out/'summary.json'
            sentinel.write_text('preserved')
            with self.assertRaises(FileExistsError):
                smoke.run(sys.executable, sys.executable, out)
            self.assertEqual(sentinel.read_text(),'preserved')
            self.assertEqual(list(out.iterdir()),[sentinel])

    def test_missing_binary_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'new'
            with self.assertRaises(FileNotFoundError):
                smoke.run(Path(d)/'missing',sys.executable,out)
            self.assertFalse(out.exists())

    def test_actual_failed_command_retains_evidence_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Executed process is a deliberate failure fixture, not a model or agent.
            binary = root/'failed-binary'
            binary.write_text('#!/bin/sh\nprintf "intentional fixture failure\\n" >&2\nexit 7\n')
            binary.chmod(0o700)
            out = root/'attempt'
            with self.assertRaisesRegex(ValueError,'index failed'):
                smoke.run(binary,sys.executable,out,timeout_s=5)
            self.assertEqual(json.loads((out/'index-execution.json').read_text())['returncode'],7)
            self.assertIn('intentional fixture failure',(out/'index.stderr').read_text())
            self.assertIn('index failed',json.loads((out/'failure.json').read_text())['error'])
            self.assertTrue((out/'inputs-before.json').exists())
            self.assertFalse((out/'manifest.json').exists())
            self.assertFalse((out/'summary.json').exists())


if __name__=='__main__':
    unittest.main()
