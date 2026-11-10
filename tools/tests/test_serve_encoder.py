from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("rq1_serve_encoder", TOOLS / "serve_encoder.py")
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)


class ServeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = {"dimensions": 2, "max_tokens": 8, "query_prefix": "Q:", "document_prefix": "D:"}
        self.calls = []
        def tokenizer(texts, **_kwargs):
            return {"input_ids": [[1, 2] for _ in texts]}
        def embed(texts):
            self.calls.append(list(texts))
            return [[1., 0.] for _ in texts]
        self.tokenizer, self.embed = tokenizer, embed

    def invoke(self, lines):
        output = io.StringIO()
        status = worker.serve(self.config, self.tokenizer, self.embed, .125,
                              io.StringIO(lines), output)
        return status, [json.loads(line) for line in output.getvalue().splitlines()]

    def request(self, **overrides):
        value = {"protocol_version": 1, "id": 1, "role": "query", "texts": ["find it"]}
        value.update(overrides)
        return json.dumps(value) + "\n"

    def test_ready_then_query_result_has_prefix_and_flushable_protocol_records(self):
        status, records = self.invoke(self.request())
        self.assertEqual(status, 0)
        self.assertEqual(records[0], {"protocol_version": 1, "event": "ready", "config": self.config, "model_load_ms": 125.})
        self.assertEqual(records[1]["id"], 1)
        self.assertEqual(records[1]["vectors"], [[1., 0.]])
        self.assertEqual(self.calls, [["Q:find it"]])

    def test_document_prefix_added_once_and_worker_reuses_loaded_encoder(self):
        status, _ = self.invoke(self.request(id=5, role="document", texts=["code"]) +
                                self.request(id=6, role="document", texts=["more"]))
        self.assertEqual(status, 0)
        self.assertEqual(self.calls, [["D:code"], ["D:more"]])

    def test_invalid_schema_role_count_and_blank_query_fail(self):
        for line in ("{\"protocol_version\":1,\"id\":1,\"role\":\"wat\",\"texts\":[\"x\"]}\n",
                     self.request(texts=[]), self.request(texts=["x", "y"]), self.request(texts=["  "]),
                     self.request(extra=True), "{bad}\n"):
            with self.subTest(line=line[:40]), self.assertRaises(ValueError):
                self.invoke(line)

    def test_oversize_frame_and_nonfinite_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            self.invoke(" " * (worker.MAX_FRAME + 1) + "\n")
        self.embed = lambda _texts: [[float("nan"), 0.]]
        status, records = self.invoke(self.request())
        self.assertEqual(status, 1)
        self.assertIn("error", records[-1])

    def test_invalid_dimensions_and_nonfinite_load_metric_rejected(self):
        output = io.StringIO()
        status = worker.serve({**self.config, "dimensions": 3}, self.tokenizer, self.embed, 0,
                              io.StringIO(self.request()), output)
        self.assertEqual(status, 1)
        with self.assertRaises(ValueError):
            worker._json_bytes({"metric": float("nan")})

    def test_oversize_unterminated_frame_reads_at_most_limit_plus_one(self):
        class BoundedReader:
            requested = None
            def readline(self, size):
                self.requested = size
                return "x" * size
        source, output = BoundedReader(), io.StringIO()
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            worker.serve(self.config, self.tokenizer, self.embed, 0, source, output)
        self.assertEqual(source.requested, worker.MAX_FRAME + 1)
        self.assertEqual(self.calls, [])

    def test_rss_ceiling_checked_after_inference_before_vectors_are_written(self):
        source = io.StringIO(self.request())
        output = io.StringIO()
        with mock.patch.object(worker, "_peak_rss_bytes", side_effect=[1_000_000, 1_000_000, 20_000_000]):
            status = worker.serve(self.config, self.tokenizer, self.embed, 0, source, output, max_rss_mb=10)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(status, 1)
        self.assertEqual(records[-1]["id"], 1)
        self.assertIn("ceiling", records[-1]["error"])
        self.assertNotIn("vectors", records[-1])

    def test_main_keeps_protocol_on_stdout_while_model_stdout_is_redirected(self):
        spec_value = {"name": "tiny", "revision": "rev", "files": {}}
        def fake_loader(*_args):
            print("model loading chatter")
            def embed(texts):
                self.calls.extend([[text] for text in texts])
                return [[1.] + [0.] * 895 for _ in texts]
            return self.tokenizer, embed, 0.01
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["serve_encoder.py", "--model", "jina", "--model-dir", "/local/model", "--max-tokens", "8"]), \
             mock.patch.object(sys, "stdin", io.StringIO(self.request())), \
             mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr), \
             mock.patch.object(worker.recipe, "spec_for", return_value=spec_value), \
             mock.patch.object(worker.recipe, "fetch_spec", return_value={}), \
             mock.patch.object(worker.recipe, "runtime_versions", return_value={"torch": "x"}), \
             mock.patch.object(worker.recipe, "load_local_encoder", side_effect=fake_loader):
            status = worker.main()
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual([record.get("event") for record in records], ["ready", None])
        self.assertNotIn("model loading chatter", stdout.getvalue())
        self.assertIn("model loading chatter", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
