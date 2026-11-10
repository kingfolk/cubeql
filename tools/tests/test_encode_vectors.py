"""Offline contract checks for the RQ1 vector exporter; never loads a model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "encode_vectors.py"
spec = importlib.util.spec_from_file_location("rq1_encode_vectors", TOOL)
encoder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = encoder
spec.loader.exec_module(encoder)


class InputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import blake3
        except ImportError:
            raise unittest.SkipTest("install requirements-code-embeddings.txt for BLAKE3 input checks")
        cls.blake3 = blake3

    def record(self, identifier, text):
        return {"id": identifier, "text": text,
                "content_hash": self.blake3.blake3(text.encode("utf-8")).hexdigest()}

    def verify(self, value):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.json"
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            return encoder.verify_inputs(path)

    def test_accepts_empty_document_role_and_preserves_input_bytes(self):
        value = {"format_version": 1, "documents": [],
                 "queries": [self.record("q", "  query text  ")]}
        parsed, raw, digest = self.verify(value)
        self.assertEqual(parsed, value)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_rejects_stale_content_hash_and_duplicate_ids_per_role(self):
        record = self.record("d", "text")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.verify({"format_version": 1, "documents": [{**record, "content_hash": "0" * 64}], "queries": []})
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.verify({"format_version": 1, "documents": [record, record], "queries": []})

    def test_rejects_whitespace_query_and_both_roles_empty(self):
        with self.assertRaisesRegex(ValueError, "whitespace-only"):
            self.verify({"format_version": 1, "documents": [], "queries": [self.record("q", " \n ")]})
        with self.assertRaisesRegex(ValueError, "both be empty"):
            self.verify({"format_version": 1, "documents": [], "queries": []})


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
        except ImportError:
            raise unittest.SkipTest("install requirements-code-embeddings.txt for FP32 export checks")
        cls.np = numpy

    def test_exports_rust_embedding_config_and_empty_matrix(self):
        config = encoder.embedding_config({"name": "model", "revision": "rev", "files": {}},
                                          "jina", 1024, {"torch": "2.7.1", "transformers": "4.53.3"})
        self.assertEqual(set(config), {"model", "revision", "dimensions", "normalization", "implementation",
                                       "pooling", "max_tokens", "document_prefix", "query_prefix", "artifacts"})
        self.assertEqual(config["normalization"], "l2")
        self.assertIn("torch=2.7.1", config["implementation"])
        self.assertIn("transformers=4.53.3", config["implementation"])
        self.assertLess(config["implementation"].index("torch="), config["implementation"].index("transformers="))
        documents, queries = [], [{"id": "q", "text": "text", "content_hash": "0" * 64}]
        matrices = {"documents": self.np.empty((0, 896), dtype=self.np.float32),
                    "queries": self.np.asarray([[1.0] + [0.0] * 895], dtype=self.np.float32)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = b'{"format_version":1,"documents":[],"queries":[]}'
            (root / "inputs.json").write_bytes(inputs)
            input_sha = hashlib.sha256(inputs).hexdigest()
            encoder.export(root, input_sha, config, documents, queries, matrices)
            manifest = json.loads((root / "vectors.json").read_text())
            self.assertEqual(set(manifest), {"format_version", "dtype", "config", "inputs", "documents", "queries"})
            self.assertEqual(manifest["dtype"], "float32_le")
            self.assertEqual(manifest["documents"]["sha256"], hashlib.sha256(b"").hexdigest())
            self.assertEqual((root / "documents.f32").read_bytes(), b"")
            self.assertEqual((root / "queries.f32").stat().st_size, 896 * 4)
            self.assertEqual(manifest["inputs"], {"file": "inputs.json", "sha256": input_sha})

    def test_runtime_change_changes_embedding_identity(self):
        model = {"name": "model", "revision": "rev", "files": {}}
        first = encoder.embedding_config(model, "coderank", 1024, {"torch": "2.7.1"})
        second = encoder.embedding_config(model, "coderank", 1024, {"torch": "2.7.2"})
        self.assertNotEqual(first["implementation"], second["implementation"])

    def test_inference_device_changes_contract_without_changing_role_prompts(self):
        model = {"name": "model", "revision": "rev", "files": {}}
        cpu = encoder.embedding_config(model, "jina", 1024, {"torch": "2.7.1"}, "cpu")
        mps = encoder.embedding_config(model, "jina", 1024, {"torch": "2.7.1"}, "mps")
        self.assertNotEqual(cpu["implementation"], mps["implementation"])
        self.assertEqual(cpu["document_prefix"], mps["document_prefix"])
        self.assertEqual(cpu["query_prefix"], mps["query_prefix"])


class CliTests(unittest.TestCase):
    def test_help_runs_without_model_packages_or_model_files(self):
        result = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fetch", result.stdout)
        self.assertIn("encode", result.stdout)


if __name__ == "__main__":
    unittest.main()
