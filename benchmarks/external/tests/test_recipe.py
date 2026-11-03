from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_file, dataset_hashes, read_json, seal, start_directory, verify_seal, write_json
from data import fetch_spec, load_dataset
from scoring import score, verify_exports

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_NATIVE = importlib.util.find_spec("pytrec_eval") is not None and bool(os.environ.get("CUBEQL_COIR_EVALUATOR"))


def fixture(root):
    (root / "qrels").mkdir(parents=True)
    docs = [{"_id": "../config", "text": "load settings"},
            {"_id": "cache", "title": "Cache", "text": "store value"},
            {"_id": "http", "text": "send request"}]
    queries = [{"_id": "q-config", "text": "read settings"},
               {"_id": "q-cache", "text": "cache value"},
               {"_id": "q-unjudged", "text": "unknown judgment"}]
    for name, rows in [("corpus", docs), ("queries", queries)]:
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq-config\t../config\t2\nq-config\tcache\t0\nq-cache\tcache\t1\n")
    return docs, queries


def rankings(root, empty=False):
    result = {"q-config": {"cache": .9, "http": .8, "../config": .7},
              "q-cache": {"cache": .9, "../config": .8, "http": .7},
              "q-unjudged": {"http": .9, "cache": .8, "../config": .7}}
    if empty:
        result["q-config"] = {}
    write_json(root / "results.json", result)
    with (root / "run.trec").open("x") as out:
        for qid, scores in result.items():
            for rank, (did, value) in enumerate(scores.items(), 1):
                out.write(f"{qid} Q0 {did} {rank} {value} fixture\n")
    return result


class DataAndArchiveTests(unittest.TestCase):
    def test_input_preserves_unjudged_queries_zero_and_negative_grades(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            with (root / "qrels/test.tsv").open("a") as out:
                out.write("q-cache\thttp\t-1\n")
            docs, queries, qrels, hashes = load_dataset(root)
            self.assertEqual(len(queries), 3)
            self.assertNotIn("q-unjudged", qrels)
            self.assertEqual(qrels["q-cache"]["http"], -1)
            self.assertEqual(qrels["q-config"]["cache"], 0)
            self.assertEqual(hashes, dataset_hashes(root))
            self.assertEqual(docs[0]["_id"], "../config")

    def test_bad_references_ids_and_blank_records_fail(self):
        for name, text in [("queries.jsonl", '{"_id":"bad id","text":"q"}\n'),
                           ("corpus.jsonl", '{"_id":"a","text":"a"}\n{"_id":"a","text":"b"}\n'),
                           ("queries.jsonl", "\n"),
                           ("queries.jsonl", '{"_id":"a","_id":"b","text":"q"}\n'),
                           ("qrels/test.tsv", "query-id\tcorpus-id\tscore\nq-config\tcache\t1_000\n"),
                           ("qrels/test.tsv", "query-id\tcorpus-id\tscore\nq-config\tcache\t2147483648\n"),
                           ("qrels/test.tsv", "query-id\tcorpus-id\tscore\nq-config\tmissing\t1\n")]:
            with self.subTest(name=name, text=text), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture(root)
                (root / name).write_text(text)
                with self.assertRaises(ValueError):
                    load_dataset(root)

    def test_download_cache_validates_before_publication_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.write_bytes(b"fixed snapshot")
            spec = {"files": {"a": {"url": source.as_uri(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}
            cache = root / "cache"
            with self.assertRaises(ValueError):
                fetch_spec(spec, cache)
            self.assertFalse(cache.exists())
            fetch_spec(spec, cache, True)
            self.assertEqual((cache / "a").read_bytes(), b"fixed snapshot")
            (cache / "a").write_bytes(b"conflict")
            with self.assertRaises(ValueError):
                fetch_spec(spec, cache, True)
            self.assertEqual((cache / "a").read_bytes(), b"conflict")
            spec["files"]["b"] = {"url": source.as_uri(), "sha256": "bad"}
            del spec["files"]["a"]
            with self.assertRaises(ValueError):
                fetch_spec(spec, cache, True)
            self.assertFalse((cache / "b").exists())

    def test_seals_detect_changes_new_files_and_nested_manifest_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            (root / "nested/manifest.json").write_text("nested evidence")
            (root / "result").write_text("original")
            seal(root, "fixture")
            self.assertEqual(verify_seal(root)["files"], 2)
            (root / "extra").write_text("new evidence")
            with self.assertRaises(ValueError):
                verify_seal(root)
            (root / "extra").unlink()
            (root / "nested/manifest.json").write_text("changed")
            with self.assertRaises(ValueError):
                verify_seal(root)

    def test_create_new_and_resume_are_bound_to_full_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            request = {"model": "a", "dataset": "one", "context": 256}
            start_directory(root, request)
            start_directory(root, request, resume=True)
            with self.assertRaises(ValueError):
                start_directory(root, {**request, "context": 8192}, resume=True)
            with self.assertRaises(ValueError):
                start_directory(root, request)
            seal(root, "completed")
            with self.assertRaises(ValueError):
                start_directory(root, request, resume=True)

    def test_atomic_file_cannot_overwrite_previous_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result"
            atomic_file(path, lambda out: out.write(b"one"))
            with self.assertRaises(FileExistsError):
                atomic_file(path, lambda out: out.write(b"two"))
            self.assertEqual(path.read_bytes(), b"one")
            def interrupted(out):
                out.write(b"partial")
                raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError):
                atomic_file(Path(folder) / "incomplete", interrupted)
            self.assertFalse((Path(folder) / "incomplete").exists())


class ExportTests(unittest.TestCase):
    def test_json_trec_interop_and_unjudged_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, output = root / "dataset", root / "rankings"
            fixture(dataset)
            output.mkdir()
            rankings(output)
            checked = verify_exports(dataset, output / "results.json", output / "run.trec")
            self.assertEqual(checked["trec_rows"], 9)
            self.assertEqual(checked["unjudged_queries"], 1)
            result = read_json(output / "results.json")
            del result["q-unjudged"]
            (output / "results.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                verify_exports(dataset, output / "results.json", output / "run.trec")

    def test_duplicate_trec_and_nonfinite_json_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "dataset")
            output = root / "rankings"
            output.mkdir()
            results = rankings(output)
            trec = output / "run.trec"
            original = trec.read_text()
            trec.write_text(original + original.splitlines()[0] + "\n")
            with self.assertRaises(ValueError):
                verify_exports(root / "dataset", output / "results.json", trec)
            trec.write_text(original)
            results["q-config"]["cache"] = float("nan")
            (output / "results.json").write_text(json.dumps(results))
            with self.assertRaises(ValueError):
                verify_exports(root / "dataset", output / "results.json", trec)


@unittest.skipUnless(HAS_NUMPY, "install benchmark requirements for vector/checkpoint tests")
class VectorTests(unittest.TestCase):
    def test_interrupted_encoding_resumes_batches_without_changing_row_order(self):
        import numpy as np
        from encoding import encode_batches
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            texts = [str(i) for i in range(20)]
            ids = [f"opaque-{i}" for i in range(20)]
            lengths = list(reversed(range(1,21)))
            def embed(batch):
                return np.asarray([[1,0] if int(text) % 2 else [0,1] for text in batch], dtype=np.float32)
            calls = []
            def interrupt(batch):
                calls.append(batch)
                if len(calls) == 2:
                    raise RuntimeError("inference interrupted")
                return embed(batch)
            with self.assertRaises(RuntimeError):
                encode_batches(texts, ids, lengths, 256, 2, root, interrupt)
            matrix, stats = encode_batches(texts, ids, lengths, 256, 2, root, embed)
            np.testing.assert_array_equal(matrix, embed(texts))
            self.assertEqual(stats["rows_reused"], 16)
            self.assertEqual(stats["rows_encoded_this_attempt"], 4)
            bad_ids = list(reversed(ids))
            with self.assertRaises(ValueError):
                encode_batches(texts, bad_ids, lengths, 256, 2, root, embed)

    def test_replay_manifest_hashes_ids_and_independent_cosine(self):
        import numpy as np
        from encoding import export_vectors
        from reference import exhaustive, load_vectors
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, encoded = root / "dataset", root / "vectors"
            docs, queries = fixture(dataset)
            encoded.mkdir()
            config = {"model":"fixture", "revision":"1", "dimensions":2, "normalization":"l2",
                "implementation":"fixture-v1", "pooling":"none", "max_tokens":None,
                "document_prefix":"", "query_prefix":"", "artifacts":{}}
            export_vectors(encoded, dataset_hashes(dataset), config, docs, queries,
                np.asarray([[1,0],[0,1],[.6,.8]],dtype=np.float32),
                np.asarray([[1,0],[0,1],[1,0]],dtype=np.float32))
            exhaustive(dataset, encoded / "vectors.json", root / "reference", 3)
            result = read_json(root / "reference/results.json")
            self.assertEqual(list(result["q-config"]), ["../config","http","cache"])
            self.assertAlmostEqual(result["q-config"]["http"], .6, places=6)
            manifest = read_json(encoded / "vectors.json")
            manifest["document_ids"].reverse()
            (encoded / "vectors.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_vectors(dataset, encoded / "vectors.json")


@unittest.skipUnless(HAS_NATIVE, "set CUBEQL_COIR_EVALUATOR to a verified local evaluator cache")
class NativeScoringTests(unittest.TestCase):
    def test_hand_computable_graded_ndcg_mrr_and_sparse_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, results = root / "dataset", root / "rankings"
            fixture(dataset)
            results.mkdir()
            rankings(results)
            before = (results / "results.json").read_bytes()
            summary = score(dataset, results / "results.json", root / "scores", Path(os.environ["CUBEQL_COIR_EVALUATOR"]), [1,3],
                results / "run.trec", candidate_window=3)
            self.assertAlmostEqual(summary["direct_pytrec_unrounded"]["ndcg_cut_3"], .75)
            self.assertAlmostEqual(summary["native_coir"]["mrr"]["MRR@3"], .66667)
            self.assertEqual(summary["unjudged_queries_excluded"], 1)
            self.assertEqual((results / "results.json").read_bytes(), before)
            self.assertEqual(verify_seal(root / "scores")["kind"], "native-coir-scores")
            sparse = root / "sparse"
            sparse.mkdir()
            rankings(sparse, empty=True)
            summary = score(dataset, sparse / "results.json", root / "sparse-scores", Path(os.environ["CUBEQL_COIR_EVALUATOR"]), [3],
                sparse / "run.trec", candidate_window=3)
            self.assertEqual(summary["judged_queries"], 2)
            self.assertAlmostEqual(summary["direct_pytrec_unrounded"]["ndcg_cut_3"], .5)

    def test_requested_cutoffs_cannot_exceed_declared_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "dataset")
            (root / "rankings").mkdir()
            rankings(root / "rankings")
            with self.assertRaises(ValueError):
                score(root / "dataset", root / "rankings/results.json", root / "scores",
                    Path(os.environ["CUBEQL_COIR_EVALUATOR"]), [3], candidate_window=2)
            self.assertFalse((root / "scores").exists())

    def test_identical_id_policy_is_consistent_for_native_mrr_and_ndcg(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, results = root / "dataset", root / "rankings"
            fixture(dataset)
            (dataset / "queries.jsonl").write_text((dataset / "queries.jsonl").read_text().replace('q-config', '../config'))
            (dataset / "qrels/test.tsv").write_text((dataset / "qrels/test.tsv").read_text().replace('q-config', '../config'))
            results.mkdir()
            original = rankings(results)
            original['../config'] = original.pop('q-config')
            (results / "results.json").write_text(json.dumps(original))
            cache = Path(os.environ["CUBEQL_COIR_EVALUATOR"])
            included = score(dataset, results / "results.json", root / "included", cache, [3],
                             ignore_identical_ids=False, candidate_window=3)
            excluded = score(dataset, results / "results.json", root / "excluded", cache, [3],
                             ignore_identical_ids=True, candidate_window=3)
            self.assertAlmostEqual(included["native_coir"]["ndcg"]["NDCG@3"], .75)
            self.assertAlmostEqual(excluded["native_coir"]["ndcg"]["NDCG@3"], .5)
            self.assertAlmostEqual(excluded["native_coir"]["mrr"]["MRR@3"], .5)


if __name__ == "__main__":
    unittest.main()
