"""Independent exhaustive float64 cosine on saved FP32 vectors, with tie diagnostics."""
from __future__ import annotations

import csv
from pathlib import Path
import time

from common import capture, failure, provenance, read_json, seal, sha256, start_directory, write_json
from data import document_text, load_dataset


def load_vectors(dataset: Path, path: Path):
    import numpy as np
    docs, queries, _, hashes = load_dataset(dataset)
    manifest = read_json(path)
    if "dtype" in manifest:
        # RQ1 production bundles bind text/IDs directly, without embedding labels or
        # tying reusable document vectors to an experiment's relevance judgments.
        if set(manifest) != {"format_version", "dtype", "config", "inputs", "documents", "queries"} or manifest["dtype"] != "float32_le":
            raise ValueError("unsupported production vector bundle")
        entry = manifest["inputs"]
        if set(entry) != {"file", "sha256"} or not entry["file"] or Path(entry["file"]).name != entry["file"] or "\\" in entry["file"]:
            raise ValueError("inputs filename must be one path component")
        import hashlib
        import blake3
        from common import decode_json
        raw = (path.parent / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("input manifest checksum mismatch")
        inputs = decode_json(raw)
        if set(inputs) != {"format_version", "documents", "queries"} or type(inputs["format_version"]) is not int or inputs["format_version"] != 1:
            raise ValueError("invalid production embedding inputs")
        for role, expected in [("documents", [(d["_id"], document_text(d)) for d in docs]),
                               ("queries", [(q["_id"], q["text"]) for q in queries])]:
            records = inputs[role]
            if len(records) != len(expected):
                raise ValueError("production input coverage differs from dataset")
            for record, (identifier, text) in zip(records, expected):
                if set(record) != {"id", "text", "content_hash"} or record["id"] != identifier or record["text"] != text or record["content_hash"] != blake3.blake3(text.encode()).hexdigest():
                    raise ValueError("production input content, ID or order differs from dataset")
        manifest = {**manifest, "dataset_sha256": hashes,
                    "document_ids": [d["_id"] for d in docs], "query_ids": [q["_id"] for q in queries]}
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1 or manifest["dataset_sha256"] != hashes:
        raise ValueError("vector manifest does not bind the current dataset")
    if manifest["document_ids"] != [d["_id"] for d in docs] or manifest["query_ids"] != [q["_id"] for q in queries]:
        raise ValueError("vector IDs/order do not match dataset")
    dimensions = manifest["config"]["dimensions"]
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
        raise ValueError("invalid vector dimensions")
    matrices = {}
    for kind, records in [("documents", docs), ("queries", queries)]:
        file = manifest[kind]
        if Path(file["file"]).name != file["file"] or not file["file"]:
            raise ValueError("vector filename must be a single path component")
        source = path.parent / file["file"]
        # Hash and parse the same immutable byte buffer, not two independently opened files.
        raw = source.read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != file["sha256"]:
            raise ValueError("vector checksum mismatch")
        if len(raw) != len(records) * dimensions * 4:
            raise ValueError("vector shape/byte count mismatch")
        matrix = np.frombuffer(raw, dtype="<f4").reshape(len(records), dimensions)
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite vector")
        normalization = manifest["config"]["normalization"]
        if normalization not in {"none", "l2"}:
            raise ValueError("unsupported normalization contract")
        if normalization == "l2":
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
            if np.any((norms != 0) & (np.abs(norms - 1) > 1e-3)):
                raise ValueError("declared L2 vectors are not unit normalized")
        matrices[kind] = matrix
    return manifest, matrices


def exhaustive(dataset: Path, vectors: Path, output: Path, top_k: int, compare_report: Path | None = None):
    import numpy as np
    from threadpoolctl import threadpool_limits
    manifest, matrices = load_vectors(dataset, vectors)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    request = {"operation": "reference-cosine", "vectors_manifest_sha256": sha256(vectors),
        "dataset_sha256": manifest["dataset_sha256"],
        "top_k": top_k, "arithmetic": "float64 exhaustive cosine on saved FP32 inputs",
        "compare_report_sha256": sha256(compare_report) if compare_report else None,
        "recipe_sha256": provenance()["recipe_sha256"]}
    start_directory(output, request)
    try:
        doc_ids, query_ids = manifest["document_ids"], manifest["query_ids"]
        docs, queries = (matrices[kind].astype(np.float64) for kind in ["documents", "queries"])
        for matrix in [docs, queries]:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, norms, out=matrix, where=norms != 0)
        compare = read_json(compare_report) if compare_report else None
        if compare and (compare.get("status") != "complete" or [q["query_id"] for q in compare["queries"]] != query_ids):
            raise ValueError("comparison report must be complete and query-aligned")
        if compare and compare["options"]["top_k"] != top_k:
            raise ValueError("comparison reports must use the same requested top-k")
        if compare and (compare["embedding"] != manifest["config"] or compare["metric"] != "cosine" or compare["precision"] != "fp32"):
            raise ValueError("comparison must use the same FP32 embedding contract and cosine metric")
        diagnostics = {"queries": len(query_ids), "top10_order_identical": 0, "topk_order_identical": 0,
            "topk_set_identical": 0, "max_returned_score_error": 0., "mismatches": []}
        output_results = {}
        order = {did: i for i, did in enumerate(doc_ids)}
        rows = np.arange(len(docs))
        begin = time.perf_counter()
        with threadpool_limits(limits=4):
            for offset in range(0, len(queries), 32):
                values = queries[offset:offset+32] @ docs.T
                for local, scores in enumerate(values):
                    index = offset + local
                    indices = np.lexsort((rows, -scores))[:top_k]
                    ranked = [doc_ids[row] for row in indices]
                    output_results[query_ids[index]] = {doc_ids[row]: float(scores[row]) for row in indices}
                    if not compare:
                        continue
                    hits = compare["queries"][index]["hits"]
                    if any(not isinstance(hit["score"], (float, int)) or not np.isfinite(hit["score"]) for hit in hits):
                        raise ValueError("non-finite comparison score")
                    got = [hit["document_id"] for hit in hits]
                    if len(set(got)) != len(got) or len(got) != len(ranked) or not set(got) <= set(order):
                        raise ValueError("invalid comparison candidates")
                    diagnostics["top10_order_identical"] += int(got[:10] == ranked[:10])
                    diagnostics["topk_order_identical"] += int(got == ranked)
                    diagnostics["topk_set_identical"] += int(set(got) == set(ranked))
                    error = max((abs(hit["score"]-scores[order[hit["document_id"]]]) for hit in hits), default=0.)
                    diagnostics["max_returned_score_error"] = max(diagnostics["max_returned_score_error"], float(error))
                    if got != ranked:
                        diagnostics["mismatches"].append({"query_id": query_ids[index], "differences": [
                            {"rank": rank + 1, "engine": a, "reference": b,
                             "reference_score_gap": float(abs(scores[order[a]]-scores[order[b]]))}
                            for rank, (a, b) in enumerate(zip(got, ranked)) if a != b]})
        diagnostics["reference_seconds"] = time.perf_counter() - begin
        diagnostics["tolerance_is_diagnostic"] = "rank mismatches are retained; none are removed or perturbed"
        write_json(output / "results.json", output_results)
        with (output / "run.trec").open("x") as out:
            for qid, scores in output_results.items():
                for rank, (did, value) in enumerate(scores.items(), 1):
                    out.write(f"{qid} Q0 {did} {rank} {value} numpy-float64\n")
        write_json(output / "parity.json", diagnostics)
        capture(output)
        seal(output, "numpy-exact-reference")
        return diagnostics
    except Exception as error:
        failure(output, error)
        raise
