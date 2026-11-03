"""Resumable pinned CodeRankEmbed recipe; optional dependencies are imported on demand."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from common import HERE, BusyError, atomic_file, capture, exclusive_lock, failure, read_json, seal, sha256, source_hashes, start_directory, versions, write_json
from data import document_text, fetch_spec, load_dataset

PREFIX = "Represent this query for searching relevant code: "


def batches(lengths, maximum):
    order = sorted(range(len(lengths)), key=lambda i: (min(lengths[i], maximum), i))
    offset = 0
    while offset < len(order):
        length = min(lengths[order[offset]], maximum)
        size, boundary = next(((size, limit) for size, limit in [(16,128),(8,256),(4,512),(2,1024)]
                               if length <= limit), (1, maximum))
        indices = [order[offset]]
        while len(indices) < size and offset + len(indices) < len(order):
            index = order[offset + len(indices)]
            if min(lengths[index], maximum) > boundary:
                break
            indices.append(index)
        yield indices
        offset += len(indices)


def encode_batches(texts, ids, lengths, maximum, dimensions, directory: Path, embed):
    """Reusable checkpoint loop; callbacks allow offline crash/resume tests without a model."""
    import numpy as np
    if not (len(texts) == len(ids) == len(lengths)) or len(set(ids)) != len(ids):
        raise ValueError("texts, IDs and lengths must have identical unique row alignment")
    directory.mkdir(exist_ok=True)
    matrix = np.empty((len(ids), dimensions), dtype=np.float32)
    seconds, reused, encoded = 0., 0, 0
    for number, indices in enumerate(batches(lengths, maximum)):
        path = directory / f"{number:05}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if cached["indices"].tolist() != indices or cached["ids"].tolist() != [ids[i] for i in indices]:
                    raise ValueError(f"checkpoint row alignment changed: {path.name}")
                vectors = cached["vectors"]
                duration = float(cached["seconds"])
            reused += len(indices)
        else:
            start = time.perf_counter()
            vectors = np.asarray(embed([texts[i] for i in indices]), dtype=np.float32)
            duration = time.perf_counter() - start
            validate_vectors(vectors, len(indices), dimensions)
            atomic_file(path, lambda out: np.savez(out, indices=np.asarray(indices),
                ids=np.asarray([ids[i] for i in indices]), vectors=vectors, seconds=np.asarray(duration)))
            encoded += len(indices)
        validate_vectors(vectors, len(indices), dimensions)
        if not math_finite_nonnegative(duration):
            raise ValueError("invalid checkpoint inference duration")
        matrix[indices] = vectors
        seconds += duration
        if number % 25 == 0:
            print(json.dumps({"stage": directory.name, "batch": number + 1,
                "encoded_rows": encoded, "reused_rows": reused, "total": len(ids)}), flush=True)
    return matrix, {"inference_seconds_including_reused_batches": seconds,
                    "rows_encoded_this_attempt": encoded, "rows_reused": reused}


def math_finite_nonnegative(value):
    import math
    return math.isfinite(value) and value >= 0


def validate_vectors(values, rows, dimensions):
    import numpy as np
    if values.dtype != np.float32 or values.shape != (rows, dimensions) or not np.isfinite(values).all():
        raise ValueError("expected finite FP32 vectors with the declared shape")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1) > 1e-3):
        raise ValueError("CodeRankEmbed vectors must be unit normalized")


def export_vectors(output: Path, dataset_sha256, config, documents, queries, doc_vectors, query_vectors):
    """Stage-complete outputs are create-new; a crash before the manifest can resume safely."""
    import numpy as np
    for kind, matrix in [("documents", doc_vectors), ("queries", query_vectors)]:
        validate_vectors(matrix, len(documents) if kind == "documents" else len(queries), config["dimensions"])
        raw = output / f"{kind}.f32"
        expected = matrix.astype("<f4").tobytes()
        if raw.exists():
            if sha256(raw) != hashlib.sha256(expected).hexdigest():
                raise ValueError(f"conflicting final vector file: {raw.name}")
        else:
            atomic_file(raw, lambda out: out.write(expected))
    manifest = {"format_version": 1, "config": config, "dataset_sha256": dataset_sha256,
        "document_ids": [d["_id"] for d in documents], "query_ids": [q["_id"] for q in queries],
        "documents": {"file": "documents.f32", "sha256": sha256(output / "documents.f32")},
        "queries": {"file": "queries.f32", "sha256": sha256(output / "queries.f32")}}
    target = output / "vectors.json"
    if target.exists():
        if read_json(target) != manifest:
            raise ValueError("conflicting vector manifest")
    else:
        payload = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        atomic_file(target, lambda out: out.write(payload))


def encode(dataset: Path, model_dir: Path, output: Path, maximum: int, resume=False):
    if maximum not in {256, 8192}:
        raise ValueError("the frozen CodeRankEmbed recipe supports context 256 or 8192")
    docs, queries, _, hashes = load_dataset(dataset)
    model_spec = read_json(HERE / "specs/coderank.json")
    fetch_spec(model_spec, model_dir)
    import numpy as np
    installed = versions(["torch", "sentence-transformers", "transformers", "numpy", "tokenizers", "einops", "safetensors"])
    request = {"operation": "encode-coderank", "dataset_sha256": hashes, "model": model_spec,
        "maximum_tokens": maximum, "pooling": "cls", "normalization": "l2", "dtype": "float32",
        "device": "cpu", "query_prefix": PREFIX, "document_prefix": "", "threads": 4,
        "versions": installed, "recipe_sha256": source_hashes()}
    start_directory(output, request, resume)
    # Caches stay next to the explicit run directory rather than modifying account settings.
    os.environ["HF_HOME"] = str(output / "runtime-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        with exclusive_lock(output / "encoding.lock"):
            if not resume:
                capture(output)
            attempt = next(number for number in range(1, 10000) if not (output / f"attempt-{number:04}.json").exists())
            write_json(output / f"attempt-{attempt:04}.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                "resume": resume, "pid": os.getpid(), "request_sha256": sha256(output / "request.json")})
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.manual_seed(20261101)
            torch.use_deterministic_algorithms(True)
            loaded = time.perf_counter()
            model = SentenceTransformer(str(model_dir.resolve()), device="cpu", trust_remote_code=True,
                local_files_only=True, config_kwargs={"use_flash_attn": False})
            model.max_seq_length = maximum
            model.eval().float()
            if model.get_sentence_embedding_dimension() != 768 or not model[1].pooling_mode_cls_token or model[1].pooling_mode_mean_tokens:
                raise ValueError("pinned model did not load with the expected CLS configuration")
            if model[0].auto_model.config.use_flash_attn:
                raise ValueError("CPU attention configuration was not applied")
            load_seconds = time.perf_counter() - loaded
            matrices, timings = {}, {}
            for kind, records in [("documents", docs), ("queries", queries)]:
                texts = [document_text(d) for d in records] if kind == "documents" else [PREFIX + q["text"] for q in records]
                ids = [r["_id"] for r in records]
                lengths = [len(value) for value in model.tokenizer(texts, add_special_tokens=True,
                    truncation=False, padding=False)["input_ids"]]
                def embed(batch):
                    with torch.inference_mode():
                        return model.encode(batch, batch_size=len(batch), show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
                start = time.perf_counter()
                matrices[kind], info = encode_batches(texts, ids, lengths, maximum, 768, output / f"{kind}-batches", embed)
                timings[kind] = dict(info, count=len(ids), current_attempt_wall_seconds=time.perf_counter()-start,
                    beyond_context=sum(length > maximum for length in lengths),
                    median_tokens=float(np.median(lengths)), maximum_tokens=max(lengths))
            if hashes != load_dataset(dataset)[3]:
                raise ValueError("dataset changed during encoding")
            if request["recipe_sha256"] != source_hashes():
                raise ValueError("recipe source changed during encoding; retain this attempt and use a new run")
            config = {"model": model_spec["name"], "revision": model_spec["revision"], "dimensions": 768,
                "normalization": "l2", "implementation": "rq0-pinned-sentence-transformers-fp32-cpu-v1",
                "pooling": "cls", "max_tokens": maximum, "document_prefix": "", "query_prefix": PREFIX,
                "artifacts": {name: item["sha256"] for name, item in model_spec["files"].items()}}
            export_vectors(output, hashes, config, docs, queries, matrices["documents"], matrices["queries"])
            # Attempt-specific timings preserve failed/resumed history rather than replacing it.
            write_json(output / f"encoding-{attempt:04}.json", {"status": "complete", "model_load_seconds": load_seconds,
                "timings": timings, "paid_model_calls": 0})
            # Dynamic import caches are runtime details, not immutable experiment artifacts.
            # Do not bundle them; only declared model hashes identify inference code.
            seal(output, "coderank-vectors", excludes=("encoding.lock", *(
                str(p.relative_to(output)) for p in (output / "runtime-cache").rglob("*") if p.is_file()
            )))
    except BusyError:
        raise
    except (Exception, KeyboardInterrupt) as error:
        failure(output, error)
        raise
