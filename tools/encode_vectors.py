#!/usr/bin/env python3
"""Offline local embedding export for the RQ1 code-retrieval models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unicodedata
from datetime import datetime, timezone

# Reuse the repository's verified benchmark primitives without a machine-specific path.
_TOOL_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = _TOOL_ROOT / "benchmarks" / "external"
sys.path.insert(0, str(EXTERNAL))
from common import (HERE, BusyError, atomic_file, capture, exclusive_lock, failure,
                    read_json, seal, sha256, start_directory, versions, write_json)
from data import fetch_spec
from encoding import PREFIX as CODERANK_QUERY_PREFIX, encode_batches

TOOL_DIR = Path(__file__).resolve().parent
SPEC_DIR = TOOL_DIR / "specs"
JINA_QUERY_PREFIX = "Find the most relevant code snippet given the following query:\n"
JINA_DOCUMENT_PREFIX = "Candidate code snippet:\n"
THREADS = 4
SEED = 20261102
CODE_RANK_DIMS = 768
JINA_DIMS = 896


def strict_json_bytes(data: bytes, origin: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {origin}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"invalid JSON numeric constant {value} in {origin}")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON in {origin}: {error}") from error


def verify_inputs(path: Path):
    raw = path.read_bytes()
    value = strict_json_bytes(raw, str(path))
    if not isinstance(value, dict) or set(value) != {"format_version", "documents", "queries"} or type(value["format_version"]) is not int or value["format_version"] != 1:
        raise ValueError("inputs must have exactly format_version:1, documents and queries")
    if not isinstance(value["documents"], list) or not isinstance(value["queries"], list):
        raise ValueError("documents and queries must be arrays")
    if not value["documents"] and not value["queries"]:
        raise ValueError("documents and queries cannot both be empty")
    try:
        import blake3
    except ImportError as error:
        raise ValueError("install pinned requirements before encoding (missing blake3)") from error
    for role in ("documents", "queries"):
        seen = set()
        for number, record in enumerate(value[role], 1):
            if not isinstance(record, dict) or set(record) != {"id", "text", "content_hash"}:
                raise ValueError(f"{role}[{number}] must contain exactly id, text, content_hash")
            identifier, text, claimed = record["id"], record["text"], record["content_hash"]
            if not isinstance(identifier, str) or not identifier or any(c.isspace() or unicodedata.category(c) == "Cc" for c in identifier):
                raise ValueError(f"{role}[{number}].id must be a nonempty token")
            if identifier in seen:
                raise ValueError(f"duplicate ID {identifier!r} in {role}")
            seen.add(identifier)
            if not isinstance(text, str):
                raise ValueError(f"{role}[{number}].text must be a string")
            if role == "queries" and not text.strip():
                raise ValueError(f"{role}[{number}].text cannot be whitespace-only")
            if not isinstance(claimed, str) or len(claimed) != 64 or any(c not in "0123456789abcdef" for c in claimed):
                raise ValueError(f"{role}[{number}].content_hash must be a lowercase BLAKE3 hex digest")
            actual = blake3.blake3(text.encode("utf-8")).hexdigest()
            if actual != claimed:
                raise ValueError(f"{role}[{number}].content_hash does not match exact UTF-8 text")
    return value, raw, hashlib.sha256(raw).hexdigest()


def spec_for(model: str):
    return read_json(SPEC_DIR / ("coderank.json" if model == "coderank" else "jina-code-embeddings-0.5b.json"))


def runtime_versions(model: str):
    packages = ["numpy", "blake3", "torch", "transformers", "tokenizers", "safetensors"]
    if model == "coderank":
        packages.append("sentence-transformers")
    return versions(packages)


def implementation_identity(model: str, installed: dict[str, str], device="cpu") -> str:
    recipe = ("rq1-coderank-st-cpu-fp32-cls-v1" if model == "coderank"
              else f"rq1-jina-last-token-{device}-fp32-sdpa-v1")
    runtime = ";".join(f"{name}={installed[name]}" for name in sorted(installed))
    return f"{recipe}[{runtime};threads={THREADS};seed={SEED}]"


def embedding_config(model_spec: dict, model: str, maximum: int, installed: dict[str, str], device="cpu") -> dict:
    return {"model": model_spec["name"], "revision": model_spec["revision"],
        "dimensions": CODE_RANK_DIMS if model == "coderank" else JINA_DIMS,
        "normalization": "l2", "max_tokens": maximum,
        "query_prefix": CODERANK_QUERY_PREFIX if model == "coderank" else JINA_QUERY_PREFIX,
        "document_prefix": "" if model == "coderank" else JINA_DOCUMENT_PREFIX,
        "implementation": implementation_identity(model, installed, device),
        "pooling": "cls" if model == "coderank" else "last-token",
        "artifacts": {n: item["sha256"] for n, item in model_spec["files"].items()}}


def source_identity():
    shared = {name: sha256(EXTERNAL / name) for name in ("common.py", "encoding.py", "data.py")}
    staged = {str(path.relative_to(TOOL_DIR)): sha256(path)
              for path in sorted([*TOOL_DIR.glob("*.py"), *SPEC_DIR.glob("*.json")]) if path.is_file()}
    return {"encoder_sha256": sha256(Path(__file__)), "shared_helpers_sha256": shared,
            "staged_source_sha256": staged}


def copy_inputs(output: Path, raw: bytes):
    path = output / "inputs.json"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("existing inputs.json differs from this request")
    else:
        atomic_file(path, lambda target: target.write(raw))


def copy_staged_capture(output: Path):
    destination = output / "recipe-source" / "tools"
    for path in [*TOOL_DIR.glob("*.py"), *TOOL_DIR.glob("tests/*.py"), TOOL_DIR / "requirements-code-embeddings.txt",
                 TOOL_DIR / "requirements-jina.txt", *SPEC_DIR.glob("*.json")]:
        if not path.is_file():
            continue
        target = destination / path.relative_to(TOOL_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(path.read_bytes())


def normalize_validate(matrix, rows: int, dimensions: int):
    import numpy as np
    if matrix.dtype != np.float32 or matrix.shape != (rows, dimensions) or not np.isfinite(matrix).all():
        raise ValueError("encoder returned vectors with invalid dtype, shape or values")
    if rows:
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-3):
            raise ValueError("encoder returned vectors that are not L2 normalized")


def load_local_encoder(model_name: str, model_dir: Path, maximum: int, device: str):
    """Load the pinned local model once and return its tokenizer and batch encoder."""
    import numpy as np
    import torch
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was explicitly requested but is unavailable; no CPU fallback is used")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    path = str(model_dir.resolve())
    if model_name == "coderank":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(path, device="cpu", trust_remote_code=True,
            local_files_only=True, config_kwargs={"use_flash_attn": False})
        model.max_seq_length = maximum
        model.eval().float()
        if model.get_sentence_embedding_dimension() != CODE_RANK_DIMS or not model[1].pooling_mode_cls_token or model[1].pooling_mode_mean_tokens:
            raise ValueError("CodeRank model lacks the pinned 768-dimensional CLS configuration")
        if model[0].auto_model.config.use_flash_attn:
            raise ValueError("CodeRank model enabled flash attention")
        tokenizer = model.tokenizer
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa").to(device).float().eval()
        model.config.use_cache = False
        if int(model.config.hidden_size) != JINA_DIMS:
            raise ValueError("Jina model hidden size is not the pinned 896 dimensions")

    def embed(batch):
        if model_name == "coderank":
            with torch.inference_mode():
                values = model.encode(batch, batch_size=len(batch), show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True)
                return np.asarray(values, dtype=np.float32)
        tokenizer.padding_side = "right"
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=maximum,
            return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            mask = encoded["attention_mask"]
            indices = mask.sum(dim=1) - 1
            vectors = outputs.last_hidden_state[torch.arange(len(batch), device=outputs.last_hidden_state.device), indices]
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
            return vectors.cpu().numpy().astype(np.float32, copy=False)
    return tokenizer, embed, time.perf_counter() - started


def encode_text_batch(texts, tokenizer, embed, maximum: int, dimensions: int):
    """Shared prefix-free batch inference primitive used by export and resident modes."""
    import math
    lengths = [len(value) for value in tokenizer(texts, add_special_tokens=True,
        truncation=False, padding=False)["input_ids"]] if texts else []
    if any(length < 1 for length in lengths):
        raise ValueError("tokenizer returned an empty token sequence")
    values = embed(texts)
    rows = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise ValueError("encoder returned vectors with invalid dtype, shape or values")
    for row in rows:
        if not isinstance(row, list) or len(row) != dimensions or any(type(x) not in (float, int) or not math.isfinite(x) for x in row):
            raise ValueError("encoder returned vectors with invalid dtype, shape or values")
        norm = math.sqrt(sum(float(x) * float(x) for x in row))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("encoder returned vectors that are not L2 normalized")
    return rows, lengths


def export(output: Path, input_sha: str, config: dict, documents, queries, matrices):
    import numpy as np
    files = {}
    for role, records in (("documents", documents), ("queries", queries)):
        matrix = matrices[role]
        normalize_validate(matrix, len(records), config["dimensions"])
        raw = np.asarray(matrix, dtype="<f4").tobytes(order="C")
        path = output / f"{role}.f32"
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(f"conflicting final vector file {path.name}")
        else:
            atomic_file(path, lambda target: target.write(raw))
        files[role] = {"file": path.name, "sha256": sha256(path)}
    manifest = {"format_version": 1, "dtype": "float32_le", "config": config,
                "inputs": {"file": "inputs.json", "sha256": input_sha},
                "documents": files["documents"], "queries": files["queries"]}
    target = output / "vectors.json"
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("conflicting vector manifest")
    else:
        atomic_file(target, lambda out: out.write(payload))


def encode(args):
    inputs, input_raw, input_sha = verify_inputs(args.input)
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    limit = 8192 if args.model == "coderank" else 32768
    if args.max_tokens > limit:
        raise ValueError(f"{args.model} supports at most {limit} tokens")
    if args.model == "coderank" and args.device != "cpu":
        raise ValueError("the pinned CodeRank recipe supports CPU inference only")
    model_spec = spec_for(args.model)
    paths = fetch_spec(model_spec, args.model_dir)
    installed = runtime_versions(args.model)
    identity = source_identity()
    config = embedding_config(model_spec, args.model, args.max_tokens, installed, args.device)
    request = {"operation": "encode-vectors", "input_sha256": input_sha,
        "model_spec": model_spec, "config": config, "runtime_versions": installed,
        "threads": THREADS, "seed": SEED, "device": args.device, "dtype": "float32",
        "attention": "sdpa" if args.model == "jina" else "cpu-model-default",
        "source_identity": identity}
    start_directory(args.output, request, args.resume)
    os.environ["HF_HOME"] = str(args.output / "runtime-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        with exclusive_lock(args.output / "encoding.lock"):
            if not args.resume:
                capture(args.output)
                copy_staged_capture(args.output)
            copy_inputs(args.output, input_raw)
            attempt = next(n for n in range(1, 10000) if not (args.output / f"attempt-{n:04}.json").exists())
            write_json(args.output / f"attempt-{attempt:04}.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                "resume": args.resume, "pid": os.getpid(), "request_sha256": sha256(args.output / "request.json")})
            import numpy as np
            tokenizer, embed, model_load_seconds = load_local_encoder(args.model, args.model_dir, args.max_tokens, args.device)
            records = {role: inputs[role] for role in ("documents", "queries")}
            matrices, timings = {}, {}
            for role in ("documents", "queries"):
                rows = records[role]
                prefix = config["document_prefix"] if role == "documents" else config["query_prefix"]
                texts = [prefix + row["text"] for row in rows]
                ids = [row["id"] for row in rows]
                dimensions = config["dimensions"]
                lengths = [len(value) for value in tokenizer(texts, add_special_tokens=True,
                    truncation=False, padding=False)["input_ids"]] if rows else []
                started = time.perf_counter()
                matrices[role], details = encode_batches(texts, ids, lengths, args.max_tokens,
                    dimensions, args.output / f"{role}-batches", embed)
                timings[role] = {**details, "count": len(rows),
                    "current_attempt_wall_seconds": time.perf_counter() - started,
                    "beyond_context": sum(length > args.max_tokens for length in lengths),
                    "median_tokens": float(np.median(lengths)) if lengths else None,
                    "maximum_tokens": max(lengths) if lengths else None}
            if sha256(args.input) != input_sha or source_identity() != identity:
                raise ValueError("input or recipe sources changed while encoding; retain this attempt")
            export(args.output, input_sha, config, records["documents"], records["queries"], matrices)
            write_json(args.output / f"encoding-{attempt:04}.json", {"status": "complete",
                "model_load_seconds": model_load_seconds, "timings": timings,
                "paid_model_calls": 0})
            seal(args.output, f"{args.model}-vectors", excludes=("encoding.lock", *(
                str(p.relative_to(args.output)) for p in (args.output / "runtime-cache").rglob("*") if p.is_file())))
    except BusyError:
        raise
    except (Exception, KeyboardInterrupt) as error:
        failure(args.output, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="verify or explicitly download pinned model files")
    fetch.add_argument("--model", choices=("coderank", "jina"), required=True)
    fetch.add_argument("--model-dir", type=Path, required=True)
    fetch.add_argument("--download", action="store_true", help="download missing pinned files")
    run = commands.add_parser("encode", help="encode verified input JSON into portable FP32 vectors")
    run.add_argument("input", type=Path)
    run.add_argument("--model", choices=("coderank", "jina"), required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Jina inference device; separate from CubeQL retrieval backend")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            spec = spec_for(args.model)
            result = fetch_spec(spec, args.model_dir, args.download)
            print(json.dumps({"status": "verified", "model": spec["name"],
                "files": {name: str(path) for name, path in result.items()}}, indent=2))
        else:
            encode(args)
    except Exception as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
