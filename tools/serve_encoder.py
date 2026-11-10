#!/usr/bin/env python3
"""Serve the pinned RQ1 encoder over a strict, bounded JSONL protocol."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

import encode_vectors as recipe

MAX_FRAME = 1024 * 1024
MAX_TEXTS = 32


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _write(value, out):
    data = _json_bytes(value)
    if len(data) > MAX_FRAME:
        raise ValueError("response exceeds 1 MiB")
    out.write(data.decode("utf-8"))
    out.flush()


def _peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def _diagnostic(event, **fields):
    print(json.dumps({"event": event, "pid": os.getpid(), "peak_rss_bytes": _peak_rss_bytes(), **fields},
                     allow_nan=False, separators=(",", ":")), file=sys.stderr, flush=True)


def _parse(line):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(line, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON number")))


def _request(value):
    if not isinstance(value, dict) or set(value) != {"protocol_version", "id", "role", "texts"}:
        raise ValueError("request must contain exactly protocol_version, id, role and texts")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != 1:
        raise ValueError("unsupported protocol_version")
    if type(value["id"]) is not int or not 1 <= value["id"] <= 2**64 - 1:
        raise ValueError("id must be a positive u64")
    if value["role"] not in ("query", "document"):
        raise ValueError("role must be query or document")
    texts = value["texts"]
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS or any(not isinstance(t, str) for t in texts):
        raise ValueError("texts must contain 1..32 strings")
    if value["role"] == "query" and (len(texts) != 1 or not texts[0].strip()):
        raise ValueError("query requests require exactly one nonblank text")
    return value["id"], value["role"], texts


def serve(config, tokenizer, embed, load_seconds, inp=None, out=None, max_rss_mb=None):
    inp, out = inp or sys.stdin, out or sys.stdout
    if max_rss_mb is not None:
        peak = _peak_rss_bytes()
        if peak is None:
            raise ValueError("--max-rss-mb is unsupported on this platform")
        current_mb = peak / (1024 * 1024)
        if current_mb > max_rss_mb:
            raise MemoryError("resident memory ceiling exceeded during model load")
    load_ms = float(load_seconds * 1000)
    if not math.isfinite(load_ms) or load_ms < 0:
        raise ValueError("invalid model load time")
    _write({"protocol_version": 1, "event": "ready", "config": config, "model_load_ms": load_ms}, out)
    _diagnostic("ready", model_load_ms=load_ms)
    while True:
        line = inp.readline(MAX_FRAME + 1)
        if not line:
            break
        if len(line.encode("utf-8")) > MAX_FRAME or not line.endswith("\n"):
            raise ValueError("request frame exceeds 1 MiB or is unterminated")
        request_id = None
        try:
            value = _parse(line)
            request_id, role, texts = _request(value)
            if max_rss_mb is not None:
                peak = _peak_rss_bytes()
                if peak is None or peak > max_rss_mb * 1024 * 1024:
                    raise MemoryError("resident memory ceiling exceeded before request")
            prefix = config["query_prefix"] if role == "query" else config["document_prefix"]
            started = time.perf_counter()
            vectors, _lengths = recipe.encode_text_batch([prefix + text for text in texts], tokenizer,
                embed, config["max_tokens"], config["dimensions"])
            elapsed = (time.perf_counter() - started) * 1000
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid encoding time")
            if max_rss_mb is not None:
                peak = _peak_rss_bytes()
                if peak is None or peak > max_rss_mb * 1024 * 1024:
                    raise MemoryError("resident memory ceiling exceeded")
            _write({"protocol_version": 1, "id": request_id, "vectors": vectors, "encoding_ms": elapsed}, out)
            _diagnostic("result", request_id=request_id, role=role, encoding_ms=elapsed)
        except Exception as error:
            if request_id is None:
                raise
            _write({"protocol_version": 1, "id": request_id, "error": str(error)[:512]}, out)
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("coderank", "jina"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-rss-mb", type=int)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_tokens > (8192 if args.model == "coderank" else 32768):
        parser.error("--max-tokens is outside the pinned recipe limits")
    if args.model == "coderank" and args.device != "cpu":
        parser.error("the pinned CodeRank recipe supports CPU inference only")
    if args.max_rss_mb is not None and args.max_rss_mb <= 0:
        parser.error("--max-rss-mb must be positive")
    spec = recipe.spec_for(args.model)
    recipe.fetch_spec(spec, args.model_dir)  # verification only; downloads require explicit --download in the offline exporter
    installed = recipe.runtime_versions(args.model)
    config = recipe.embedding_config(spec, args.model, args.max_tokens, installed, args.device)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    protocol_out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        tokenizer, embed, load_seconds = recipe.load_local_encoder(args.model, args.model_dir, args.max_tokens, args.device)
        return serve(config, tokenizer, embed, load_seconds, out=protocol_out, max_rss_mb=args.max_rss_mb)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"encoder worker: {error}", file=sys.stderr)
        raise SystemExit(2)
