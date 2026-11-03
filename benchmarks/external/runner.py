#!/usr/bin/env python3
"""Portable, explicit stages for CubeQL external retrieval experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from common import HERE, REPO, capture, dataset_hashes, failure, provenance, read_json, run_process, seal, sha256, start_directory, verify_seal, write_json
from data import fetch_spec, load_dataset


def run_cli(binary: Path, dataset: Path, output: Path, kind: str, top_k: int,
            vectors: Path | None = None, model_dir: Path | None = None):
    load_dataset(dataset)
    binary = binary.resolve(strict=True)
    request = {"operation": kind, "dataset_sha256": dataset_hashes(dataset), "top_k": top_k,
        "binary_sha256": sha256(binary), "vectors_sha256": sha256(vectors) if vectors else None,
        "recipe_sha256": provenance()["recipe_sha256"]}
    if top_k < 1:
        raise ValueError("top_k must be positive")
    start_directory(output, request)
    try:
        capture(output, binary)
        target = output / "retrieval"
        if kind == "replay":
            command = [str(binary), str(dataset.resolve()), str(vectors.resolve()), "--output", str(target.resolve()), "--top-k", str(top_k)]
        else:
            command = [str(binary), "benchmark", "run", str(dataset.resolve()), "--output", str(target.resolve()),
                       "--top-k", str(top_k), "--embedding", "hash" if kind == "smoke" else "minilm"]
            if kind == "smoke":
                command += ["--dimensions", "32"]
            else:
                command += ["--model-dir", str(model_dir.resolve())]
        run_process(command, output)
        from scoring import verify_exports
        checks = verify_exports(dataset, target / "results.json", target / "run.trec", target / "report.json")
        if dataset_hashes(dataset) != request["dataset_sha256"]:
            raise ValueError("dataset changed during retrieval")
        write_json(output / "verification.json", checks)
        if kind == "smoke":
            write_json(output / "scope.json", {"synthetic_hash_fixture": True, "semantic_quality_claim": False,
                       "public_benchmark_score": False, "paid_model_calls": 0})
        seal(output, "offline-smoke" if kind == "smoke" else "cubeql-retrieval")
        return checks
    except Exception as error:
        failure(output, error)
        raise


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("dataset", type=Path)
    p = sub.add_parser("prepare-apps")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p = sub.add_parser("fetch")
    p.add_argument("kind", choices=["coderank", "evaluator"])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p = sub.add_parser("smoke")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("encode-coderank")
    p.add_argument("dataset", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, choices=[256,8192], required=True)
    p.add_argument("--resume", action="store_true")
    for kind in ["replay", "minilm"]:
        p = sub.add_parser(kind)
        p.add_argument("dataset", type=Path)
        p.add_argument("--binary", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--top-k", type=int, default=100)
        p.add_argument("--vectors" if kind == "replay" else "--model-dir", type=Path, required=True)
    p = sub.add_parser("reference")
    p.add_argument("dataset", type=Path)
    p.add_argument("--vectors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--compare-report", type=Path)
    p = sub.add_parser("score")
    p.add_argument("dataset", type=Path)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--trec", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluator-cache", type=Path, required=True)
    p.add_argument("--cutoffs", type=int, nargs="+", default=[1,5,10,100])
    p.add_argument("--candidate-window", type=int)
    p.add_argument("--include-identical-ids", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("pack-baseline")
    p.add_argument("--legacy-run", type=Path, required=True)
    p.add_argument("--coderank-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audit-baseline")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("score-baseline")
    p.add_argument("directory", type=Path)
    p.add_argument("--mode", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--evaluator-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "validate":
        docs, queries, qrels, hashes = load_dataset(args.dataset)
        result = {"documents": len(docs), "queries": len(queries), "judgments": sum(map(len, qrels.values())),
                  "judged_queries": len(qrels), "unjudged_queries": len(queries)-len(qrels), "sha256": hashes}
    elif args.command == "prepare-apps":
        from data import prepare_apps
        prepare_apps(args.cache, args.output, args.download)
        result = verify_seal(args.output)
    elif args.command == "fetch":
        if args.kind == "evaluator":
            from scoring import evaluator_cache
            result = evaluator_cache(args.cache, args.download)
        else:
            spec = read_json(HERE / "specs/coderank.json")
            fetch_spec(spec, args.cache, args.download)
            result = {"model": spec["name"], "revision": spec["revision"], "status": "verified"}
    elif args.command == "smoke":
        result = run_cli(args.binary, REPO / "crates/cubeql-code/tests/fixtures/external-benchmark", args.output, "smoke", 3)
    elif args.command == "encode-coderank":
        from encoding import encode
        encode(args.dataset, args.model_dir, args.output, args.max_tokens, args.resume)
        result = verify_seal(args.output)
    elif args.command in {"replay", "minilm"}:
        result = run_cli(args.binary, args.dataset, args.output, args.command, args.top_k,
                         getattr(args, "vectors", None), getattr(args, "model_dir", None))
    elif args.command == "reference":
        from reference import exhaustive
        result = exhaustive(args.dataset, args.vectors, args.output, args.top_k, args.compare_report)
    elif args.command == "score":
        from scoring import score
        result = score(args.dataset, args.results, args.output, args.evaluator_cache, args.cutoffs,
            args.trec, not args.include_identical_ids, candidate_window=args.candidate_window)
    elif args.command == "pack-baseline":
        from baseline import pack
        result = pack(args.legacy_run, args.coderank_run, args.output)
    elif args.command == "audit-baseline":
        from baseline import audit
        result = audit(args.directory)
    elif args.command == "score-baseline":
        from baseline import rescore
        result = rescore(args.directory, args.mode, args.dataset, args.evaluator_cache, args.output)
    else:
        result = verify_seal(args.directory)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(1)
