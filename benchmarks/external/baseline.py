"""Portable historical evidence: own rankings/metrics, without corpus text or weights."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path, PurePosixPath
import tarfile

from common import HERE, capture, dataset_hashes, failure, read_json, seal, sha256, start_directory, verify_seal, write_json


def pack(legacy: Path, coderank: Path, output: Path):
    data_spec = read_json(HERE / "specs/coir-apps.json")
    frozen = read_json(legacy / "freeze.json")
    expected = {"dataset/" + name: value for name, value in data_spec["converted_sha256"].items()}
    if frozen["input_sha256"] != expected:
        raise ValueError("historical MiniLM inputs do not match pinned APPS")
    for context in [256, 8192]:
        if read_json(coderank / f"context-{context}/encoding-freeze.json")["input_sha256"] != expected:
            raise ValueError("historical CodeRank inputs do not match pinned APPS")
    request = {"operation": "package-historical-baseline", "dataset": data_spec,
               "legacy_freeze_sha256": sha256(legacy / "freeze.json"),
               "coderank_summary_sha256": sha256(coderank / "summary.json")}
    start_directory(output, request)
    try:
        capture(output)
        files, modes = {}, {}
        old_summary = read_json(legacy / "summary.json")
        for name in ["minilm", "bm25_word", "bm25_code", "experimental_rrf"]:
            result = legacy / ("minilm/results.json" if name == "minilm" else f"{name}.results.json")
            trec = legacy / ("minilm/run.trec" if name == "minilm" else f"{name}.trec")
            prefix = f"rankings/{name}"
            files[prefix + "/results.json"], files[prefix + "/run.trec"] = result, trec
            modes[name] = {"results": prefix + "/results.json", "trec": prefix + "/run.trec",
                           "expected": old_summary["models"][name]}
        for context in [256,8192]:
            for engine in ["cubeql", "reference"]:
                name = f"coderank_{context}_{engine}"
                root = coderank / f"context-{context}"
                result = root / ("cubeql/results.json" if engine == "cubeql" else "reference.results.json")
                trec = root / ("cubeql/run.trec" if engine == "cubeql" else "reference.trec")
                prefix = f"rankings/{name}"
                files[prefix + "/results.json"], files[prefix + "/run.trec"] = result, trec
                modes[name] = {"results": prefix + "/results.json", "trec": prefix + "/run.trec",
                    "expected": read_json(root / f"{engine}.scores.json")["summary"]}
        for name in ["freeze.json", "summary.json", "per-query-metrics.csv", "tie-policy-diagnostic.json"]:
            files["historical/minilm/" + name] = legacy / name
        for name in ["summary.json", "per-query-comparison.csv", "tie-gap-diagnostic.json", "baseline-source-equivalence.json"]:
            files["historical/coderank/" + name] = coderank / name
        for context in [256,8192]:
            for name in ["encoding-freeze.json", "encoding-complete.json", "retrieval-freeze.json", "engine-parity.json"]:
                files[f"historical/coderank/context-{context}/{name}"] = coderank / f"context-{context}" / name
        for name in ["failed-context-256-model-kwargs/failure.json", "minilm-fidelity-preflight-failure.json",
                     "source-freeze-amendment.json"]:
            files["historical/coderank/" + name] = coderank / name
        members = {name: sha256(source) for name, source in files.items()}
        # Fixed metadata makes packaging deterministic; no local paths or user IDs in tar headers.
        archive = output / "evidence.tar.gz"
        with archive.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as target:
                for name, source in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = source.stat().st_size, 0o644, 0
                    with source.open("rb") as content:
                        target.addfile(info, content)
        index = {"format_version": 1, "benchmark": "CoIR APPS full official test task",
            "documents": 8765, "queries": 3765, "candidate_window": 100, "cutoffs": [1,5,10,100],
            "dataset_sha256": data_spec["converted_sha256"], "modes": modes,
            "archive": {"file": archive.name, "sha256": sha256(archive), "members_sha256": members},
            "evidence_status": "historical locally measured runs; packaging does not rerun inference",
            "original_source_note": "MiniLM used the adapter before commit 89c431e; original freeze retains dirty source identity. CodeRank helper and dependency identities remain separate.",
            "excluded": ["corpus/query text", "qrels", "model weights", "Arrow indexes", "vectors", "third-party evaluator source"],
            "new_paid_model_calls": 0}
        write_json(output / "baseline.json", index)
        lines = ["# Frozen CoIR APPS baseline — 2026-11-01", "",
            "Historical locally measured results over all 8,765 programs and 3,765 official test queries.",
            "Packaging these files does not constitute a new inference run or a competitor reproduction.", "",
            "| Configuration | nDCG@10 | Recall@10 | MRR@100 |", "|---|---:|---:|---:|"]
        for name, mode in modes.items():
            s = mode["expected"]["native_coir"]
            lines.append(f"| {name} | {s['ndcg']['NDCG@10']:.5f} | {100*s['recall']['Recall@10']:.2f}% | {s['mrr']['MRR@100']:.5f} |")
        lines += ["", "nDCG/MRR are normalized, not accuracy. MRR is limited to exported Top-100 candidates.",
            "This is one CoIR task, not its ten-task average or an agent evaluation. Each query has one labeled positive; equivalent unlabeled programs may exist.", "",
            "## Verify and rescore", "",
            "From the repository root, audit archive bytes without installing models or extracting corpus data:", "",
            "```sh", "python3 -B benchmarks/external/runner.py audit-baseline benchmarks/external/baselines/" + output.name, "```", "",
            "Pass this directory's actual path to the command. See [the portable recipe](../../README.md) for dataset preparation and `score-baseline`.",
            "The archive contains own ID/score exports, per-query metrics, original provenance and failed preflight diagnostics. It excludes corpus/query text, qrels, vectors, weights and third-party source.", "",
            "The original MiniLM run used a dirty adapter checkout before commit `89c431e`; do not infer binary build identity from the directory suffix. Original executable/source/dependency fingerprints are preserved.",
            "RRF contains many ties: native CoIR's stable JSON sort and pytrec's document-ID tie ordering have different MRR. Both values remain in baseline.json and the evidence; scores are not perturbed.",
            "Published model/peer numbers are not substituted for local measurements. The CodeRank context comparison was exploratory after the MiniLM result, and changes both query/document token ceilings.", ""]
        (output / "README.md").write_text("\n".join(lines))
        seal(output, "historical-baseline-bundle")
        return audit(output)
    except Exception as error:
        failure(output, error)
        raise


def audit(directory: Path) -> dict:
    verify_seal(directory)
    index = read_json(directory / "baseline.json")
    if index.get("format_version") != 1 or Path(index["archive"]["file"]).name != index["archive"]["file"]:
        raise ValueError("unsupported baseline archive")
    archive = directory / index["archive"]["file"]
    if sha256(archive) != index["archive"]["sha256"]:
        raise ValueError("baseline archive checksum mismatch")
    expected, seen = index["archive"]["members_sha256"], set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = member.name
            path = PurePosixPath(name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or name not in expected or name in seen:
                raise ValueError("invalid, duplicate or unexpected tar member")
            with source.extractfile(member) as content:
                digest = hashlib.file_digest(content, "sha256").hexdigest()
            if digest != expected[name]:
                raise ValueError(f"tar member checksum mismatch: {name}")
            seen.add(name)
    if seen != set(expected):
        raise ValueError("missing baseline archive members")
    return {"status": "verified", "modes": list(index["modes"]), "archive_members": len(seen),
            "model_inference_required": False, "paid_model_calls": 0}


def rescore(directory: Path, mode: str, dataset: Path, cache: Path, output: Path):
    audit(directory)
    index = read_json(directory / "baseline.json")
    if dataset_hashes(dataset) != index["dataset_sha256"]:
        raise ValueError("baseline requires the original pinned dataset bytes and judgments")
    entry = index["modes"][mode]
    start_directory(output, {"operation": "rescore-historical-baseline", "mode": mode,
        "baseline_sha256": sha256(directory / "baseline.json"), "dataset_sha256": index["dataset_sha256"]})
    try:
        capture(output)
        inputs = output / "rankings"
        inputs.mkdir()
        with tarfile.open(directory / index["archive"]["file"], "r:gz") as archive:
            for key, name in [("results", "results.json"), ("trec", "run.trec")]:
                with archive.extractfile(entry[key]) as source, (inputs / name).open("xb") as target:
                    target.write(source.read())
        from scoring import score
        result = score(dataset, inputs / "results.json", output / "scores", cache,
            index["cutoffs"], inputs / "run.trec", candidate_window=index["candidate_window"])
        expected, actual = entry["expected"], result["native_coir"]
        checks = {
            "ndcg10": [actual["ndcg"]["NDCG@10"], expected["native_coir"]["ndcg"]["NDCG@10"]],
            "recall100": [actual["recall"]["Recall@100"], expected["native_coir"]["recall"]["Recall@100"]],
            "mrr100": [actual["mrr"]["MRR@100"], expected["native_coir"]["mrr"]["MRR@100"]],
            "pytrec_mrr100": [result["direct_pytrec_unrounded"]["recip_rank"], expected["pytrec_mrr_at_100"]],
        }
        if any(abs(a-b) > 0.000006 for a,b in checks.values()):
            raise ValueError("rescored baseline differs from historical metrics; retain and investigate")
        write_json(output / "comparison.json", {"status": "matches-historical-scores", "checks": checks})
        seal(output, "rescored-historical-baseline")
        return checks
    except Exception as error:
        failure(output, error)
        raise
