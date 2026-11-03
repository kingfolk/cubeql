"""Export checks and native CoIR metrics. Never infer relevance from unjudged queries."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys

from common import HERE, capture, failure, provenance, read_json, seal, sha256, start_directory, versions, write_json
from data import fetch_spec, load_dataset, token


def evaluator_cache(cache: Path, download=False) -> dict:
    spec = read_json(HERE / "specs/coir-evaluator.json")
    fetch_spec(spec, cache, download)
    for name in ["native_coir/__init__.py", "native_coir/search/__init__.py"]:
        path = cache / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x"):
                pass
        if path.read_bytes() != b"":
            raise ValueError(f"conflicting native evaluator package stub: {path}")
    return spec


def verify_exports(dataset: Path, results_path: Path, trec_path: Path, report_path: Path | None = None,
                   split="test") -> dict:
    documents, queries, qrels, hashes = load_dataset(dataset, split)
    dids, qids = {d["_id"] for d in documents}, {q["_id"] for q in queries}
    results = read_json(results_path)
    if not isinstance(results, dict) or set(results) != qids:
        raise ValueError("results must contain every input query, including unjudged queries")
    for qid, scores in results.items():
        if not isinstance(scores, dict) or not set(scores) <= dids:
            raise ValueError(f"unknown document or invalid scores for {qid}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values()):
            raise ValueError("all scores must be finite numbers")
    trec = {qid: {} for qid in qids}
    ranks = {qid: {} for qid in qids}
    tags = set()
    for number, line in enumerate(trec_path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"TREC line {number}: expected six columns")
        qid, iteration, did, rank, value, tag = fields
        if qid not in qids or did not in dids or iteration != "Q0" or not token(tag):
            raise ValueError(f"TREC line {number}: invalid ID, iteration or tag")
        rank, score = int(rank), float(value)
        if not math.isfinite(score) or rank < 1 or did in trec[qid] or rank in ranks[qid]:
            raise ValueError(f"TREC line {number}: invalid/duplicate rank or document")
        trec[qid][did] = score
        ranks[qid][rank] = did
        tags.add(tag)
    for qid in qids:
        if set(trec[qid]) != set(results[qid]) or sorted(ranks[qid]) != list(range(1, len(trec[qid]) + 1)):
            raise ValueError(f"JSON/TREC IDs or rank sequence differ for {qid}")
        for did, score in trec[qid].items():
            if not math.isclose(score, results[qid][did], rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"JSON/TREC score differs for {qid}/{did}")
        ordered_scores = [trec[qid][ranks[qid][r]] for r in sorted(ranks[qid])]
        if any(a < b for a, b in zip(ordered_scores, ordered_scores[1:])):
            raise ValueError(f"TREC scores are not descending for {qid}")
    if len(tags) > 1:
        raise ValueError("TREC contains more than one run tag")
    if report_path:
        report = read_json(report_path)
        if report.get("status") != "complete" or [q["query_id"] for q in report["queries"]] != [q["_id"] for q in queries]:
            raise ValueError("incomplete report or query order mismatch")
        for query in report["queries"]:
            qid = query["query_id"]
            if [hit["document_id"] for hit in query["hits"]] != [ranks[qid][r] for r in sorted(ranks[qid])]:
                raise ValueError(f"report/TREC rank order differs for {qid}")
            if {h["document_id"]: h["score"] for h in query["hits"]} != results[qid]:
                raise ValueError(f"report/JSON scores differ for {qid}")
    return {"status": "verified", "documents": len(dids), "queries": len(qids),
            "judged_queries": len(qrels), "unjudged_queries": len(qids - set(qrels)),
            "trec_rows": sum(map(len, trec.values())), "dataset_sha256": hashes,
            "tie_policy": "TREC/report retain engine order; score-based evaluators may order ties differently"}


def load_evaluator(cache: Path):
    # Separate caches cannot accidentally reuse an already imported module from another path.
    import hashlib
    name = "_cubeql_coir_" + hashlib.sha256(str(cache.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, cache / "native_coir/__init__.py",
                submodule_search_locations=[str(cache / "native_coir")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return __import__(name + ".evaluation", fromlist=["EvaluateRetrieval"]).EvaluateRetrieval


def score(dataset: Path, results_path: Path, output: Path, cache: Path, cutoffs: list[int],
          trec_path: Path | None = None, ignore_identical_ids=True, split="test", candidate_window=None):
    spec = evaluator_cache(cache)
    import pytrec_eval
    installed = versions(["pytrec-eval-terrier", "numpy"])
    if installed["pytrec-eval-terrier"] != spec["pytrec_eval_terrier"]:
        raise ValueError("install the pinned pytrec-eval-terrier version before scoring")
    documents, queries, qrels, hashes = load_dataset(dataset, split)
    if not qrels:
        raise ValueError("native scoring needs at least one judged query; unjudged is not no-answer")
    cutoffs = sorted(set(cutoffs))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("cutoffs must be positive")
    results = read_json(results_path)
    qids, dids = {q["_id"] for q in queries}, {d["_id"] for d in documents}
    if set(results) != qids:
        raise ValueError("results must retain every input query, including unjudged ones")
    for hits in results.values():
        if not isinstance(hits, dict) or not set(hits) <= dids:
            raise ValueError("invalid document scores")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in hits.values()):
            raise ValueError("scores must be finite numbers")
    report_path = results_path.parent / "report.json"
    window_source = "caller-supplied"
    if report_path.exists():
        report = read_json(report_path)
        recorded_window = report["options"]["top_k"]
        if candidate_window is not None and candidate_window != recorded_window:
            raise ValueError("candidate window conflicts with completed run report")
        candidate_window, window_source = recorded_window, "completed-run-report"
    if candidate_window is None or candidate_window < 1:
        raise ValueError("supply the exported candidate window or a completed sibling report.json")
    # Sparse/empty retrievals are valid misses. A declared budget, not result count,
    # distinguishes them from a deliberately truncated export.
    if min(max(cutoffs), len(documents)) > candidate_window or any(len(hits) > candidate_window for hits in results.values()):
        raise ValueError("cutoff/results exceed the declared candidate window")
    checks = verify_exports(dataset, results_path, trec_path, split=split) if trec_path else None
    request = {"operation": "native-score", "dataset_sha256": hashes,
        "results_sha256": sha256(results_path), "trec_sha256": sha256(trec_path) if trec_path else None,
        "cutoffs": cutoffs, "candidate_window": candidate_window, "window_source": window_source,
        "ignore_identical_ids": ignore_identical_ids,
        "evaluator": spec, "versions": installed, "recipe_sha256": provenance()["recipe_sha256"]}
    start_directory(output, request)
    try:
        capture(output)
        effective = {qid: {did: score for did, score in results[qid].items()
                           if not ignore_identical_ids or qid != did} for qid in qrels}
        evaluator = load_evaluator(cache)
        native = evaluator.evaluate(qrels, copy.deepcopy(effective), cutoffs, ignore_identical_ids=False)
        mrr = evaluator.evaluate_custom(qrels, effective, cutoffs, "mrr")
        metrics = {"ndcg_cut." + ",".join(map(str, cutoffs)),
                   "recall." + ",".join(map(str, cutoffs)), "recip_rank"}
        per = pytrec_eval.RelevanceEvaluator(qrels, metrics).evaluate(effective)
        if set(per) != set(qrels):
            raise ValueError("native evaluator omitted judged queries; denominator would be ambiguous")
        aggregate = {key: sum(p[key] for p in per.values()) / len(qrels) for key in next(iter(per.values()))}
        for cutoff in cutoffs:
            if abs(aggregate[f"ndcg_cut_{cutoff}"] - native[0][f"NDCG@{cutoff}"]) > 0.000006:
                raise ValueError("native/direct pytrec nDCG differs")
            if abs(aggregate[f"recall_{cutoff}"] - native[2][f"Recall@{cutoff}"]) > 0.000006:
                raise ValueError("native/direct pytrec recall differs")
        summary = {"judged_queries": len(qrels), "unjudged_queries_excluded": len(qids - set(qrels)),
            "candidate_window": candidate_window, "candidate_window_source": window_source,
            "actual_returned_range": [min(map(len, results.values())), max(map(len, results.values()))], "cutoffs": cutoffs,
            "native_coir": {"ndcg": native[0], "map": native[1], "recall": native[2], "precision": native[3], "mrr": mrr},
            "direct_pytrec_unrounded": aggregate,
            "pytrec_reciprocal_rank_window": "exported candidate window, after explicit self-ID filtering",
            "mrr_tie_policy": "native stable JSON score sort and pytrec document-ID tie sort are both retained",
            "ignore_identical_ids": ignore_identical_ids, "judgments_modified": False,
            "paid_model_calls": 0}
        write_json(output / "summary.json", summary)
        write_json(output / "per-query.json", per)
        if checks:
            write_json(output / "exports.json", checks)
        seal(output, "native-coir-scores")
        return summary
    except Exception as error:
        failure(output, error)
        raise
