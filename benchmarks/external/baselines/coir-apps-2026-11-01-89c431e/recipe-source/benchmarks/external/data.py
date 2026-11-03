"""Canonical data loading and pinned downloads; neither model nor device is initialized."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import unicodedata

from common import HERE, atomic_file, capture, dataset_hashes, decode_json, failure, read_json, seal, sha256, start_directory, write_json


def token(value) -> bool:
    return isinstance(value, str) and bool(value) and not any(c.isspace() or unicodedata.category(c) == "Cc" for c in value)


def load_dataset(root: Path, split="test"):
    hashes = dataset_hashes(root, split)
    rows = {}
    for name in ["corpus", "queries"]:
        seen, values = set(), []
        for number, line in enumerate((root / f"{name}.jsonl").read_text().splitlines(), 1):
            try:
                record = decode_json(line, f"{name}.jsonl:{number}")
                if not token(record["_id"]) or record["_id"] in seen:
                    raise ValueError("invalid/duplicate ID")
                if not isinstance(record["text"], str):
                    raise ValueError("text must be a string")
                if name == "queries" and not record["text"].strip():
                    raise ValueError("empty query text")
                if name == "corpus" and record.get("title") is not None and not isinstance(record["title"], str):
                    raise ValueError("title must be a string or null")
                seen.add(record["_id"])
                values.append(record)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{name}.jsonl:{number}: {error}") from error
        if not values:
            raise ValueError(f"empty {name}")
        rows[name] = values
    docs = {row["_id"] for row in rows["corpus"]}
    queries = {row["_id"] for row in rows["queries"]}
    qrels = {}
    with (root / f"qrels/{split}.tsv").open(newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        if next(reader, None) != ["query-id", "corpus-id", "score"]:
            raise ValueError("invalid qrels header")
        for number, fields in enumerate(reader, 2):
            if len(fields) != 3:
                raise ValueError(f"qrels line {number}: expected three fields")
            qid, did, grade = fields
            if qid not in queries or did not in docs or did in qrels.get(qid, {}):
                raise ValueError(f"qrels line {number}: duplicate or unknown reference")
            try:
                if not re.fullmatch(r"[+-]?[0-9]+", grade):
                    raise ValueError("invalid integer syntax")
                value = int(grade)
                if not -(2**31) <= value < 2**31:
                    raise ValueError("grade exceeds signed 32-bit range")
            except ValueError as error:
                raise ValueError(f"qrels line {number}: integer grade required") from error
            qrels.setdefault(qid, {})[did] = value
    return rows["corpus"], rows["queries"], qrels, hashes


def document_text(row):
    return (row["title"] + "\n" if row.get("title") else "") + row["text"]


def fetch_spec(spec: dict, cache: Path, download=False) -> dict[str, Path]:
    """Checksummed caches only. Existing corrupt files are never replaced automatically."""
    paths = {}
    for name, item in spec["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid pinned filename")
        path = cache / name
        if not path.exists():
            if not download:
                raise ValueError(f"missing pinned file: {path}; use explicit --download to fetch")
            print(f"downloading {name}", flush=True)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "CubeQL-benchmark"})
            def stream(output):
                hasher = hashlib.sha256()
                with urllib.request.urlopen(request, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        hasher.update(chunk)
                        output.write(chunk)
                if hasher.hexdigest() != item["sha256"]:
                    raise ValueError(f"download checksum mismatch: {name}")
            atomic_file(path, stream)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"existing pinned file has conflicting bytes: {path}")
        paths[name] = path
    return paths


def prepare_apps(cache: Path, output: Path, download=False):
    spec = read_json(HERE / "specs/coir-apps.json")
    paths = fetch_spec(spec, cache, download)
    import pyarrow.parquet as pq
    corpus = pq.read_table(paths["corpus.parquet"]).to_pylist()
    queries = pq.read_table(paths["queries.parquet"]).to_pylist()
    judgments = pq.read_table(paths["test-qrels.parquet"]).to_pylist()
    if [len(corpus), len(queries), len(judgments)] != [8765, 8765, 3765]:
        raise ValueError("pinned APPS release has unexpected row counts")
    by_query = {q["_id"]: q for q in queries}
    if len(by_query) != len(queries) or len({d["_id"] for d in corpus}) != len(corpus):
        raise ValueError("duplicate source IDs")
    selected_ids = list(dict.fromkeys(j["query_id"] for j in judgments))
    if len(selected_ids) != 3765 or any(j["score"] != 1 for j in judgments):
        raise ValueError("unexpected APPS test judgments; do not silently relabel")
    request = {"operation": "prepare-apps", "spec": spec, "selection": "official-test-qrels-query-IDs"}
    start_directory(output, request)
    try:
        capture(output)
        dataset = output / "dataset"
        (dataset / "qrels").mkdir(parents=True)
        for name, values in [("corpus", corpus), ("queries", [by_query[q] for q in selected_ids])]:
            with (dataset / f"{name}.jsonl").open("x", encoding="utf-8") as target:
                for row in values:
                    record = {"_id": row["_id"], "text": row["text"]}
                    if name == "corpus":
                        record["title"] = row.get("title")
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (dataset / "qrels/test.tsv").open("x") as target:
            target.write("query-id\tcorpus-id\tscore\n")
            for row in judgments:
                target.write(f"{row['query_id']}\t{row['corpus_id']}\t{row['score']}\n")
        docs, selected, qrels, hashes = load_dataset(dataset)
        if hashes != spec["converted_sha256"]:
            raise ValueError("converted APPS bytes differ from frozen baseline")
        write_json(output / "summary.json", {"documents": len(docs), "queries": len(selected),
                   "judgments": sum(map(len, qrels.values())), "dataset_sha256": hashes,
                   "source_revisions": spec["source_revisions"], "query_selection": request["selection"]})
        seal(output, "pinned-coir-apps-inputs")
    except Exception as error:
        failure(output, error)
        raise
