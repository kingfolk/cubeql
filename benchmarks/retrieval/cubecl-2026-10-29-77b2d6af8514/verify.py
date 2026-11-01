#!/usr/bin/env python3
"""Verify the published evidence and recompute summaries without models or extraction."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
MODES = ('literal', 'symbol', 'lexical', 'semantic', 'experimental_rrf', 'hybrid')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        check(Path(name).name == name, 'unexpected publication path')
        check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
              'publication hash mismatch: ' + name)
    metadata = json.loads((ROOT / 'metadata.json').read_text())
    summary = json.loads((ROOT / 'summary.json').read_text())
    raw = {}
    with tarfile.open(ROOT / 'artifacts.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        check(len(members) == len({m.name for m in members}), 'duplicate archive paths')
        check({m.name for m in members} == set(metadata['archive']['files_sha256']),
              'archive inventory mismatch')
        for member in members:
            check(member.isfile(), 'unexpected non-file archive entry')
            content = archive.extractfile(member).read()
            check(hashlib.sha256(content).hexdigest() ==
                  metadata['archive']['files_sha256'][member.name], member.name)
            raw[member.name] = content

    def original(name):
        return json.loads(raw[name])

    check((ROOT / 'queries.json').read_bytes() == raw['queries-combined-v2.json'],
          'published labels differ from the frozen dataset')
    check((ROOT / 'wording-queries.json').read_bytes() == raw['paired-wording-diagnostic.json'],
          'published wording labels changed')
    for freeze_name in ('freeze-v2.json', 'diagnostics-freeze.json'):
        for name, digest in original(freeze_name)['files'].items():
            check(hashlib.sha256(raw[name]).hexdigest() == digest, 'frozen input: ' + name)
    check(not original('analysis.json')['separate_vs_combined_metric_mismatches'],
          'historical separate/combined mismatch')

    csv_rows = list(csv.DictReader(io.StringIO((ROOT / 'per-query-metrics.csv').read_text())))
    check((ROOT / 'per-query-metrics.csv').read_bytes() ==
          raw['per-query-metrics.csv'].replace(b'\r\n', b'\n'),
          'published CSV changed')
    expected_csv = {}
    per_set = {}
    for run in summary['sets']:
        name = run['name']
        check(original(name + '.execution.json')['exit_code'] == 0, 'failed run: ' + name)
        report = original(name + '.json')['evaluation']
        variant = report['variants'][0]
        check(variant['snapshot']['corpus_fingerprint'] == metadata['corpus']['fingerprint'],
              'corpus fingerprint: ' + name)
        check(variant['snapshot']['embedding'] == metadata['embedding'], 'embedding mismatch')
        check(report['options'] == run['options'], 'options mismatch')
        cases = {q['id']: q for q in report['dataset']['queries']}
        rows = variant['queries']
        per_set[name] = {q['id']: q for q in rows}
        for mode in MODES:
            all_anchors = []
            for q in rows:
                metric = q[mode]['metrics']
                expected = {(a['path'], a['symbol']) for a in cases[q['id']]['relevant']}
                hits = q[mode]['top_hits']
                for k in (5, 10):
                    covered = {(h['path'], h['symbol']) for h in hits[:k]}
                    check(abs(metric[f'recall_at_{k}'] - len(expected & covered)/len(expected))
                          < 1e-12, 'recall mismatch: ' + q['id'])
                complete = expected <= {(h['path'], h['symbol']) for h in hits[:10]}
                all_anchors.append(complete)
                expected_csv[(name, q['id'], mode)] = metric
            actual = run['overall'][mode]
            for key in ('recall_at_5', 'recall_at_10'):
                check(abs(actual[key] - statistics.mean(q[mode]['metrics'][key] for q in rows))
                      < 1e-12, 'aggregate recall mismatch')
            check(abs(actual['mrr'] - statistics.mean(q[mode]['metrics']['reciprocal_rank']
                                                    for q in rows)) < 1e-12, 'MRR mismatch')
            check(abs(actual['all_anchors_top10_rate'] - statistics.mean(all_anchors)) < 1e-12,
                  'completeness mismatch')
        check(run['by_category'] == variant['by_category'], 'category summary mismatch')
    check(len(csv_rows) == len(expected_csv), 'CSV row count mismatch')
    seen = set()
    for row in csv_rows:
        key = (row['set'], row['id'], row['mode'])
        check(key not in seen, 'duplicate CSV row')
        seen.add(key)
        for field, metric in (('recall5', 'recall_at_5'), ('recall10', 'recall_at_10'),
                              ('reciprocal_rank', 'reciprocal_rank')):
            check(float(row[field]) == expected_csv[key][metric], 'CSV metric mismatch')
    for name in ('v1-k50', 'extension-v2-k50'):
        for qid, q in per_set[name].items():
            for mode in MODES:
                check(q[mode]['metrics'] == per_set['combined-v2-k50'][qid][mode]['metrics'],
                      'separate/combined mismatch: ' + qid)
    dedup = original('dedup-analysis.json')['runs']
    check(len(dedup) == 48, 'expected 48 dedup diagnostic runs')
    pairs = {}
    for run in dedup:
        pairs.setdefault((run['id'], run['mode']), {})[run['deduplicate']] = run['metrics']
    check(len(pairs) == 24 and all(r[True] == r[False] for r in pairs.values()),
          'dedup diagnostic conclusion mismatch')
    print(f'OK: {len(members)} archived files, {len(csv_rows)} per-query rows, '
          'four main runs, frozen inputs and dedup diagnostics verified.')


if __name__ == '__main__':
    main()
