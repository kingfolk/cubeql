"""Verify published native development evidence without models or archive extraction."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
STRATEGIES = ('semantic', 'literal', 'experimental_rrf', 'lexical', 'symbol', 'hybrid')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def diagnostic_summary(full):
    """Keep reviewable summaries here; all per-query output remains archived."""
    variants = []
    for variant in full['variants']:
        hybrid = next(p for p in variant['paired_comparisons'] if p['strategy'] == 'hybrid')
        variants.append({
            'precision': variant['precision'], 'overall': variant['overall'],
            'by_category': variant['by_category'],
            'paired_comparisons': [{k: v for k, v in p.items() if k != 'queries'}
                                   for p in variant['paired_comparisons']],
            'hybrid_regressions': [q for q in hybrid['queries'] if q['outcome'] in ('lost', 'regressed')],
            'incomplete_hybrid_query_ids': [q['id'] for q in variant['queries'] if q['hybrid']['recall_at_10'] < 1],
        })
    return dict(full, variants=variants,
                full_diagnostics_archive_member='diagnostics/diagnostics.json')


def main():
    inventory = {}
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        check(Path(name).name == name and name not in inventory, 'publication path')
        check(sha((ROOT / name).read_bytes()) == digest, 'publication hash: ' + name)
        inventory[name] = digest
    check(set(inventory) | {'SHA256SUMS'} == {p.name for p in ROOT.iterdir()}, 'publication inventory')
    expected = json.loads((ROOT / 'evidence-members.json').read_bytes())
    raw = {}
    with tarfile.open(ROOT / 'evidence.tar.gz', 'r:gz') as archive:
        entries = archive.getmembers()
        check(len(entries) == len(expected) and len(entries) == len({m.name for m in entries}), 'archive inventory')
        check(sum(m.size for m in entries) <= 32 * 1024 * 1024, 'archive size limit')
        for member in entries:
            path = PurePosixPath(member.name)
            check(member.isfile() and not path.is_absolute() and '..' not in path.parts, 'archive path/type')
            content = archive.extractfile(member).read()
            check(member.name in expected and sha(content) == expected[member.name], 'archive hash: ' + member.name)
            raw[member.name] = content
    load = lambda name: json.loads(raw[name])
    metadata = json.loads((ROOT / 'experiment.json').read_bytes())
    recipe = load('run/recipe.json')
    canonical = json.dumps({k: v for k, v in recipe.items() if k != 'fingerprint'}, sort_keys=True,
                           separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    check(sha(canonical) == recipe['fingerprint'] == metadata['recipe_fingerprint'], 'recipe fingerprint')
    check(sha(raw['producer/repository_baseline.py']) == recipe['inputs']['baseline_source_sha256'], 'baseline producer')
    for name, digest in recipe['inputs']['runner_sources'].items():
        check(sha(raw['producer/' + Path(name).name]) == digest, 'runner producer')
    labels = raw['inputs/queries.json']
    check(labels == (ROOT / 'queries.json').read_bytes(), 'published labels')
    check(sha(labels) == recipe['inputs']['queries_sha256'] == metadata['queries_sha256'], 'label digest')
    summary = load('run/summary.json')
    check(summary == json.loads((ROOT / 'summary.json').read_bytes()), 'summary copy')
    check(summary['successful_condition_count'] == summary['expected_condition_count'] == 1, 'condition coverage')
    check(all(item['status'] == 'ok' for item in load('run/execution.json')), 'execution failure')
    condition = summary['conditions'][0]
    check(condition['query_count'] == len(json.loads(labels)['queries']), 'query denominator')
    check(condition['snapshot_sha256'] == metadata['snapshot_sha256'], 'snapshot fingerprint receipt')
    manifest = load('run/manifest.json')
    omitted = []
    for name, digest in manifest.items():
        if name.endswith('.arrow'):
            omitted.append(digest)
        else:
            check(sha(raw['run/' + name]) == digest, 'run manifest: ' + name)
    check(omitted == [metadata['snapshot_sha256']], 'excluded snapshot receipt')
    native = load(metadata['raw_native_report_member'])['evaluation']
    check(native['options'] == metadata['native_options'], 'native options')
    dataset = json.loads(labels)
    canonical_dataset = dict(dataset, queries=[dict(q, filter={'language': None, 'path_prefix': None,
                                                             **q.get('filter', {})}) for q in dataset['queries']])
    check(native['dataset'] == canonical_dataset, 'native label echo')
    cases = native['dataset']['queries']
    check(len(cases) == condition['query_count'], 'native query coverage')
    for variant, recorded in zip(native['variants'], condition['variants'], strict=True):
        check(variant['snapshot'] == recorded['snapshot'], 'snapshot summary')
        check(variant['overall'] == recorded['overall'] and variant['by_category'] == recorded['by_category'], 'metric summaries')
        rows = variant['queries']
        check([q['id'] for q in rows] == [q['id'] for q in cases], 'native query order')
        for case, row in zip(cases, rows, strict=True):
            anchors = {(a['path'], a['symbol']) for a in case['relevant']}
            for mode in STRATEGIES:
                hits = row[mode]['top_hits']
                metric = row[mode]['metrics']
                for k in (5, 10):
                    retrieved = {(h['path'], h['symbol']) for h in hits[:k]}
                    check(abs(metric[f'recall_at_{k}'] - len(anchors & retrieved) / len(anchors)) < 1e-12,
                          'display-hit recall: ' + case['id'])
                rank = metric['first_relevant_rank']
                check(abs(metric['reciprocal_rank'] - (1 / rank if rank else 0)) < 1e-12, 'reciprocal rank')
        for mode in STRATEGIES:
            check(recorded['overall'][mode]['queries'] == len(rows), 'overall denominator')
            for target, field in [('recall_at_5', 'recall_at_5'), ('recall_at_10', 'recall_at_10'), ('mrr', 'reciprocal_rank')]:
                value = statistics.mean(row[mode]['metrics'][field] for row in rows)
                check(abs(value - recorded['overall'][mode][target]) < 1e-12, 'overall mean')
                for category, group in variant['by_category'].items():
                    selected = [row for row in rows if row['category'] == category]
                    check(group[mode]['queries'] == len(selected), 'category denominator')
                    value = statistics.mean(row[mode]['metrics'][field] for row in selected)
                    check(abs(value - group[mode][target]) < 1e-12, 'category mean')
    check(json.loads((ROOT / 'diagnostics.json').read_bytes()) == diagnostic_summary(load('diagnostics/diagnostics.json')), 'diagnostic projection')
    print(f'OK: {len(entries)} archived files; {len(cases)} original queries; native recall and aggregate metrics verified.')
    print('MRR beyond the ten display hits is checked against native rank receipts, not a full ranking export.')


if __name__ == '__main__':
    main()
