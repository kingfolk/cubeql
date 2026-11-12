"""Produce descriptive paired repository diagnostics from saved, validated rankings."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

import repository_eval as ev


def _latency(run):
    successful = [r for r in run['results'] if r['status'] == 'ok']
    samples = [r['elapsed_ms'] for r in successful if 'elapsed_ms' in r]
    return {'timing_boundary': run['configuration'].get('timing_boundary', 'unspecified'),
            'successful_query_count': len(successful), 'sample_count': len(samples),
            'complete_query_coverage': len(samples) == len(run['results']),
            'median_ms': statistics.median(samples) if samples else None,
            'min_ms': min(samples) if samples else None,
            'max_ms': max(samples) if samples else None,
            'population': 'successful queries with elapsed_ms; failures excluded from latency only'}


def _outcome(before, after):
    if before == after == 0:
        return 'shared_miss'
    if before == 0 < after:
        return 'recovered'
    if after == 0 < before:
        return 'lost'
    return 'improved' if after > before else 'regressed' if after < before else 'tied'


def _cell(value):
    return html.escape(str(value)).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _markdown(report):
    lines = ['# Repository retrieval diagnostics', '',
             'Descriptive location-coverage results on the declared frozen queries. '
             'These are not coding-task success rates or controlled equal-work engine comparisons.', '',
             '| Condition | Queries | Successful coverage | Anchor Recall@10 | All required locations@10 | Exported-window recall | Request median ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in report['runs']:
        overall = run['overall']
        median = run['latency']['median_ms']
        median_text = f'{median:.3f}' if median is not None else 'unavailable'
        lines.append(f"| {_cell(run['name'])} | {overall['query_count']} | {overall['coverage']:.4f} | "
                     f"{overall['recall_at_10']:.4f} | {overall['all_required_at_10']:.4f} | "
                     f"{run['window_recall_query_macro']:.4f} | {median_text} |")
    lines.extend(['', 'Each request timing boundary is recorded below; startup, indexing, source reads and native '
                  'candidate budgets may differ. No speedup ratio or causal ranking attribution is made.', ''])
    for run in report['runs']:
        lines.append(f"- {_cell(run['name'])}: {_cell(run['latency']['timing_boundary'])}; "
                     f"{run['latency']['sample_count']} latency observations; exported window {run['candidate_k']}.")
    for paired in report['paired_comparisons']:
        lines.extend(['', f"## {_cell(paired['candidate'])} versus {_cell(paired['baseline'])}", '',
                      f"Recall outcomes: {_cell(json.dumps(paired['outcome_counts'], sort_keys=True))}.", '',
                      '| Query | Repository | Outcome | Baseline recall@10 | Candidate recall@10 | Candidate diagnosis |',
                      '| --- | --- | --- | ---: | ---: | --- |'])
        for q in paired['queries']:
            lines.append(f"| {_cell(q['query_id'])} | {_cell(q['repository'])} | {q['outcome']} | "
                         f"{q['baseline_recall_at_10']:.4f} | {q['candidate_recall_at_10']:.4f} | "
                         f"{q['candidate_failure_class']} |")
    lines.extend(['', '## Interpretation limits', ''])
    lines.extend(f'- {value}' for value in report['limits'])
    return '\n'.join(lines) + '\n'


def diagnose(plan, runs, output):
    """Validate all runs first; preserve complete query denominators and native configurations."""
    score = ev.score(plan, runs)
    queries = plan['spec']['queries']
    diagnoses = []
    for raw, scored in zip(runs, score['runs']):
        rows = {}
        for query in queries:
            item = scored['queries'][query['id']]
            ranks = item['anchor_ranks']
            missing = [name for name, rank in ranks.items() if rank is None]
            below = [name for name, rank in ranks.items() if rank is not None and rank > 10]
            failure_class = ('execution_failure' if item['status'] != 'ok' else
                             'missing_from_exported_window' if missing else
                             'present_below_top10' if below else 'all_locations_in_top10')
            rows[query['id']] = {'query_id': query['id'], 'repository': query['repository'],
                'category': query['category'], 'status': item['status'], 'anchor_ranks': ranks,
                'missing_anchor_ids': missing, 'below_top10_anchor_ids': below,
                'window_recall': (len(ranks) - len(missing)) / len(ranks),
                'recall_at_10': item['recall_at']['10'],
                'all_required_at_10': item['all_required_at_10'], 'failure_class': failure_class}
        by_repo = {}
        for repo in plan['spec']['repositories']:
            group = [row for row in rows.values() if row['repository'] == repo['id']]
            by_repo[repo['id']] = {'query_count': len(group),
                'window_recall': sum(r['window_recall'] for r in group) / len(group),
                'failure_classes': dict(sorted(Counter(r['failure_class'] for r in group).items()))}
        condition = {'name': raw['name'], 'candidate_k': raw['candidate_k'],
            'configuration': raw['configuration'], 'tool': raw['tool'], 'overall': scored['overall'],
            'by_repository': by_repo, 'by_category': scored['by_category'],
            'window_recall_query_macro': sum(r['window_recall'] for r in rows.values()) / len(rows),
            'window_recall_repository_macro': sum(r['window_recall'] for r in by_repo.values()) / len(by_repo),
            'latency': _latency(raw), 'queries': rows}
        if 'repository_macro' in scored:
            condition['repository_macro'] = scored['repository_macro']
        diagnoses.append(condition)
    paired_diagnostics = []
    for candidate, paired in zip(diagnoses[1:], score['paired_comparisons']):
        baseline = diagnoses[0]
        rows = []
        for query in queries:
            b, c = baseline['queries'][query['id']], candidate['queries'][query['id']]
            rows.append({'query_id': query['id'], 'repository': query['repository'],
                         'category': query['category'],
                         'outcome': _outcome(b['recall_at_10'], c['recall_at_10']),
                         'baseline_recall_at_10': b['recall_at_10'],
                         'candidate_recall_at_10': c['recall_at_10'],
                         'candidate_failure_class': c['failure_class']})
        btime, ctime = baseline['latency'], candidate['latency']
        paired_diagnostics.append({'baseline': baseline['name'], 'candidate': candidate['name'],
            'queries': rows, 'outcome_counts': dict(sorted(Counter(r['outcome'] for r in rows).items())),
            'macro_delta': paired['macro_delta'],
            'same_declared_timing_boundary': btime['timing_boundary'] != 'unspecified' and btime['timing_boundary'] == ctime['timing_boundary'],
            'latency_complete_both_conditions': btime['complete_query_coverage'] and ctime['complete_query_coverage'],
            'equal_exported_window': baseline['candidate_k'] == candidate['candidate_k'],
            'equal_internal_work_attested': False})
    report = {'format_version': 1, 'plan_fingerprint': plan['fingerprint'], 'dataset': plan['spec']['name'],
        'split': plan['spec']['split'], 'runs': diagnoses, 'paired_comparisons': paired_diagnostics,
        'limits': [
            'Recall and all-required metrics measure labeled source locations, not proof that every source byte was returned or used.',
            'An anchor missing from an exported top-k window may exist in an internal candidate pool; this is not a candidate-generation diagnosis.',
            'Unjudged results are not automatically false positives. Development queries and incomplete judgments do not establish held-out generalization.',
            'Different native chunking, models, reranking and candidate budgets remain product-condition differences, not isolated ranking or engine effects.',
            'Native response payloads and source-read work can differ even when the resident request timing boundary has the same name; no equal-context latency is attested.',
            'Latency summaries exclude failed queries and missing timing observations; coverage is explicit. Matched timing boundaries do not establish equal internal work.',
            'These descriptive paired counts are not significance tests, model forecasts, or coding-agent task outcomes.']}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ev._write_new(output / 'plan.json', plan)
    ev._write_new(output / 'runs.json', runs)
    ev._write_new(output / 'scores.json', score)
    ev._write_new(output / 'diagnostics.json', report)
    markdown = _markdown(report)
    with (output / 'README.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown)
    ev._write_new(output / 'manifest.json', {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()})
    return {'report': report, 'score': score, 'markdown': markdown}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True, action='append')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        diagnose(ev._load(args.plan), [ev._load(path) for path in args.run], args.output)
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f'repository_diagnostics: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
