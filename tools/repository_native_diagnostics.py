"""Validate and summarize a saved native symbol-label evaluation report.

This consumer is model-free: it reads one completed native report and writes a
create-new diagnostic bundle. It never runs the indexer, evaluator, or a model.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import repository_baseline as baseline
import repository_eval as ev


STRATEGIES = ('semantic', 'literal', 'experimental_rrf', 'lexical', 'symbol', 'hybrid')
PAIRS = ('lexical', 'symbol', 'hybrid')
OUTCOMES = ('recovered', 'lost', 'improved', 'regressed', 'shared_miss', 'tied')
TOLERANCE = 1e-9


def _number(value, where, *, unit=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ev.EvaluationError(f'{where} must be a finite number')
    try:
        finite = math.isfinite(value)
        converted = float(value)
    except OverflowError as error:
        raise ev.EvaluationError(f'{where} must be a finite number') from error
    if not finite:
        raise ev.EvaluationError(f'{where} must be a finite number')
    if value < 0 or (unit and value > 1):
        raise ev.EvaluationError(f'{where} must be in [0, 1]' if unit else f'{where} must be nonnegative')
    return converted


def _signed_number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ev.EvaluationError(f'{where} must be a finite number')
    try:
        if not math.isfinite(value):
            raise ev.EvaluationError(f'{where} must be a finite number')
        return float(value)
    except OverflowError as error:
        raise ev.EvaluationError(f'{where} must be a finite number') from error


def _parse_json(raw_bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ev.EvaluationError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def bad_constant(value):
        raise ev.EvaluationError(f'non-finite JSON number: {value}')

    try:
        return json.loads(raw_bytes.decode('utf-8'), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ev.EvaluationError(f'cannot parse native report JSON: {error}') from error


def _keys(value, required, where):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ev.EvaluationError(f'{where} keys mismatch')


def _close(actual, expected, where):
    if abs(actual - expected) > TOLERANCE:
        raise ev.EvaluationError(f'{where} does not match per-query recomputation')


def _metrics(strategy, label_pairs, candidate_k, where):
    _keys(strategy, ('metrics', 'retrieval', 'materialization_ms', 'top_hits'), where)
    metrics = strategy['metrics']
    _keys(metrics, ('recall_at_5', 'recall_at_10', 'reciprocal_rank', 'first_relevant_rank'), f'{where}.metrics')
    recall5 = _number(metrics['recall_at_5'], f'{where}.recall_at_5', unit=True)
    recall10 = _number(metrics['recall_at_10'], f'{where}.recall_at_10', unit=True)
    rr = _number(metrics['reciprocal_rank'], f'{where}.reciprocal_rank', unit=True)
    rank = metrics['first_relevant_rank']
    if rank is not None and (type(rank) is not int or not 1 <= rank <= candidate_k):
        raise ev.EvaluationError(f'{where}.first_relevant_rank must be null or an integer in 1..candidate_k')
    _close(rr, 0.0 if rank is None else 1.0 / rank, f'{where}.reciprocal_rank')
    if recall5 > recall10:
        raise ev.EvaluationError(f'{where} recall_at_5 exceeds recall_at_10')

    hits = strategy['top_hits']
    if not isinstance(hits, list) or len(hits) > 10:
        raise ev.EvaluationError(f'{where}.top_hits must contain at most 10 hits')
    rows = set()
    hit_pairs = []
    for i, hit in enumerate(hits):
        if not isinstance(hit, dict):
            raise ev.EvaluationError(f'{where}.top_hits[{i}] must be an object')
        ev._integer(hit.get('row'), f'{where}.top_hits[{i}].row', minimum=0)
        ev._string(hit.get('path'), f'{where}.top_hits[{i}].path')
        ev._string(hit.get('symbol'), f'{where}.top_hits[{i}].symbol')
        _signed_number(hit.get('score'), f'{where}.top_hits[{i}].score')
        row = hit['row']
        identity = (hit['path'], hit['symbol'])
        if row in rows:
            raise ev.EvaluationError(f'{where}.top_hits contains a duplicate row identity')
        rows.add(row)
        hit_pairs.append(identity)
    relevant = set(label_pairs)
    if not relevant:
        raise ev.EvaluationError(f'{where} has no native relevance labels')
    relevant_hit_pairs = [pair for pair in hit_pairs if pair in relevant]
    if len(relevant_hit_pairs) != len(set(relevant_hit_pairs)):
        raise ev.EvaluationError(f'{where}.top_hits repeats a relevant native label identity')
    expected5 = len(set(hit_pairs[:5]) & relevant) / len(relevant)
    expected10 = len(set(hit_pairs[:10]) & relevant) / len(relevant)
    _close(recall5, expected5, f'{where}.recall_at_5 vs top_hits')
    _close(recall10, expected10, f'{where}.recall_at_10 vs top_hits')
    first_display_rank = next((i + 1 for i, pair in enumerate(hit_pairs) if pair in relevant), None)
    if first_display_rank is not None and rank != first_display_rank:
        raise ev.EvaluationError(f'{where}.first_relevant_rank disagrees with top_hits')
    if first_display_rank is None and rank is not None and rank <= 10:
        raise ev.EvaluationError(f'{where}.first_relevant_rank claims a hit absent from top_hits')
    if rank is not None and rank > 10 and expected10 != 0:
        raise ev.EvaluationError(f'{where} rank beyond 10 conflicts with relevant top_hits')
    if rank is not None and rank > 10 and len(hits) < 10:
        raise ev.EvaluationError(f'{where} rank beyond 10 requires a complete displayed top ten')
    if rank is None and expected10 != 0:
        raise ev.EvaluationError(f'{where} null rank conflicts with relevant top_hits')

    latency = strategy['retrieval']
    _keys(latency, ('median_ms', 'min_ms', 'max_ms', 'samples'), f'{where}.retrieval')
    median = _number(latency['median_ms'], f'{where}.median_ms')
    minimum = _number(latency['min_ms'], f'{where}.min_ms')
    maximum = _number(latency['max_ms'], f'{where}.max_ms')
    ev._integer(latency['samples'], f'{where}.samples')
    if minimum > median or median > maximum:
        raise ev.EvaluationError(f'{where} retrieval min/median/max are inconsistent')
    _number(strategy['materialization_ms'], f'{where}.materialization_ms')
    return {'recall_at_5': recall5, 'recall_at_10': recall10,
            'reciprocal_rank': rr, 'first_relevant_rank': rank}


def _summary_values(summary, where):
    _keys(summary, ('queries', 'recall_at_5', 'recall_at_10', 'mrr'), where)
    ev._integer(summary['queries'], f'{where}.queries')
    return {'queries': summary['queries'],
            'recall_at_5': _number(summary['recall_at_5'], f'{where}.recall_at_5', unit=True),
            'recall_at_10': _number(summary['recall_at_10'], f'{where}.recall_at_10', unit=True),
            'mrr': _number(summary['mrr'], f'{where}.mrr', unit=True)}


def _check_summary(summary, rows, strategy_name, where):
    actual = _summary_values(summary, where)
    if actual['queries'] != len(rows):
        raise ev.EvaluationError(f'{where}.queries denominator mismatch')
    for source_key, summary_key in (('recall_at_5', 'recall_at_5'),
                                    ('recall_at_10', 'recall_at_10'),
                                    ('reciprocal_rank', 'mrr')):
        expected = sum(row[strategy_name][source_key] for row in rows) / len(rows)
        _close(actual[summary_key], expected, f'{where}.{summary_key}')


def _outcome(baseline, challenger):
    if baseline == 0 and challenger == 0:
        return 'shared_miss'
    if baseline == 0:
        return 'recovered'
    if challenger == 0:
        return 'lost'
    if challenger > baseline:
        return 'improved'
    if challenger < baseline:
        return 'regressed'
    return 'tied'


def _validate(raw):
    if not isinstance(raw, dict) or type(raw.get('format_version')) is not int or raw['format_version'] != 1:
        raise ev.EvaluationError('invalid native evaluation wrapper format_version')
    wrapper_keys = {'format_version', 'evaluation'}
    wrapper_optional = {'input_file_bytes', 'model_load_ms', 'read_ms'}
    if set(raw) - wrapper_keys - wrapper_optional or not wrapper_keys <= set(raw):
        raise ev.EvaluationError('native evaluation wrapper keys mismatch')
    if 'input_file_bytes' in raw:
        ev._integer(raw['input_file_bytes'], 'input_file_bytes', minimum=0)
    for key in ('model_load_ms', 'read_ms'):
        if key in raw:
            _number(raw[key], key)
    report = raw.get('evaluation')
    expected_report_keys = ('format_version', 'dataset', 'backend', 'options', 'rrf_rank_constant',
                            'embedding_total_ms', 'variants')
    if (not isinstance(report, dict) or not set(expected_report_keys) <= set(report)
            or set(report) - set(expected_report_keys) - {'embedding_execution'}):
        raise ev.EvaluationError('native evaluation keys mismatch')
    if type(report['format_version']) is not int or report['format_version'] != 2:
        raise ev.EvaluationError('native evaluation format_version must be 2')
    dataset = report['dataset']
    baseline.validate_queries(dataset)
    options = report['options']
    _keys(options, ('candidate_k', 'repeats', 'compare_fp16'), 'native evaluation options')
    ev._integer(options['candidate_k'], 'candidate_k', minimum=10)
    ev._integer(options['repeats'], 'repeats')
    if type(options['compare_fp16']) is not bool:
        raise ev.EvaluationError('compare_fp16 must be boolean')
    config = {key: options[key] for key in ('candidate_k', 'repeats', 'compare_fp16')}
    baseline._validate_native(raw, dataset, config)
    ev._string(report['backend'], 'backend')
    ev._integer(report['rrf_rank_constant'], 'rrf_rank_constant')
    _number(report['embedding_total_ms'], 'embedding_total_ms')
    if 'embedding_execution' in report:
        ev._string(report['embedding_execution'], 'embedding_execution')
    queries = dataset['queries']
    categories = list(dict.fromkeys(q['category'] for q in queries))
    category_queries = {category: [q for q in queries if q['category'] == category]
                        for category in categories}
    variants = report['variants']
    if not isinstance(variants, list):
        raise ev.EvaluationError('native variants must be an array')
    expected_precisions = ['fp32', 'fp16'] if options['compare_fp16'] else ['fp32']
    analyses = []
    for vi, variant in enumerate(variants):
        where = f'variants[{vi}]'
        _keys(variant, ('snapshot', 'baseline_conversion_ms', 'baseline_upload_ms', 'conversion_ms',
                        'upload_ms', 'text_index_ms', 'queries', 'overall', 'by_category'), where)
        snapshot = variant['snapshot']
        if not isinstance(snapshot, dict) or snapshot.get('precision') != expected_precisions[vi]:
            raise ev.EvaluationError(f'{where}.snapshot precision mismatch')
        for key in ('baseline_conversion_ms', 'baseline_upload_ms', 'conversion_ms', 'upload_ms', 'text_index_ms'):
            _number(variant[key], f'{where}.{key}')
        rows = variant['queries']
        if not isinstance(rows, list) or len(rows) != len(queries):
            raise ev.EvaluationError(f'{where}.queries coverage mismatch')
        per_query = []
        for qi, (query, result) in enumerate(zip(queries, rows)):
            qwhere = f'{where}.queries[{qi}]'
            expected_query_keys = {'id', 'category', 'embedding_ms', 'filter_ms', 'first_semantic_ms',
                                   'hybrid_filter_ms', *STRATEGIES}
            _keys(result, expected_query_keys, qwhere)
            if not isinstance(result, dict) or result.get('id') != query['id']:
                raise ev.EvaluationError(f'{qwhere} query ID/order mismatch')
            if result.get('category') != query['category']:
                raise ev.EvaluationError(f'{qwhere} category echo mismatch')
            ev._string(result['id'], f'{qwhere}.id')
            for key in ('embedding_ms', 'filter_ms', 'first_semantic_ms', 'hybrid_filter_ms'):
                _number(result.get(key), f'{qwhere}.{key}')
            labels = [(item['path'], item['symbol']) for item in query['relevant']]
            metrics = {}
            for strategy_name in STRATEGIES:
                if strategy_name not in result:
                    raise ev.EvaluationError(f'{qwhere} missing strategy {strategy_name}')
                metrics[strategy_name] = _metrics(result[strategy_name], labels, options['candidate_k'],
                                                   f'{qwhere}.{strategy_name}')
                if result[strategy_name]['retrieval']['samples'] != options['repeats']:
                    raise ev.EvaluationError(f'{qwhere}.{strategy_name} retrieval sample count mismatch')
            per_query.append({'id': query['id'], 'category': query['category'], **metrics})

        overall = variant['overall']
        _keys(overall, STRATEGIES, f'{where}.overall')
        for name in STRATEGIES:
            _check_summary(overall[name], per_query, name, f'{where}.overall.{name}')
        by_category = variant['by_category']
        _keys(by_category, categories, f'{where}.by_category')
        category_summaries = {}
        for category in categories:
            subset = [row for row in per_query if row['category'] == category]
            _keys(by_category[category], STRATEGIES, f'{where}.by_category.{category}')
            category_summaries[category] = {}
            for name in STRATEGIES:
                summary_where = f'{where}.by_category.{category}.{name}'
                _check_summary(by_category[category][name], subset, name, summary_where)
                category_summaries[category][name] = _summary_values(by_category[category][name], summary_where)

        paired = []
        for name in PAIRS:
            outcome_counts = {outcome: 0 for outcome in OUTCOMES}
            pairs = []
            for row in per_query:
                base_metric, compared = row['semantic'], row[name]
                outcome = _outcome(base_metric['recall_at_10'], compared['recall_at_10'])
                outcome_counts[outcome] += 1
                pairs.append({'query_id': row['id'], 'category': row['category'], 'outcome': outcome,
                              'semantic': base_metric, 'compared': compared,
                              'delta': {metric: compared[metric] - base_metric[metric]
                                        for metric in ('recall_at_5', 'recall_at_10', 'reciprocal_rank')}})
            paired.append({'baseline': 'semantic', 'strategy': name, 'outcome_metric': 'recall_at_10',
                           'outcome_counts': outcome_counts, 'queries': pairs})
        analyses.append({'precision': expected_precisions[vi], 'overall': {
                            name: _summary_values(overall[name], f'{where}.overall.{name}') for name in STRATEGIES},
                         'by_category': category_summaries, 'queries': per_query,
                         'paired_comparisons': paired})
    return {'format_version': 1, 'source': {'canonical_sha256': ev._digest(raw)},
            'dataset': {'name': dataset['name'], 'query_count': len(queries),
                        'query_ids': [q['id'] for q in queries],
                        'categories': {category: len(rows) for category, rows in category_queries.items()},
                        'label_unit': 'native (path, symbol)'},
            'options': config, 'variants': analyses,
            'interpretation_limits': [
                'This is an original symbol-label development-set observation, not independently reviewed validation quality.',
                'Warm retrieval repeats are repeated timings and are not independent quality observations.',
                'The report does not establish a causal diagnosis, equal-work speedup, or agent outcome.',
                'Recall@10 is checked against the exported top ten; no Top100 anchor recall is inferred from those hits.',
                'MRR beyond the exported top ten cannot be independently recomputed; reciprocal-rank/rank consistency is checked.',
                'Native path/symbol labels remain native labels and are not converted to source spans.',
                'Native latency is intentionally omitted from this quality diagnostic.'
            ]}


def _markdown(diagnostics):
    dataset_name = html.escape(diagnostics['dataset']['name'], quote=True).replace('`', '&#96;')
    dataset_name = dataset_name.replace('\n', ' ').replace('\r', ' ')
    lines = ['# Native symbol-label diagnostics', '',
             f"Dataset: `{dataset_name}` ({diagnostics['dataset']['query_count']} queries; native `(path, symbol)` labels).", '',
             '| Precision | Strategy | Queries | Recall@5 | Recall@10 | MRR |',
             '|---|---:|---:|---:|---:|---:|']
    for variant in diagnostics['variants']:
        for name in STRATEGIES:
            row = variant['overall'][name]
            lines.append(f"| {variant['precision']} | {name} | {row['queries']} | {row['recall_at_5']:.6f} | {row['recall_at_10']:.6f} | {row['mrr']:.6f} |")
    lines.extend(['', 'Paired outcomes compare semantic with lexical, symbol, or hybrid using per-query Recall@10. Full per-query deltas and first relevant ranks are in `diagnostics.json`.', '',
                  'This is a saved original symbol-label development observation. Warm repeats are not independent quality samples. It establishes no causal diagnosis, equal-work speedup, or agent outcome. Display hits do not establish Top100 anchor recall; no source-span conversion is performed. MRR beyond the displayed top ten is retained from the validated report and cannot be independently recomputed here. Native latency is omitted.', ''])
    return '\n'.join(lines)


def diagnose(raw, output, *, raw_bytes=None):
    """Validate a native wrapper and save a create-new diagnostics bundle."""
    try:
        canonical = ev._canonical(raw)
    except (TypeError, ValueError) as error:
        raise ev.EvaluationError(f'native report is not finite canonical JSON: {error}') from error
    if raw_bytes is not None:
        parsed = _parse_json(raw_bytes)
        if ev._canonical(parsed) != canonical:
            raise ev.EvaluationError('raw_bytes do not encode the supplied native report')
    result = _validate(raw)
    result['source']['raw_bytes_sha256'] = hashlib.sha256(raw_bytes if raw_bytes is not None else canonical).hexdigest()
    markdown = _markdown(result)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = {'input-report.json': canonical + b'\n',
             'diagnostics.json': ev._canonical(result) + b'\n',
             'README.md': markdown.encode('utf-8')}
    for name, content in files.items():
        with (output / name).open('xb') as handle:
            handle.write(content)
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    with (output / 'manifest.json').open('xb') as handle:
        handle.write(ev._canonical(manifest) + b'\n')
    return {'diagnostics': result, 'markdown': markdown, 'manifest': manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, help='saved native wrapper JSON report')
    parser.add_argument('--output', required=True, help='new output directory')
    args = parser.parse_args(argv)
    try:
        source = Path(args.report)
        raw_bytes = source.read_bytes()
        raw = _parse_json(raw_bytes)
        diagnose(raw, args.output, raw_bytes=raw_bytes)
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f'repository_native_diagnostics: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
