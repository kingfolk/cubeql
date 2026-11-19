"""Validate observed Hybrid stages and locate each labeled anchor's first loss."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path
import struct
import sys
import repository_eval as ev
import repository_native_diagnostics as native

SOURCES = ('semantic', 'lexical', 'symbol')
STAGES = ('not_in_filtered_branch_windows', 'suppressed_by_overlap',
          'truncated_after_deduplication', 'retained_below_top10', 'retained_top10')
CANDIDATE_KEYS = ('rank', 'row', 'path', 'symbol', 'start_line', 'end_line', 'score')


def _float32(value):
    native._signed_number(value, 'trace score')
    try:
        return struct.pack('<f', value)
    except (OverflowError, struct.error) as error:
        raise ev.EvaluationError('trace score is outside finite float32 range') from error


def _overlap(a, b):
    intersection = max(0, min(a['end_line'], b['end_line']) - max(a['start_line'], b['start_line']) + 1)
    union = a['end_line'] - a['start_line'] + 1 + b['end_line'] - b['start_line'] + 1 - intersection
    return a['path'] == b['path'] and intersection * 5 >= union * 4


def _array(value, limit, chunks, identities, label_rows, labels):
    if not isinstance(value, list) or len(value) > limit:
        raise ev.EvaluationError('trace array exceeds its declared stage budget')
    seen = set()
    for rank, hit in enumerate(value, 1):
        ev._object(hit, CANDIDATE_KEYS, where='trace candidate')
        for key, minimum in [('rank', 1), ('row', 0), ('start_line', 1), ('end_line', 1)]:
            ev._integer(hit[key], 'trace ' + key, minimum)
        if hit['rank'] != rank or hit['row'] >= chunks or hit['row'] in seen:
            raise ev.EvaluationError('trace has noncontiguous ranks, duplicate or out-of-snapshot rows')
        if not hit['start_line'] <= hit['end_line'] <= 2**32 - 1:
            raise ev.EvaluationError('trace line range is invalid')
        seen.add(hit['row'])
        ev._relpath(hit['path'], 'trace path')
        ev._string(hit['symbol'], 'trace symbol')
        _float32(hit['score'])
        identity = tuple(hit[key] for key in ('path', 'symbol', 'start_line', 'end_line'))
        if identities.setdefault(hit['row'], identity) != identity:
            raise ev.EvaluationError('trace snapshot row metadata changed between stages or queries')
        label = (hit['path'], hit['symbol'])
        if label in labels and label_rows.setdefault(label, hit['row']) != hit['row']:
            raise ev.EvaluationError('trace has an ambiguous labeled symbol')
    return value


def _trace(trace, query, case, options, chunks, identities, label_rows, labels):
    ev._object(trace, ('format_version', 'candidate_k', 'top_k', 'deduplicate', 'rrf_rank_constant',
                      'sources', 'fused', 'deduplicated', 'suppressions', 'final_rows'), where='hybrid trace')
    if type(trace['format_version']) is not int or trace['format_version'] != 1:
        raise ev.EvaluationError('unsupported hybrid trace version')
    for key in ('candidate_k', 'top_k', 'rrf_rank_constant'):
        ev._integer(trace[key], key)
    if (trace['candidate_k'] != options['candidate_k'] or trace['top_k'] != options['candidate_k']
            or trace['rrf_rank_constant'] != 60 or type(trace['deduplicate']) is not bool):
        raise ev.EvaluationError('trace budgets or fusion policy differ from native evaluation')
    sources = trace['sources']
    if not isinstance(sources, list) or len(sources) != 3:
        raise ev.EvaluationError('trace requires all three filtered source windows')
    arrays, scores, by_row = {}, {}, {}
    parse = lambda value, limit: _array(value, limit, chunks, identities, label_rows, labels)
    for name, source in zip(SOURCES, sources):
        ev._object(source, ('source', 'candidates'), where='trace source')
        if source['source'] != name:
            raise ev.EvaluationError('trace source order/coverage mismatch')
        arrays[name] = parse(source['candidates'], trace['candidate_k'])
        for hit in arrays[name]:
            by_row[hit['row']] = hit
            scores[hit['row']] = scores.get(hit['row'], 0.0) + 1.0 / (60 + hit['rank'])
    fused = parse(trace['fused'], 3 * trace['candidate_k'])
    expected_rows = sorted(scores, key=lambda row: (-scores[row], row))
    if [h['row'] for h in fused] != expected_rows:
        raise ev.EvaluationError('trace fusion is not the complete ordered RRF union')
    for hit in fused:
        if _float32(hit['score']) != _float32(scores[hit['row']]):
            raise ev.EvaluationError('trace RRF score mismatch')
    kept, suppressed = [], {}
    for hit in fused:
        witness = next((other for other in kept if _overlap(hit, other)), None) if trace['deduplicate'] else None
        if witness is None:
            kept.append(hit)
        else:
            suppressed[hit['row']] = witness['row']
    dedup = parse(trace['deduplicated'], len(fused))
    if [h['row'] for h in dedup] != [h['row'] for h in kept]:
        raise ev.EvaluationError('trace deduplication does not match greedy inclusive-range IoU policy')
    for actual, expected in zip(dedup, kept):
        if _float32(actual['score']) != _float32(expected['score']):
            raise ev.EvaluationError('trace changed a score during deduplication')
    if not isinstance(trace['suppressions'], list):
        raise ev.EvaluationError('trace suppressions must be an array')
    edges = {}
    for edge in trace['suppressions']:
        ev._object(edge, ('row', 'kept_row'), where='trace suppression')
        for key in ('row', 'kept_row'):
            ev._integer(edge[key], 'suppression ' + key, 0)
        if edge['row'] in edges:
            raise ev.EvaluationError('duplicate trace suppression edge')
        edges[edge['row']] = edge['kept_row']
    if edges != suppressed:
        raise ev.EvaluationError('trace suppression witnesses differ from actual greedy decisions')
    final = query['hybrid']['candidate_ranking']
    final_rows = [h['row'] for h in kept[:trace['top_k']]]
    if (not isinstance(trace['final_rows'], list)
            or any(type(row) is not int for row in trace['final_rows'])
            or trace['final_rows'] != final_rows or [h['row'] for h in final] != final_rows):
        raise ev.EvaluationError('trace final window is not the deduplicated prefix')
    for actual, expected in zip(final, kept):
        if (any(actual[key] != expected[key] for key in ('row', 'path', 'symbol'))
                or _float32(actual['score']) != _float32(expected['score'])):
            raise ev.EvaluationError('trace final metadata differs from exported ranking')
    for hit, expected in zip(query['hybrid']['top_hits'], kept):
        if any(hit[key] != expected[key] for key in ('start_line', 'end_line')):
            raise ev.EvaluationError('trace line range differs from displayed snapshot metadata')
    stage_ranks = [{h['row']: h['rank'] for h in array} for array in (fused, dedup, final)]
    source_labels = {name: {(h['path'], h['symbol']): h['rank'] for h in array} for name, array in arrays.items()}
    rows_for_labels = {(hit['path'], hit['symbol']): row for row, hit in by_row.items()}
    findings = []
    for label in case['relevant']:
        identity = (label['path'], label['symbol'])
        row = rows_for_labels.get(identity)
        fused_rank, dedup_rank, final_rank = [ranks.get(row) for ranks in stage_ranks]
        stage = ('not_in_filtered_branch_windows' if row is None else
                 'suppressed_by_overlap' if row in suppressed else
                 'truncated_after_deduplication' if final_rank is None else
                 'retained_below_top10' if final_rank > 10 else 'retained_top10')
        witness_row = suppressed.get(row)
        witness = None if witness_row is None else dict(row=witness_row,
            path=by_row[witness_row]['path'], symbol=by_row[witness_row]['symbol'],
            final_rank=stage_ranks[2].get(witness_row))
        findings.append(dict(label, stage=stage, branch_ranks={name: ranks.get(identity) for name, ranks in source_labels.items()},
            fused_rank=fused_rank, deduplicated_rank=dedup_rank, final_rank=final_rank, suppressed_by=witness))
    return {'id': case['id'], 'category': case['category'], 'anchors': findings}


def _summary(queries):
    anchors = [a for q in queries for a in q['anchors']]
    counts = Counter(a['stage'] for a in anchors)
    return {'query_count': len(queries), 'anchor_judgment_count': len(anchors),
        'anchor_stage_counts': {stage: counts[stage] for stage in STAGES},
        'queries_with_stage': {stage: sum(any(a['stage'] == stage for a in q['anchors']) for q in queries) for stage in STAGES}}


def analyze(raw):
    if (not isinstance(raw, dict) or not isinstance(raw.get('evaluation'), dict)
            or type(raw['evaluation'].get('format_version')) is not int or raw['evaluation']['format_version'] != 4):
        raise ev.EvaluationError('hybrid-stage diagnostics require native report version 4')
    projection = copy.deepcopy(raw)
    projection['evaluation']['format_version'] = 3
    projected_variants = projection['evaluation'].get('variants')
    if not isinstance(projected_variants, list):
        raise ev.EvaluationError('native variants must be an array')
    for variant in projected_variants:
        if not isinstance(variant, dict) or not isinstance(variant.get('queries'), list):
            raise ev.EvaluationError('invalid native variant/query list')
        for query in variant['queries']:
            if not isinstance(query, dict) or 'hybrid_trace' not in query:
                raise ev.EvaluationError('every query needs a complete hybrid trace')
            del query['hybrid_trace']
    quality = native._validate(projection)
    if raw['evaluation']['rrf_rank_constant'] != 60:
        raise ev.EvaluationError('trace report RRF constant must be 60')
    cases = raw['evaluation']['dataset']['queries']
    labels = {(a['path'], a['symbol']) for q in cases for a in q['relevant']}
    variants = []
    for variant, scored in zip(raw['evaluation']['variants'], quality['variants']):
        identities, label_rows = {}, {}
        queries = [_trace(q['hybrid_trace'], q, case, quality['options'], variant['snapshot']['chunks'],
                          identities, label_rows, labels) for case, q in zip(cases, variant['queries'])]
        for query in variant['queries']:
            for name in native.STRATEGIES:
                for hit in query[name]['candidate_ranking']:
                    if hit['row'] in identities and identities[hit['row']][:2] != (hit['path'], hit['symbol']):
                        raise ev.EvaluationError('trace row identity differs from native rankings')
                for hit in query[name]['top_hits']:
                    identity = tuple(hit[key] for key in ('path', 'symbol', 'start_line', 'end_line'))
                    if hit['row'] in identities and identities[hit['row']] != identity:
                        raise ev.EvaluationError('trace geometry differs from native displayed metadata')
        variants.append({'precision': scored['precision'], 'overall': _summary(queries),
            'by_category': {cat: _summary([q for q in queries if q['category'] == cat]) for cat in quality['dataset']['categories']},
            'native_overall': scored['overall'],
            'native_final_windows': {name: {'overall': detail['overall'], 'by_category': detail['by_category']}
                                     for name, detail in scored['candidate_window'].items()},
            'queries': queries})
    return {'format_version': 1, 'native_report_version': 4, 'source': {'canonical_sha256': ev._digest(raw)},
        'native_projection_canonical_sha256': quality['source']['canonical_sha256'],
        'dataset': quality['dataset'], 'options': quality['options'], 'variants': variants,
        'limits': ['Stage attribution is conditional on the exported filtered, bounded branch windows and fixed pipeline.',
            'An anchor absent from those branch windows is not proven absent from the repository or the unbounded branch ranking.',
            'Suppression records are geometric policy decisions, not judgments that the suppressed code is irrelevant.',
            'Counts of anchor judgments include repeated labels across queries. Queries-with-stage counts can overlap.',
            'Trace instrumentation can affect the native deduplication and last hybrid warm-repeat timings. No latency improvement or ranking-policy benefit is claimed.',
            'These positive-label development observations do not establish independently reviewed validation or coding-agent success.']}


def _markdown(result):
    lines = ['# Observed Hybrid stage diagnostics', '',
        '| Precision | Stage | Anchor judgments | Queries containing stage |', '| --- | --- | ---: | ---: |']
    for variant in result['variants']:
        for stage in STAGES:
            overall = variant['overall']
            lines.append(f'| {variant["precision"]} | {stage} | {overall["anchor_stage_counts"][stage]} | {overall["queries_with_stage"][stage]} |')
    lines.extend(['', 'Branch windows are filtered and bounded; absence here is not an unbounded retrieval or repository-coverage conclusion.', '',
                  'Suppression uses observed same-file inclusive-range IoU decisions. Query counts can overlap across stages.', ''])
    lines.extend('- ' + line for line in result['limits'])
    return '\n'.join(lines) + '\n'


def diagnose(raw, output, *, raw_bytes=None):
    try:
        canonical = ev._canonical(raw)
    except (TypeError, ValueError) as error:
        raise ev.EvaluationError(f'report is not finite canonical JSON: {error}') from error
    if raw_bytes is not None and ev._canonical(native._parse_json(raw_bytes)) != canonical:
        raise ev.EvaluationError('raw bytes do not encode the supplied report')
    result = analyze(raw)
    result['source']['raw_bytes_sha256'] = hashlib.sha256(raw_bytes if raw_bytes is not None else canonical).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    content = {'input-report.json': canonical + b'\n', 'diagnostics.json': ev._canonical(result) + b'\n',
               'README.md': _markdown(result).encode()}
    for name, data in content.items():
        with (output / name).open('xb') as stream:
            stream.write(data)
    ev._write_new(output / 'manifest.json', {name: hashlib.sha256(data).hexdigest() for name, data in content.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        data = Path(args.report).read_bytes()
        diagnose(native._parse_json(data), args.output, raw_bytes=data)
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f'repository_hybrid_trace: {error}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
