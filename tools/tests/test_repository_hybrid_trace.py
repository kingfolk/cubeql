"""Contract tests for observed Hybrid stage attribution on native v4 reports."""
from __future__ import annotations

import copy
import importlib.util
import math
import os
import struct
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
_default_repo = HERE.parents[1] if (HERE.parents[1] / 'tools').is_dir() else HERE.parent
REPO = Path(os.environ.get('CUBEQL_REPO', str(_default_repo)))
TOOLS = REPO / 'tools'
TESTS = TOOLS / 'tests'
for candidate in (HERE, TOOLS, TESTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import repository_eval as ev

_native_path = HERE / 'repository_native_diagnostics.py'
if not _native_path.exists():
    _native_path = TOOLS / 'repository_native_diagnostics.py'
_native_spec = importlib.util.spec_from_file_location('_hybrid_native_diagnostics', _native_path)
if _native_spec is None or _native_spec.loader is None:
    raise RuntimeError(f'cannot load native consumer at {_native_path}')
native = importlib.util.module_from_spec(_native_spec)
_native_spec.loader.exec_module(native)

_consumer_path = HERE / 'repository_hybrid_trace.py'
if not _consumer_path.exists():
    _consumer_path = TOOLS / 'repository_hybrid_trace.py'
_consumer_spec = importlib.util.spec_from_file_location('_hybrid_trace_consumer', _consumer_path)
if _consumer_spec is None or _consumer_spec.loader is None:
    raise RuntimeError(f'cannot load hybrid trace consumer at {_consumer_path}')
trace = importlib.util.module_from_spec(_consumer_spec)
_consumer_spec.loader.exec_module(trace)


def _old_fixture():
    fixture_path = TESTS / 'test_repository_native_diagnostics.py'
    spec = importlib.util.spec_from_file_location('_hybrid_old_native_fixture', fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load fixture at {fixture_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativeDiagnosticsTests()


def _legacy_to_v3(test, compare_fp16=False):
    fixture = _old_fixture()
    fixture.setUp()
    test.addCleanup(fixture.doCleanups)
    raw = fixture.make_report(compare_fp16=compare_fp16)
    raw['evaluation']['format_version'] = 3
    row_by_identity = {}

    def get_row(hit):
        key = (hit['path'], hit['symbol'])
        if key not in row_by_identity:
            row_by_identity[key] = len(row_by_identity)
        return row_by_identity[key]

    for variant in raw['evaluation']['variants']:
        for case, query in zip(raw['evaluation']['dataset']['queries'], variant['queries']):
            for name in native.STRATEGIES:
                strategy = query[name]
                pairs = [(hit['path'], hit['symbol']) for hit in strategy['top_hits']]
                first = strategy['metrics']['first_relevant_rank']
                target = 10 if first is None or first <= 10 else first
                ranking = []
                for i in range(target):
                    if i < len(pairs):
                        path, symbol = pairs[i]
                    elif first is not None and first > 10 and i + 1 == first:
                        label = next((x for x in case['relevant'] if (x['path'], x['symbol']) not in pairs), None)
                        path, symbol = ((label['path'], label['symbol']) if label else
                                        (f'decoy/{case["id"]}-{name}-{i}.rs', f'{case["id"]}-{name}-{i}'))
                    else:
                        path, symbol = f'decoy/{case["id"]}-{name}-{i}.rs', f'{case["id"]}-{name}-{i}'
                    row = get_row({'path': path, 'symbol': symbol})
                    ranking.append({'rank': i + 1, 'row': row, 'path': path, 'symbol': symbol,
                                    'score': 0.8 if row else 0.2})
                strategy['candidate_ranking'] = ranking
                strategy['top_hits'] = [
                    {'row': x['row'], 'path': x['path'], 'symbol': x['symbol'], 'score': x['score'],
                     'language': 'rust', 'kind': 'function', 'start_line': 1, 'end_line': 2,
                     'provenance': [], 'suppressed_overlaps': []} for x in ranking[:10]]
        variant['snapshot']['chunks'] = len(row_by_identity)
    return raw


def _candidate(row, rank, *, path=None, symbol=None, start=1, end=2, score=None):
    return {'rank': rank, 'row': row, 'path': path or f'fixture/r{row}.rs',
            'symbol': symbol or f'symbol_{row}', 'start_line': start, 'end_line': end,
            'score': (0.9 - row / 1000) if score is None else score}


def _trace_from_sources(sources, *, k=20, deduplicate=True):
    scores = {}
    metadata = {}
    for source in sources:
        for hit in source['candidates']:
            row = hit['row']
            scores[row] = scores.get(row, 0.0) + 1.0 / (60 + hit['rank'])
            metadata[row] = hit
    fused_rows = sorted(scores, key=lambda row: (-scores[row], row))
    fused = []
    for i, row in enumerate(fused_rows):
        hit = metadata[row]
        fused.append({'rank': i + 1, 'row': row, 'path': hit['path'], 'symbol': hit['symbol'],
                      'start_line': hit['start_line'], 'end_line': hit['end_line'], 'score': scores[row]})
    kept = []
    suppressions = []
    for hit in fused:
        # Fixture has one declared overlap pair: kept_range and the label
        # "two" have identical, hand-authored inclusive ranges in src/b.rs.
        witness = next((other for other in kept if deduplicate and
                        ((hit['symbol'], other['symbol']) in {('two', 'kept_range'),
                                                              ('tail_suppressed', 'tail_kept')})
                        and hit['path'] == other['path']
                        and (hit['start_line'], hit['end_line']) == (other['start_line'], other['end_line'])), None)
        if witness is None:
            kept.append(hit)
        else:
            suppressions.append({'row': hit['row'], 'kept_row': witness['row']})
    dedup = [dict(hit, rank=i + 1) for i, hit in enumerate(kept)]
    final = dedup[:k]
    return {'format_version': 1, 'candidate_k': k, 'top_k': k, 'deduplicate': deduplicate,
            'rrf_rank_constant': 60, 'sources': sources, 'fused': fused,
            'deduplicated': dedup, 'suppressions': suppressions,
            'final_rows': [hit['row'] for hit in final]}, final


def _empty_trace():
    sources = [{'source': name, 'candidates': []} for name in trace.SOURCES]
    return _trace_from_sources(sources)


def _set_hybrid_query(raw, query_index, case, tr, final):
    query = raw['evaluation']['variants'][0]['queries'][query_index]
    query['hybrid_trace'] = tr
    traced = {hit['row']: hit for source in tr['sources'] for hit in source['candidates']}
    for name in native.STRATEGIES:
        if name == 'hybrid':
            continue
        for hit in query[name]['top_hits']:
            expected = traced.get(hit['row'])
            if expected and (hit['path'], hit['symbol']) == (expected['path'], expected['symbol']):
                hit['start_line'], hit['end_line'] = expected['start_line'], expected['end_line']
    strategy = query['hybrid']
    strategy['candidate_ranking'] = [
        {'rank': i + 1, 'row': hit['row'], 'path': hit['path'], 'symbol': hit['symbol'], 'score': hit['score']}
        for i, hit in enumerate(final)]
    strategy['top_hits'] = [
        {'row': x['row'], 'path': x['path'], 'symbol': x['symbol'], 'score': x['score'],
         'language': 'rust', 'kind': 'function', 'start_line': x['start_line'], 'end_line': x['end_line'],
         'provenance': [], 'suppressed_overlaps': []} for x in final[:10]]
    labels = {(x['path'], x['symbol']) for x in case['relevant']}
    ranks = [i + 1 for i, x in enumerate(final) if (x['path'], x['symbol']) in labels]
    rank = min(ranks, default=None)
    strategy['metrics'] = {'recall_at_5': len({(x['path'], x['symbol']) for x in final[:5]} & labels) / len(labels),
                           'recall_at_10': len({(x['path'], x['symbol']) for x in final[:10]} & labels) / len(labels),
                           'reciprocal_rank': 0 if rank is None else 1 / rank,
                           'first_relevant_rank': rank}


def _summary(rows, name):
    metrics = [row[name]['metrics'] for row in rows]
    return {'queries': len(rows), 'recall_at_5': sum(x['recall_at_5'] for x in metrics) / len(rows),
            'recall_at_10': sum(x['recall_at_10'] for x in metrics) / len(rows),
            'mrr': sum(x['reciprocal_rank'] for x in metrics) / len(rows)}


def _refresh_hybrid_summaries(raw):
    variant = raw['evaluation']['variants'][0]
    variant['overall']['hybrid'] = _summary(variant['queries'], 'hybrid')
    for category in variant['by_category']:
        rows = [q for q in variant['queries'] if q['category'] == category]
        variant['by_category'][category]['hybrid'] = _summary(rows, 'hybrid')


def _report(test):
    raw = _legacy_to_v3(test)
    cases = raw['evaluation']['dataset']['queries']
    label_rows = {}
    for variant in raw['evaluation']['variants']:
        for query in variant['queries']:
            for name in native.STRATEGIES:
                for item in query[name]['candidate_ranking']:
                    key = (item['path'], item['symbol'])
                    if key in {(a['path'], a['symbol']) for case in cases for a in case['relevant']}:
                        label_rows.setdefault(key, item['row'])
    for variant in raw['evaluation']['variants']:
        variant['snapshot']['chunks'] = 2000
    q1 = cases[0]['relevant'][0]
    q1_second = cases[0]['relevant'][1]
    def branch(name, numbers, special=None):
        hits = []
        for i, row in enumerate(numbers, 1):
            if special and i == special[0]:
                hits.append(_candidate(special[1], i, path=special[2]['path'], symbol=special[2]['symbol'],
                                       start=20, end=22))
            elif row == 1000:
                hits.append(_candidate(label_rows[(q1['path'], q1['symbol'])], i,
                                       path=q1['path'], symbol=q1['symbol']))
            elif row == 1004:
                hits.append(_candidate(1600, i, path=q1_second['path'], symbol='kept_range', start=20, end=22))
            else:
                hits.append(_candidate(row, i))
        return {'source': name, 'candidates': hits}
    sources = [branch('semantic', range(1000, 1020)),
               branch('lexical', [1000, *range(1002, 1021)],
                      (11, label_rows[(q1_second['path'], q1_second['symbol'])], q1_second)),
               branch('symbol', [1000, *range(1004, 1023)])]
    tr, final = _trace_from_sources(sources)
    _set_hybrid_query(raw, 0, cases[0], tr, final)
    # q2's label lands at deduplicated rank 12; q3's labels miss all three
    # windows; q4's label lands after the final 20-row prefix.
    for qi in range(1, 4):
        anchor = cases[qi]['relevant'][0]
        if qi == 1:
            candidates = [_candidate(1400 + i, i + 1) for i in range(11)]
            candidates.append(_candidate(label_rows[(anchor['path'], anchor['symbol'])], 12,
                                         path=anchor['path'], symbol=anchor['symbol']))
            src = [{'source': n, 'candidates': [dict(x, rank=i + 1) for i, x in enumerate(candidates)]}
                   for n in trace.SOURCES]
        elif qi == 2:
            src = [{'source': n, 'candidates': []} for n in trace.SOURCES]
        else:
            tail = []
            for i, row in enumerate(range(1504, 1524)):
                if i == 17:
                    tail.append(_candidate(label_rows[(anchor['path'], anchor['symbol'])], i + 1,
                                           path=anchor['path'], symbol=anchor['symbol']))
                elif i == 18:
                    tail.append(_candidate(1700, i + 1, path='fixture/tail.rs', symbol='tail_kept', start=30, end=31))
                elif i == 19:
                    tail.append(_candidate(1701, i + 1, path='fixture/tail.rs', symbol='tail_suppressed', start=30, end=31))
                else:
                    tail.append(_candidate(row, i + 1))
            src = [{'source': 'semantic', 'candidates': [_candidate(row, i + 1)
                   for i, row in enumerate(range(1500, 1520))]},
                   {'source': 'lexical', 'candidates': [_candidate(row, i + 1)
                   for i, row in enumerate(range(1502, 1522))]},
                   {'source': 'symbol', 'candidates': tail}]
        tr, final = _trace_from_sources(src)
        _set_hybrid_query(raw, qi, cases[qi], tr, final)
    _refresh_hybrid_summaries(raw)
    raw['evaluation']['format_version'] = 4
    raw['evaluation']['rrf_rank_constant'] = 60
    return raw


class HybridTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = _report(self)

    def test_hand_computed_trace_and_native_v3_projection(self):
        result = trace.analyze(self.raw)
        self.assertEqual(result['format_version'], 1)
        self.assertEqual(result['native_report_version'], 4)
        self.assertEqual(result['variants'][0]['precision'], 'fp32')
        anchors = result['variants'][0]['queries'][0]['anchors']
        q1_trace = self.raw['evaluation']['variants'][0]['queries'][0]['hybrid_trace']
        # Row 1 is rank 1 in each source, so its score is exactly 3/61.
        # Row 1600 has source ranks 5, 4, and 2; these fractions put it second.
        self.assertEqual([x['row'] for x in q1_trace['fused'][:3]], [1, 1600, 1005])
        self.assertTrue(math.isclose(q1_trace['fused'][0]['score'], 3 / 61, rel_tol=0, abs_tol=1e-15))
        self.assertTrue(math.isclose(q1_trace['fused'][1]['score'], 1 / 65 + 1 / 64 + 1 / 62,
                                     rel_tol=0, abs_tol=1e-15))
        self.assertEqual(q1_trace['suppressions'], [{'row': 10, 'kept_row': 1600}])
        stages = {a['stage'] for q in result['variants'][0]['queries'] for a in q['anchors']}
        self.assertEqual(stages, set(trace.STAGES))
        self.assertEqual([a['stage'] for a in anchors], ['retained_top10', 'suppressed_by_overlap'])
        q2 = result['variants'][0]['queries'][1]['anchors'][0]
        q4 = result['variants'][0]['queries'][3]['anchors'][0]
        self.assertEqual(q2['stage'], 'retained_below_top10')
        self.assertEqual(q4['stage'], 'truncated_after_deduplication')
        q4_trace = self.raw['evaluation']['variants'][0]['queries'][3]['hybrid_trace']
        self.assertEqual(q4_trace['suppressions'], [{'row': 1701, 'kept_row': 1700}])
        self.assertGreater(q4_trace['deduplicated'].index(next(x for x in q4_trace['deduplicated']
                                                                if x['row'] == 1700)), 19)
        self.assertNotIn(1700, q4_trace['final_rows'])
        self.assertEqual(result['variants'][0]['native_overall']['hybrid']['queries'], 4)
        self.assertEqual(result['variants'][0]['by_category']['identifier']['query_count'], 2)

    def test_f64_order_is_preserved_when_exported_f32_scores_tie(self):
        raw = copy.deepcopy(self.raw)
        lower, higher = 1801, 1802
        positions = [(higher, lower), (lower, higher), (lower, higher)]
        ranks = [(1, 7), (1, 2), (2, 7)]
        sources = []
        for source_index, (name, pair, slots) in enumerate(zip(trace.SOURCES, positions, ranks)):
            records = []
            for rank in range(1, 8):
                row = pair[0] if rank == slots[0] else pair[1] if rank == slots[1] else 1810 + source_index * 10 + rank
                records.append(_candidate(row, rank))
            sources.append({'source': name, 'candidates': records})
        tr, final = _trace_from_sources(sources)
        self.assertEqual([h['row'] for h in tr['fused'][:2]], [higher, lower])
        # The higher row wins by a float64 summation difference. Both wire scores
        # become the same float32; sorting those rounded scores by row is wrong.
        for array in (tr['fused'], tr['deduplicated']):
            for hit in array:
                hit['score'] = struct.unpack('<f', struct.pack('<f', hit['score']))[0]
        self.assertEqual(tr['fused'][0]['score'], tr['fused'][1]['score'])
        _set_hybrid_query(raw, 2, raw['evaluation']['dataset']['queries'][2], tr, final)
        _refresh_hybrid_summaries(raw)
        trace.analyze(raw)
        # Keep every prefix/rank/metric internally plausible after the bad sort.
        # Only reconstruction from the unrounded RRF fractions detects it.
        for key in ('fused', 'deduplicated'):
            tr[key][0], tr[key][1] = tr[key][1], tr[key][0]
            for rank, hit in enumerate(tr[key], 1):
                hit['rank'] = rank
        tr['final_rows'] = [h['row'] for h in tr['deduplicated'][:20]]
        _set_hybrid_query(raw, 2, raw['evaluation']['dataset']['queries'][2], tr, tr['deduplicated'][:20])
        _refresh_hybrid_summaries(raw)
        with self.assertRaises(ev.EvaluationError):
            trace.analyze(raw)

    def test_all_precision_variants_require_complete_traces(self):
        raw = copy.deepcopy(self.raw)
        raw['evaluation']['options']['compare_fp16'] = True
        second = copy.deepcopy(raw['evaluation']['variants'][0])
        second['snapshot']['precision'] = 'fp16'
        raw['evaluation']['variants'].append(second)
        result = trace.analyze(raw)
        self.assertEqual([v['precision'] for v in result['variants']], ['fp32', 'fp16'])
        del raw['evaluation']['variants'][1]['queries'][0]['hybrid_trace']
        with self.assertRaises(ev.EvaluationError):
            trace.analyze(raw)

    def test_rejects_fused_order_score_budget_and_missing_source(self):
        mutations = [
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['fused'].reverse(),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['fused'][0].__setitem__('score', .5),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace'].__setitem__('candidate_k', 19),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['sources'].pop(),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['sources'][1].__setitem__('source', 'semantic'),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(self.raw)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ev.EvaluationError):
                trace.analyze(bad)

    def test_rejects_forged_suppression_geometry_metadata_and_ranks(self):
        mutations = [
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['suppressions'][0].__setitem__('kept_row', 1),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['deduplicated'].reverse(),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['final_rows'].reverse(),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['sources'][0]['candidates'][0].__setitem__('end_line', 1),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['sources'][0]['candidates'][0].__setitem__('row', True),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['hybrid_trace']['sources'][0]['candidates'][0].__setitem__('score', float('nan')),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(self.raw)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ev.EvaluationError):
                trace.analyze(bad)

    def test_valid_but_subthreshold_iou_cannot_support_a_suppression_edge(self):
        bad = copy.deepcopy(self.raw)
        query = bad['evaluation']['variants'][0]['queries'][0]
        tr = query['hybrid_trace']
        for stage in ('fused', 'deduplicated'):
            for hit in tr[stage]:
                if hit['row'] == 1600:
                    hit['start_line'], hit['end_line'] = 20, 20
        for source in tr['sources']:
            for hit in source['candidates']:
                if hit['row'] == 1600:
                    hit['start_line'], hit['end_line'] = 20, 20
        for hit in query['hybrid']['top_hits']:
            if hit['row'] == 1600:
                hit['start_line'], hit['end_line'] = 20, 20
        # The 1/3 inclusive-range IoU is valid metadata but below the 0.8 policy.
        with self.assertRaises(ev.EvaluationError):
            trace.analyze(bad)

    def test_requires_complete_query_traces_and_consistent_raw_bytes(self):
        bad = copy.deepcopy(self.raw)
        del bad['evaluation']['variants'][0]['queries'][1]['hybrid_trace']
        with self.assertRaises(ev.EvaluationError):
            trace.analyze(bad)
        with self.assertRaises(ev.EvaluationError):
            trace.diagnose(self.raw, Path(self.tmp.name) / 'mismatch', raw_bytes=b'{}')
        self.assertFalse((Path(self.tmp.name) / 'mismatch').exists())

    def test_duplicate_json_and_create_new_bundle_contract(self):
        with self.assertRaises(ev.EvaluationError):
            native._parse_json(b'{"x":1,"x":2}')
        duplicate_input = Path(self.tmp.name) / 'duplicate.json'
        duplicate_input.write_bytes(b'{"format_version":1,"format_version":1}')
        duplicate_output = Path(self.tmp.name) / 'duplicate-output'
        self.assertEqual(trace.main(['--report', str(duplicate_input), '--output', str(duplicate_output)]), 2)
        self.assertFalse(duplicate_output.exists())
        output = Path(self.tmp.name) / 'bundle'
        raw_bytes = ev._canonical(self.raw)
        result = trace.diagnose(self.raw, output, raw_bytes=raw_bytes)
        self.assertTrue((output / 'diagnostics.json').exists())
        with self.assertRaises(FileExistsError):
            trace.diagnose(self.raw, output)
        self.assertTrue((output / 'diagnostics.json').exists())
        self.assertEqual(result['source']['raw_bytes_sha256'], __import__('hashlib').sha256(raw_bytes).hexdigest())


if __name__ == '__main__':
    unittest.main()
