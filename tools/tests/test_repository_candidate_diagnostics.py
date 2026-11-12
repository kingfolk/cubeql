"""Bounded contract tests for the staged v3 native candidate-window consumer.

This file is also intended to live beside the repository's existing native
diagnostics tests at tools/tests/.  During staging, set CUBEQL_REPO to the
actual checkout and put the staged consumer before tools/ on PYTHONPATH.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('CUBEQL_REPO', str(HERE.parents[1])))
TOOLS = REPO / 'tools'
for candidate in (HERE, TOOLS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import repository_eval as ev
_consumer_path = HERE / 'repository_native_diagnostics.py'
if not _consumer_path.exists():
    _consumer_path = TOOLS / 'repository_native_diagnostics.py'
_consumer_spec = importlib.util.spec_from_file_location('_candidate_native_diagnostics', _consumer_path)
if _consumer_spec is None or _consumer_spec.loader is None:
    raise RuntimeError(f'cannot load candidate consumer at {_consumer_path}')
diagnostics = importlib.util.module_from_spec(_consumer_spec)
_consumer_spec.loader.exec_module(diagnostics)


def _load_old_fixture():
    fixture_path = HERE / 'test_repository_native_diagnostics.py'
    if not fixture_path.exists():
        fixture_path = TOOLS / 'tests' / 'test_repository_native_diagnostics.py'
    spec = importlib.util.spec_from_file_location('_old_native_diagnostics_fixture', fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load existing fixture at {fixture_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_old_fixture = _load_old_fixture()


def _fixture_report(test_case, compare_fp16=False):
    """Reuse the repository fixture, then promote it to a complete v3 export."""
    fixture = _old_fixture.NativeDiagnosticsTests()
    fixture.setUp()
    test_case.addCleanup(fixture.doCleanups)
    raw = fixture.make_report(compare_fp16=compare_fp16)
    raw['evaluation']['format_version'] = 3

    # The older fixture assigned row numbers per query. V3 requires one stable
    # row-to-(path,symbol) mapping across the entire snapshot, so normalize all
    # display hits and exported rankings to deterministic global row ids.
    rows_by_identity = {}

    def row_for(hit):
        identity = (hit['path'], hit['symbol'])
        if identity not in rows_by_identity:
            rows_by_identity[identity] = len(rows_by_identity)
        return rows_by_identity[identity]

    for variant in raw['evaluation']['variants']:
        for case, query in zip(raw['evaluation']['dataset']['queries'], variant['queries']):
            labels = {(label['path'], label['symbol']) for label in case['relevant']}
            for name in diagnostics.STRATEGIES:
                strategy = query[name]
                hits = strategy['top_hits']
                for hit in hits:
                    hit['row'] = row_for(hit)
                ranking = [dict(rank=i + 1, row=hit['row'], path=hit['path'],
                                symbol=hit['symbol'], score=hit['score'])
                           for i, hit in enumerate(hits)]
                first = strategy['metrics']['first_relevant_rank']
                target_length = 10 if first is None or first <= 10 else first
                # Preserve a complete display prefix even for short legacy hit
                # arrays. Add unique decoys, and put a beyond-ten anchor at its
                # reported full rank when the fixture supplies one.
                while len(ranking) < target_length:
                    i = len(ranking)
                    hit = {'path': f'v3-decoy/{case["id"]}-{name}-{i}.rs',
                           'symbol': f'{case["id"]}-{name}-decoy-{i}',
                           'score': 0.25}
                    if first is not None and first > 10 and i + 1 == first:
                        label = next((item for item in case['relevant']
                                      if (item['path'], item['symbol']) not in
                                      {(entry['path'], entry['symbol']) for entry in ranking}), None)
                        if label is not None:
                            hit['path'], hit['symbol'] = label['path'], label['symbol']
                    hit['row'] = row_for(hit)
                    ranking.append({'rank': i + 1, **hit})
                # At ranks <= 10 the legacy fixture already carries all hits;
                # its labels and metric receipt are checked by the v2 validator.
                strategy['candidate_ranking'] = ranking
                strategy['top_hits'] = [
                    {'row': candidate['row'], 'path': candidate['path'],
                     'symbol': candidate['symbol'], 'score': candidate['score'],
                     'language': 'rust', 'kind': 'function', 'start_line': 1,
                     'end_line': 2, 'provenance': [], 'suppressed_overlaps': []}
                    for candidate in ranking[:10]
                ]
        # Update snapshot bounds after all variants are normalized below.
    chunks = max(1, len(rows_by_identity))
    for variant in raw['evaluation']['variants']:
        variant['snapshot']['chunks'] = chunks
        # A candidate beyond the displayed prefix must still agree with the
        # report's full-rank receipt; the source fixture already records it.
    return raw


def _recompute_display_receipts(raw):
    """Refresh legacy metrics/summaries after a deliberately crafted ranking."""
    dataset = raw['evaluation']['dataset']
    for variant in raw['evaluation']['variants']:
        for case, query in zip(dataset['queries'], variant['queries']):
            labels = {(x['path'], x['symbol']) for x in case['relevant']}
            for name in diagnostics.STRATEGIES:
                strategy = query[name]
                ranking = strategy['candidate_ranking']
                strategy['top_hits'] = [
                    {'row': x['row'], 'path': x['path'], 'symbol': x['symbol'], 'score': x['score'],
                     'language': 'rust', 'kind': 'function', 'start_line': 1, 'end_line': 2,
                     'provenance': [], 'suppressed_overlaps': []} for x in ranking[:10]
                ]
                matching = [i + 1 for i, x in enumerate(ranking)
                            if (x['path'], x['symbol']) in labels]
                rank = min(matching, default=None)
                strategy['metrics'] = {
                    'recall_at_5': len({(x['path'], x['symbol']) for x in ranking[:5]} & labels) / len(labels),
                    'recall_at_10': len({(x['path'], x['symbol']) for x in ranking[:10]} & labels) / len(labels),
                    'reciprocal_rank': 0.0 if rank is None else 1.0 / rank,
                    'first_relevant_rank': rank,
                }
        names = diagnostics.STRATEGIES
        variant['overall'] = {name: _legacy_summary(variant['queries'], name) for name in names}
        variant['by_category'] = {}
        for category in dict.fromkeys(x['category'] for x in dataset['queries']):
            subset = [q for q in variant['queries'] if q['category'] == category]
            variant['by_category'][category] = {name: _legacy_summary(subset, name) for name in names}


def _legacy_summary(queries, name):
    metrics = [row[name]['metrics'] for row in queries]
    return {'queries': len(metrics),
            'recall_at_5': sum(x['recall_at_5'] for x in metrics) / len(metrics),
            'recall_at_10': sum(x['recall_at_10'] for x in metrics) / len(metrics),
            'mrr': sum(x['reciprocal_rank'] for x in metrics) / len(metrics)}


class CandidateWindowDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_v2_json_and_markdown_are_byte_for_byte_unchanged(self):
        # Frozen from the unmodified tools/repository_native_diagnostics.py
        # consumer and its original v2 fixture before the candidate consumer
        # is copied over it. These remain meaningful after that copy.
        golden_sha256 = {
            'diagnostics.json': '4dea5531f86d997529c6e2fe4cb22834ab8941339f7851cb53e9d7cccbb85791',
            'README.md': '09689cc33b2d39f4ad497fc801236c4eecb7241b9cf60c9ea72626e38a2b1c62',
        }
        fixture = _old_fixture.NativeDiagnosticsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        raw = copy.deepcopy(fixture.raw)
        output = self.base / 'v2-output'
        diagnostics.diagnose(raw, output)
        for filename, expected in golden_sha256.items():
            self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), expected)

    def test_v3_recomputes_full_window_metrics_and_requires_all_coverage(self):
        raw = _fixture_report(self)
        # Empty and short final windows are legal; displayed hits must simply
        # match the available ranking prefix.
        semantic_empty = raw['evaluation']['variants'][0]['queries'][3]['semantic']
        semantic_empty['candidate_ranking'] = []
        semantic_empty['top_hits'] = []
        semantic_short = raw['evaluation']['variants'][0]['queries'][1]['semantic']
        semantic_short['candidate_ranking'] = semantic_short['candidate_ranking'][:1]
        first = semantic_short['candidate_ranking'][0]
        semantic_short['top_hits'] = [{
            'row': first['row'], 'path': first['path'], 'symbol': first['symbol'], 'score': first['score'],
            'language': 'rust', 'kind': 'function', 'start_line': 1, 'end_line': 2,
            'provenance': [], 'suppressed_overlaps': []}]
        result = diagnostics._validate(raw)
        self.assertEqual(result['format_version'], 2)
        self.assertEqual(result['candidate_export']['native_report_version'], 3)
        self.assertEqual(set(result['variants'][0]['candidate_window']), set(diagnostics.STRATEGIES))
        q4 = result['variants'][0]['candidate_window']['experimental_rrf']['queries'][3]
        self.assertEqual(q4['anchor_ranks'][0]['rank'], 12)
        self.assertAlmostEqual(q4['reciprocal_rank'], 1 / 12)
        self.assertEqual(result['variants'][0]['candidate_window']['experimental_rrf']['overall']['query_count'], 4)
        self.assertEqual(result['variants'][0]['candidate_window']['experimental_rrf']['overall']['mrr_at_window'], 13 / 48)
        self.assertEqual(result['variants'][0]['candidate_window']['semantic']['queries'][3]['exported_count'], 0)
        self.assertEqual(result['variants'][0]['candidate_window']['semantic']['queries'][1]['exported_count'], 1)
        self.assertEqual(result['variants'][0]['candidate_window']['semantic']['queries'][3]['failure_class'],
                         'missing_from_exported_window')

    def test_missing_takes_precedence_while_both_failure_lists_are_retained(self):
        raw = _fixture_report(self)
        case = raw['evaluation']['dataset']['queries'][0]
        query = raw['evaluation']['variants'][0]['queries'][0]
        strategy = query['semantic']
        # q1 has two labels. Keep one at rank 12 and omit the other, leaving a
        # complete ten-row display prefix and an internally consistent v2 receipt.
        retained = case['relevant'][0]
        row_lookup = {}
        max_row = -1
        for variant in raw['evaluation']['variants']:
            for query_row in variant['queries']:
                for name in diagnostics.STRATEGIES:
                    for candidate in query_row[name]['candidate_ranking']:
                        row_lookup[(candidate['path'], candidate['symbol'])] = candidate['row']
                        max_row = max(max_row, candidate['row'])
        ranking = []
        for i in range(10):
            ranking.append({'rank': i + 1, 'row': max_row + i + 1,
                            'path': f'mixed-decoy/{i}.rs', 'symbol': f'mixed-decoy-{i}', 'score': 0.2})
        ranking.append({'rank': 11, 'row': max_row + 11, 'path': 'mixed-decoy/11.rs',
                        'symbol': 'mixed-decoy-11', 'score': 0.2})
        ranking.append({'rank': 12, 'row': row_lookup[(retained['path'], retained['symbol'])],
                        'path': retained['path'],
                        'symbol': retained['symbol'], 'score': 0.2})
        strategy['candidate_ranking'] = ranking
        # Ensure new identities do not alias any other exported row.
        raw['evaluation']['variants'][0]['snapshot']['chunks'] = max_row + 20
        _recompute_display_receipts(raw)
        result = diagnostics._validate(raw)
        q = result['variants'][0]['candidate_window']['semantic']['queries'][0]
        self.assertEqual(q['failure_class'], 'missing_from_exported_window')
        self.assertEqual(q['present_below_top10'], [retained])
        self.assertEqual(q['missing_from_exported_window'], [case['relevant'][1]])

    def test_full_rank_corruption_is_rejected_even_when_legacy_display_receipt_is_plausible(self):
        raw = _fixture_report(self)
        strategy = raw['evaluation']['variants'][0]['queries'][3]['experimental_rrf']
        candidate = strategy['candidate_ranking'][11]
        raw['evaluation']['variants'][0]['snapshot']['chunks'] = 5001
        decoy = {'rank': 12, 'row': 5000, 'path': 'late-decoy/12.rs',
                 'symbol': 'late-decoy-12', 'score': 0.1}
        strategy['candidate_ranking'][11] = decoy
        candidate = dict(candidate, rank=13)
        strategy['candidate_ranking'].append(candidate)
        # Its legacy top ten remain all decoys, and rank 12/RR 1/12 look valid
        # to the v2 display-only validation, but contradict the full export.
        with self.assertRaises(ev.EvaluationError):
            diagnostics._validate(raw)
        display_only = copy.deepcopy(raw)
        display_only['evaluation']['format_version'] = 2
        for variant in display_only['evaluation']['variants']:
            for query in variant['queries']:
                for name in diagnostics.STRATEGIES:
                    query[name].pop('candidate_ranking')
        self.assertEqual(diagnostics._validate_display(display_only)['variants'][0]['queries'][3]['experimental_rrf']['first_relevant_rank'], 12)

    def test_bad_candidate_exports_fail_before_output_directory_creation(self):
        mutations = {
            'duplicate row': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][1].__setitem__('row', x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0]['row']),
            'noncontiguous ranks': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('rank', 2),
            'score prefix mismatch': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('score', 123),
            'snapshot bound': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('row', x['evaluation']['variants'][0]['snapshot'].__setitem__('chunks', 0) or 0),
            'positive snapshot exclusive bound': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('row', x['evaluation']['variants'][0]['snapshot']['chunks']),
            'boolean rank': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('rank', True),
            'candidate_k overflow': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'].extend(
                [dict(x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][-1], rank=i)
                 for i in range(11, 22)]),
            'missing strategy': lambda x: x['evaluation']['variants'][0]['queries'][0].pop('hybrid')['candidate_ranking'],
            'invalid version': lambda x: x['evaluation'].__setitem__('format_version', 4),
            'boolean score': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('score', True),
            'nonfinite score': lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['candidate_ranking'][0].__setitem__('score', float('nan')),
            'crossquery row identity': lambda x: _change_row_to_other_query_identity(x),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                bad = _fixture_report(self)
                mutate(bad)
                output = self.base / f'invalid-{name.replace(" ", "-")}'
                with self.assertRaises(ev.EvaluationError):
                    diagnostics.diagnose(bad, output)
                self.assertFalse(output.exists())

    def test_two_precision_variants_are_covered_independently(self):
        raw = _fixture_report(self, compare_fp16=True)
        result = diagnostics._validate(raw)
        self.assertEqual([v['precision'] for v in result['variants']], ['fp32', 'fp16'])
        for variant in result['variants']:
            self.assertEqual(len(variant['candidate_window']['hybrid']['queries']), 4)


def _change_row_to_other_query_identity(raw):
    """Break a global row mapping without changing this query's display receipt."""
    variant = raw['evaluation']['variants'][0]
    current = variant['queries'][0]['semantic']
    used_rows = {hit['row'] for hit in current['candidate_ranking']}
    foreign = next(hit for query in variant['queries'][1:]
                   for hit in query['semantic']['candidate_ranking']
                   if hit['row'] not in used_rows)
    current['candidate_ranking'][0]['row'] = foreign['row']
    current['top_hits'][0]['row'] = foreign['row']


if __name__ == '__main__':
    unittest.main()
