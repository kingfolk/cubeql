import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_eval as ev
import repository_native_diagnostics as diagnostics


class NativeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.dataset = {'version': 1, 'name': 'native-symbol-fixture', 'queries': [
            {'id': 'q1', 'category': 'identifier', 'text': 'one', 'relevant': [
                {'path': 'src/a.rs', 'symbol': 'one'}, {'path': 'src/b.rs', 'symbol': 'two'}]},
            {'id': 'q2', 'category': 'conceptual', 'text': 'three', 'relevant': [
                {'path': 'src/c.rs', 'symbol': 'three'}]},
            {'id': 'q3', 'category': 'identifier', 'text': 'four', 'relevant': [
                {'path': f'src/{letter}.rs', 'symbol': f'{letter}'} for letter in 'defg']},
            {'id': 'q4', 'category': 'behavioral', 'text': 'five', 'relevant': [
                {'path': 'src/h.rs', 'symbol': 'five'}]},
        ]}
        for query in self.dataset['queries']:
            query['filter'] = {'language': None, 'path_prefix': None}
        self.raw = self.make_report(compare_fp16=False)

    def hit(self, pair, row):
        return {'row': row, 'path': pair[0], 'symbol': pair[1], 'language': 'rust', 'kind': 'function',
                'start_line': 1, 'end_line': 2, 'score': -0.1 if row == 0 else 0.8,
                'provenance': [], 'suppressed_overlaps': []}

    def strategies_for(self, query, strategy):
        labels = [(label['path'], label['symbol']) for label in query['relevant']]
        decoys = [(f'decoy/{query["id"]}-{i}.rs', f'decoy-{i}') for i in range(12)]
        plans = {
            'semantic': {'q1': [decoys[0], labels[0]], 'q2': [], 'q3': labels[:2], 'q4': []},
            'literal': {'q1': labels, 'q2': labels, 'q3': [], 'q4': []},
            'experimental_rrf': {'q1': [labels[0]], 'q2': [], 'q3': [], 'q4': []},
            'lexical': {'q1': labels, 'q2': labels, 'q3': [], 'q4': []},
            'symbol': {'q1': [], 'q2': [], 'q3': labels[:1], 'q4': []},
            'hybrid': {'q1': [*decoys[:6], labels[0]], 'q2': labels, 'q3': [], 'q4': []},
        }
        # For a rank beyond the display window, report no top-ten hit but retain the exact rank.
        pairs = plans[strategy][query['id']]
        if strategy == 'experimental_rrf' and query['id'] == 'q4':
            pairs = decoys[:10]
        hits = [self.hit(pair, i) for i, pair in enumerate(pairs)]
        rank = next((i + 1 for i, pair in enumerate(pairs) if pair in labels), None)
        if strategy == 'experimental_rrf' and query['id'] == 'q4':
            rank = 12
        if strategy == 'semantic' and query['id'] == 'q4':
            rank = None
        return hits, rank

    def make_variant(self, precision):
        queries = []
        for query in self.dataset['queries']:
            row = {'id': query['id'], 'category': query['category'], 'embedding_ms': 0.2,
                   'filter_ms': 0.1, 'first_semantic_ms': 0.3, 'hybrid_filter_ms': 0.05}
            for strategy in diagnostics.STRATEGIES:
                hits, rank = self.strategies_for(query, strategy)
                label_pairs = {(x['path'], x['symbol']) for x in query['relevant']}
                recall5 = sum((h['path'], h['symbol']) in label_pairs for h in hits[:5]) / len(label_pairs)
                recall10 = sum((h['path'], h['symbol']) in label_pairs for h in hits[:10]) / len(label_pairs)
                if rank is not None and rank > 10:
                    reciprocal = 1 / rank
                else:
                    reciprocal = 0 if rank is None else 1 / rank
                row[strategy] = {
                    'metrics': {'recall_at_5': recall5, 'recall_at_10': recall10,
                                'reciprocal_rank': reciprocal, 'first_relevant_rank': rank},
                    'retrieval': {'median_ms': 2.0, 'min_ms': 1.0, 'max_ms': 3.0, 'samples': 2},
                    'materialization_ms': 0.01, 'top_hits': hits}
            queries.append(row)
        overall = {name: self.summary(queries, name) for name in diagnostics.STRATEGIES}
        categories = {}
        for category in dict.fromkeys(q['category'] for q in self.dataset['queries']):
            subset = [q for q in queries if q['category'] == category]
            categories[category] = {name: self.summary(subset, name) for name in diagnostics.STRATEGIES}
        return {'snapshot': {'precision': precision}, 'baseline_conversion_ms': 0.1,
                'baseline_upload_ms': 0.2, 'conversion_ms': 0.3, 'upload_ms': 0.4,
                'text_index_ms': 0.5, 'queries': queries, 'overall': overall, 'by_category': categories}

    @staticmethod
    def summary(rows, strategy):
        metrics = [row[strategy]['metrics'] for row in rows]
        return {'queries': len(rows),
                'recall_at_5': sum(x['recall_at_5'] for x in metrics) / len(metrics),
                'recall_at_10': sum(x['recall_at_10'] for x in metrics) / len(metrics),
                'mrr': sum(x['reciprocal_rank'] for x in metrics) / len(metrics)}

    def make_report(self, compare_fp16):
        report = {'format_version': 2, 'dataset': copy.deepcopy(self.dataset),
                  'backend': 'fixture', 'options': {'candidate_k': 20, 'repeats': 2,
                                                       'compare_fp16': compare_fp16},
                  'rrf_rank_constant': 60, 'embedding_total_ms': 0.8,
                  'embedding_execution': 'fixture',
                  'variants': [self.make_variant('fp32')]}
        if compare_fp16:
            report['variants'].append(self.make_variant('fp16'))
        return {'format_version': 1, 'evaluation': report}

    def test_recomputes_summaries_and_reports_paired_query_arithmetic(self):
        result = diagnostics._validate(self.raw)
        variant = result['variants'][0]
        self.assertEqual(variant['overall']['semantic']['recall_at_10'], 0.25)
        self.assertEqual(variant['by_category']['identifier']['semantic']['queries'], 2)
        comparisons = {x['strategy']: x for x in variant['paired_comparisons']}
        self.assertEqual(comparisons['lexical']['outcome_counts']['recovered'], 1)
        self.assertEqual(comparisons['lexical']['outcome_counts']['improved'], 1)
        self.assertEqual(comparisons['lexical']['outcome_counts']['shared_miss'], 1)
        self.assertEqual(comparisons['symbol']['outcome_counts']['regressed'], 1)
        self.assertEqual(comparisons['hybrid']['outcome_counts']['lost'], 1)
        q1 = comparisons['hybrid']['queries'][0]
        self.assertEqual(q1['outcome'], 'tied')
        self.assertEqual(q1['semantic']['first_relevant_rank'], 2)
        self.assertEqual(q1['compared']['first_relevant_rank'], 7)
        self.assertEqual(q1['delta']['recall_at_10'], 0)

    def test_recomputes_category_summary_including_fractional_recall(self):
        result = diagnostics._validate(self.raw)
        self.assertEqual(result['variants'][0]['queries'][0]['semantic']['recall_at_5'], 0.5)
        bad = copy.deepcopy(self.raw)
        bad['evaluation']['variants'][0]['by_category']['identifier']['semantic']['recall_at_10'] = 0.9
        with self.assertRaises(ev.EvaluationError):
            diagnostics._validate(bad)

    def test_rejects_summary_denominator_query_order_and_category_mismatch(self):
        mutations = [
            lambda x: x['evaluation']['variants'][0]['overall']['semantic'].__setitem__('queries', 3),
            lambda x: x['evaluation']['variants'][0]['queries'].reverse(),
            lambda x: x['evaluation']['variants'][0]['queries'][0].__setitem__('category', 'behavioral'),
            lambda x: x['evaluation']['variants'][0]['overall']['semantic'].__setitem__('recall_at_10', 0.9),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(self.raw)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ev.EvaluationError):
                diagnostics._validate(bad)

    def test_rejects_rank_boolean_nan_and_hit_inconsistencies(self):
        for mutation in (
            lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['metrics'].__setitem__('first_relevant_rank', True),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['metrics'].__setitem__('recall_at_10', float('nan')),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['metrics'].__setitem__('reciprocal_rank', 0.1),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['top_hits'][1].__setitem__('row', 0),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['top_hits'][0].update(
                path='src/a.rs', symbol='one'),
            lambda x: x['evaluation']['variants'][0]['queries'][0].__setitem__('first_semantic_ms', float('nan')),
            lambda x: x['evaluation']['variants'][0]['queries'][0]['semantic']['retrieval'].__setitem__('samples', 1),
        ):
            bad = copy.deepcopy(self.raw)
            mutation(bad)
            with self.subTest(mutation=mutation), self.assertRaises(ev.EvaluationError):
                diagnostics.diagnose(bad, self.base / 'invalid')
            self.assertFalse((self.base / 'invalid').exists())

    def test_accepts_exact_fp32_fp16_variants_and_rejects_partial_variant_set(self):
        valid = self.make_report(compare_fp16=True)
        self.assertEqual([x['precision'] for x in diagnostics._validate(valid)['variants']], ['fp32', 'fp16'])
        valid['evaluation']['variants'].pop()
        with self.assertRaises(ev.EvaluationError):
            diagnostics._validate(valid)

    def test_create_new_bundle_hashes_and_copies_canonical_input(self):
        output = self.base / 'diagnostics'
        original = json.dumps(self.raw, separators=(',', ':')).encode()
        result = diagnostics.diagnose(self.raw, output, raw_bytes=original)
        canonical = ev._canonical(self.raw) + b'\n'
        self.assertEqual((output / 'input-report.json').read_bytes(), canonical)
        self.assertEqual(result['diagnostics']['source']['raw_bytes_sha256'], hashlib.sha256(original).hexdigest())
        manifest = json.loads((output / 'manifest.json').read_text())
        for name, checksum in manifest.items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), checksum)
        sentinel = output / 'sentinel'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            diagnostics.diagnose(self.raw, output)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_raw_bytes_must_match_and_cli_rejects_duplicate_json_keys(self):
        with self.assertRaises(ev.EvaluationError):
            diagnostics.diagnose(self.raw, self.base / 'mismatch', raw_bytes=b'{"format_version":1}')
        self.assertFalse((self.base / 'mismatch').exists())
        duplicate = self.base / 'duplicate.json'
        duplicate.write_text('{"format_version":1,"format_version":1,"evaluation":{}}')
        output = self.base / 'duplicate-output'
        self.assertEqual(diagnostics.main(['--report', str(duplicate), '--output', str(output)]), 2)
        self.assertFalse(output.exists())

    def test_rejects_unrepresentably_large_json_number(self):
        bad = copy.deepcopy(self.raw)
        bad['evaluation']['embedding_total_ms'] = 10 ** 10000
        with self.assertRaises(ev.EvaluationError):
            diagnostics._validate(bad)


if __name__ == '__main__':
    unittest.main()
