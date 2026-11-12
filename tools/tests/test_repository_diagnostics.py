import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_diagnostics as diagnostics
import repository_eval as ev


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        root = self.folder / 'repo'
        root.mkdir()
        (root / 'lib.rs').write_text('line\n' * 20)
        self.spec = {'format_version': 1, 'name': 'manual-development-fixture', 'split': 'development',
            'repositories': [{'id': 'repo', 'revision': 'fixture'}], 'queries': [
                {'id': 'q1', 'repository': 'repo', 'category': 'manual', 'text': 'find target', 'filter': {},
                 'anchors': [{'id': 'target', 'path': 'lib.rs', 'start_line': 12, 'end_line': 12}]}]}
        self.plan = ev.freeze(self.spec, root)

    def run_data(self, name, lines=(), status='ok', boundary='fixture request'):
        result = {'query_id': 'q1', 'status': status, 'elapsed_ms': 5,
                  'hits': [{'path': 'lib.rs', 'start_line': n, 'end_line': n} for n in lines]}
        if status != 'ok':
            result['error'] = 'retained failure'
        return {'format_version': 1, 'plan_fingerprint': self.plan['fingerprint'], 'name': name,
                'tool': {'name': 'fixture', 'revision': '1'}, 'configuration': {'timing_boundary': boundary},
                'candidate_k': 20, 'results': [result]}

    def test_missing_export_is_distinct_from_present_below_top10(self):
        missing = self.run_data('missing')
        lower = self.run_data('lower', range(1, 13))
        output = diagnostics.diagnose(self.plan, [missing, lower], self.folder / 'report')['report']
        self.assertEqual(output['runs'][0]['queries']['q1']['failure_class'], 'missing_from_exported_window')
        row = output['runs'][1]['queries']['q1']
        self.assertEqual(row['failure_class'], 'present_below_top10')
        self.assertEqual(row['anchor_ranks'], {'target': 12})
        self.assertEqual(row['window_recall'], 1)
        self.assertEqual(row['recall_at_10'], 0)
        self.assertEqual(row['below_top10_anchor_ids'], ['target'])
        self.assertEqual(output['paired_comparisons'][0]['outcome_counts'], {'shared_miss': 1})

    def test_paired_recovered_and_lost_are_manual_arithmetic(self):
        missing, found = self.run_data('missing'), self.run_data('found', [12])
        output = diagnostics.diagnose(self.plan, [missing, found], self.folder / 'up')['report']
        self.assertEqual(output['paired_comparisons'][0]['outcome_counts'], {'recovered': 1})
        self.assertEqual(output['paired_comparisons'][0]['macro_delta']['recall_at_10'], 1)
        output = diagnostics.diagnose(self.plan, [found, missing], self.folder / 'down')['report']
        self.assertEqual(output['paired_comparisons'][0]['outcome_counts'], {'lost': 1})
        self.assertEqual(output['paired_comparisons'][0]['macro_delta']['recall_at_10'], -1)

    def test_execution_failures_remain_in_quality_denominator_and_latency_coverage(self):
        failed = self.run_data('failed', status='timeout')
        out = diagnostics.diagnose(self.plan, [failed], self.folder / 'failure')['report']['runs'][0]
        self.assertEqual(out['overall']['query_count'], 1)
        self.assertEqual(out['overall']['recall_at_10'], 0)
        self.assertEqual(out['queries']['q1']['failure_class'], 'execution_failure')
        self.assertFalse(out['latency']['complete_query_coverage'])
        self.assertEqual(out['latency']['sample_count'], 0)
        self.assertIsNone(out['latency']['median_ms'])

    def test_timing_context_and_exported_windows_not_silently_equated(self):
        a = self.run_data('a', [12], boundary='resident search')
        b = self.run_data('b', [12], boundary='cold process')
        b['candidate_k'] = 10
        out = diagnostics.diagnose(self.plan, [a, b], self.folder / 'timing')['report']
        paired = out['paired_comparisons'][0]
        self.assertFalse(paired['same_declared_timing_boundary'])
        self.assertFalse(paired['equal_exported_window'])
        self.assertFalse(paired['equal_internal_work_attested'])
        self.assertTrue(paired['latency_complete_both_conditions'])
        self.assertIsNone(paired['macro_delta']['mrr'])

    def test_validation_and_create_new_before_outputs(self):
        run = self.run_data('a', [12])
        run['plan_fingerprint'] = 'wrong'
        output = self.folder / 'bad'
        with self.assertRaises(ev.EvaluationError):
            diagnostics.diagnose(self.plan, [run], output)
        self.assertFalse(output.exists())
        run = self.run_data('a', [12])
        output.mkdir()
        (output / 'sentinel').write_text('preserve')
        with self.assertRaises(FileExistsError):
            diagnostics.diagnose(self.plan, [run], output)
        self.assertEqual((output / 'sentinel').read_text(), 'preserve')

    def test_saved_artifacts_hash_and_markdown_escaping(self):
        run = self.run_data('a|<tag>\nname', [12])
        output = self.folder / 'report'
        result = diagnostics.diagnose(self.plan, [run], output)
        self.assertIn('a\\|&lt;tag&gt; name', result['markdown'])
        self.assertEqual(ev._load(output / 'runs.json'), [run])
        for filename, checksum in ev._load(output / 'manifest.json').items():
            self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), checksum)


if __name__ == '__main__':
    unittest.main()
