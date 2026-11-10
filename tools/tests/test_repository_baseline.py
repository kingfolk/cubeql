import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_baseline as baseline
import repository_eval as ev


FAKE = r'''#!/usr/bin/env python3
import json,sys
from pathlib import Path
args=sys.argv[1:]
mode_path=Path(__file__).with_name('mode')
mode=mode_path.read_text() if mode_path.exists() else ''
def arg(name):return args[args.index(name)+1]
if args[0]=='index':
    if mode=='fail-first' and 'profile-000' in arg('--output'):
        print('retained index failure',file=sys.stderr);sys.exit(7)
    Path(arg('--output')).write_bytes(b'fake-snapshot')
    if mode=='mutate':
        with (Path(args[1])/'lib.rs').open('a') as f:f.write('// mutation\n')
    print(json.dumps({'format_version':1,'snapshot':{'embedding_id':'fixture'}}))
else:
    dataset=json.loads(Path(args[2]).read_text())
    for q in dataset['queries']:
        q['filter']={'language':None,'path_prefix':None,**q.get('filter',{})}
    if mode=='wrong-query':dataset['queries'][0]['text']='changed text'
    summary={'queries':len(dataset['queries']),'recall_at_5':1.0,'recall_at_10':1.0,'mrr':1.0}
    variant={'snapshot':{'precision':'fp32'},'queries':[{'id':q['id']} for q in dataset['queries']],
        'overall':{s:summary for s in ['semantic','literal','experimental_rrf','lexical','symbol','hybrid']},'by_category':{}}
    variants=[variant]
    if '--compare-fp16' in args:variants.append({**variant,'snapshot':{'precision':'fp16'}})
    print(json.dumps({'format_version':1,'evaluation':{'format_version':2,'dataset':dataset,'variants':variants,
        'options':{'candidate_k':int(arg('--candidate-k')),'repeats':int(arg('--repeats')),'compare_fp16':'--compare-fp16' in args}}}))
'''


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.root = self.folder / 'source'
        self.root.mkdir()
        (self.root / 'lib.rs').write_text('pub fn load_config() {}\n')
        self.binary = self.folder / 'fake-cubeql'
        self.binary.write_text(FAKE)
        self.binary.chmod(0o755)
        self.dataset = {'version': 1, 'name': 'preserved-labels', 'queries': [
            {'id': 'read', 'category': 'identifier', 'text': 'load_config',
             'filter': {'language': 'rust'}, 'relevant': [{'path': 'lib.rs', 'symbol': 'load_config'}]}]}
        self.labels = self.folder / 'labels.json'
        self.labels.write_text(json.dumps(self.dataset, indent=4) + '\n\n')
        self.config = {'format_version': 1, 'name': 'native-smoke', 'binary': str(self.binary),
                       'root': str(self.root), 'queries': str(self.labels),
                       'profiles': [{'name': 'first', 'embedding': 'hash'}, {'name': 'second', 'embedding': 'hash'}],
                       'candidate_k': 10, 'repeats': 1, 'timeout_seconds': 10}
        self.config_path = self.folder / 'config.json'
        self.recipe = self.folder / 'recipe'
        self.output = self.folder / 'result'

    def prepare(self):
        self.config_path.write_text(json.dumps(self.config))
        return baseline.prepare(self.config_path, self.recipe)

    def test_prepare_preserves_label_bytes_without_executing_binary(self):
        self.prepare()
        self.assertEqual((self.recipe / 'queries.json').read_bytes(), self.labels.read_bytes())
        self.assertEqual(sorted(p.name for p in self.recipe.iterdir()), ['queries.json', 'recipe.json'])
        self.assertFalse(self.output.exists())

    def test_run_requires_explicit_execute(self):
        self.prepare()
        with self.assertRaisesRegex(ev.EvaluationError, '--execute'):
            baseline.run(self.recipe, self.output)
        self.assertFalse(self.output.exists())

    def test_native_filters_and_all_strategies_preserved_across_profiles_and_precisions(self):
        self.config['compare_fp16'] = True
        self.prepare()
        summary = baseline.run(self.recipe, self.output, execute=True)
        self.assertEqual(summary['expected_condition_count'], 2)
        self.assertEqual(summary['successful_condition_count'], 2)
        self.assertEqual(len(summary['conditions'][0]['variants']), 2)
        self.assertEqual(len(summary['conditions'][0]['variants'][0]['overall']), 6)
        raw = ev._load(self.output / summary['conditions'][0]['native_report'])
        self.assertEqual(raw['evaluation']['dataset']['queries'][0]['filter']['language'], 'rust')
        self.assertEqual(raw['evaluation']['dataset']['queries'][0]['text'], 'load_config')
        self.assertEqual((self.recipe / 'queries.json').read_bytes(), self.labels.read_bytes())

    def test_failed_condition_remains_in_expected_count_and_diagnostics(self):
        (self.folder / 'mode').write_text('fail-first')
        self.prepare()
        summary = baseline.run(self.recipe, self.output, execute=True)
        self.assertEqual(summary['expected_condition_count'], 2)
        self.assertEqual(summary['successful_condition_count'], 1)
        self.assertEqual([c['status'] for c in summary['conditions']], ['failed', 'ok'])
        self.assertNotIn('variants', summary['conditions'][0])
        self.assertIn('retained index failure', (self.output / 'profile-000-index.stderr').read_text())

    def test_wrong_native_dataset_fails_condition_instead_of_scoring_changed_queries(self):
        (self.folder / 'mode').write_text('wrong-query')
        self.prepare()
        summary = baseline.run(self.recipe, self.output, execute=True)
        self.assertEqual(summary['successful_condition_count'], 0)
        self.assertTrue(all(c['status'] == 'failed' for c in summary['conditions']))

    def test_source_mutation_during_run_does_not_produce_completed_summary(self):
        (self.folder / 'mode').write_text('mutate')
        self.prepare()
        with self.assertRaisesRegex(ev.EvaluationError, 'changed during'):
            baseline.run(self.recipe, self.output, execute=True)
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertTrue((self.output / 'profile-000-index.stdout').exists())

    def test_frozen_labels_binary_source_and_recipe_tampering_rejected_before_launch(self):
        self.prepare()
        for path in [self.recipe / 'queries.json', self.binary, self.root / 'lib.rs']:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.assertRaises(ev.EvaluationError):
                    baseline.run(self.recipe, self.output, execute=True)
                path.write_bytes(original)
                self.assertFalse(self.output.exists())
        recipe = ev._load(self.recipe / 'recipe.json')
        recipe['config']['candidate_k'] = 20
        (self.recipe / 'recipe.json').write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ev.EvaluationError, 'fingerprint'):
            baseline.run(self.recipe, self.output, execute=True)

    def test_strict_config_and_duplicate_profiles_rejected(self):
        for update in [{'extra': True}, {'candidate_k': True}, {'compare_fp16': 1},
                       {'profiles': [{'name': 'same', 'embedding': 'hash'}] * 2},
                       {'profiles': [{'name': 'bad', 'embedding': 'process'}]}]:
            with self.subTest(update=update):
                config = dict(self.config, **update)
                self.config_path.write_text(json.dumps(config))
                with self.assertRaises(ev.EvaluationError):
                    baseline.prepare(self.config_path, self.recipe)
                self.assertFalse(self.recipe.exists())

    def test_prepare_freezes_process_launch_inputs_without_invoking_model(self):
        script = self.folder / 'worker.py'
        script.write_text('raise RuntimeError("must not run during prepare")\n')
        launch = self.folder / 'encoder.json'
        launch.write_text(json.dumps({'format_version': 1, 'command': sys.executable, 'args': [str(script)]}))
        self.config['profiles'] = [{'name': 'code-model', 'embedding': 'process', 'encoder_config': str(launch)}]
        recipe = self.prepare()
        self.assertIn(str(script), recipe['inputs']['profiles'][0]['files'])
        script.write_text('changed\n')
        with self.assertRaisesRegex(ev.EvaluationError, 'changed'):
            baseline.run(self.recipe, self.output, execute=True)

    def test_create_new_and_source_output_boundaries(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            baseline.prepare(self.config_path, self.recipe)
        with self.assertRaisesRegex(ev.EvaluationError, 'outside source root'):
            baseline.run(self.recipe, self.root / 'result', execute=True)
        baseline.run(self.recipe, self.output, execute=True)
        with self.assertRaises(FileExistsError):
            baseline.run(self.recipe, self.output, execute=True)

    def test_prepare_output_cannot_mutate_frozen_model_directory(self):
        model = self.folder / 'model'
        model.mkdir()
        (model / 'weights').write_bytes(b'fixture only')
        self.config['profiles'] = [{'name': 'minilm', 'embedding': 'minilm', 'model_dir': str(model)}]
        self.config_path.write_text(json.dumps(self.config))
        output = model / 'recipe'
        with self.assertRaisesRegex(ev.EvaluationError, 'provider directories'):
            baseline.prepare(self.config_path, output)
        self.assertFalse(output.exists())

    def test_native_query_validation_catches_ambiguous_or_unsupported_labels(self):
        for mutate in [lambda q: q['queries'].append(copy.deepcopy(q['queries'][0])),
                       lambda q: q['queries'][0]['relevant'].append(copy.deepcopy(q['queries'][0]['relevant'][0])),
                       lambda q: q['queries'][0].update(category='cross_file'),
                       lambda q: q['queries'][0].update(filter={'path_prefix': '../'}),
                       lambda q: q['queries'][0].update(extra='misspelled field')]:
            dataset = copy.deepcopy(self.dataset)
            mutate(dataset)
            with self.assertRaises(ev.EvaluationError):
                baseline.validate_queries(dataset)


if __name__ == '__main__':
    unittest.main()
