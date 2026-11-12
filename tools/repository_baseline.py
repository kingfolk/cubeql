"""Freeze and execute native CubeQL repository baselines without rewriting labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import repository_collect as runner
import repository_eval as ev


def _path(value, base, directory=False):
    ev._string(value, 'input path')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    # Preserve a venv executable's launch path while resolving its parent boundary.
    path = path.parent.resolve() / path.name
    if directory and not path.is_dir() or not directory and not path.is_file():
        raise ev.EvaluationError(f'missing input: {path}')
    return str(path)


def validate_queries(dataset):
    """Check the native QuerySet schema; Rust resolves exact labels on the snapshot."""
    ev._object(dataset, ('version', 'name', 'queries'), where='native query set')
    if type(dataset['version']) is not int or dataset['version'] != 1:
        raise ev.EvaluationError('native query version must be 1')
    ev._string(dataset['name'], 'dataset name')
    if not isinstance(dataset['queries'], list) or not dataset['queries']:
        raise ev.EvaluationError('native queries must be nonempty')
    ids = []
    for query in dataset['queries']:
        ev._object(query, ('id', 'category', 'text', 'relevant'), ('filter',), 'native query')
        for key in ('id', 'text'):
            ev._string(query[key], f'query {key}')
        if query['category'] not in ('identifier', 'conceptual', 'behavioral', 'cross_file'):
            raise ev.EvaluationError('unsupported native query category')
        ids.append(query['id'])
        filt = query.get('filter', {})
        ev._object(filt, (), ('language', 'path_prefix'), 'native filter')
        if 'language' in filt and filt['language'] is not None:
            ev._string(filt['language'], 'filter language')
        if 'path_prefix' in filt and filt['path_prefix'] is not None:
            ev._filter_prefix(filt['path_prefix'], 'filter path prefix')
        labels = query['relevant']
        if not isinstance(labels, list) or not labels:
            raise ev.EvaluationError('native relevance labels must be nonempty')
        identities = []
        for label in labels:
            ev._object(label, ('path', 'symbol'), where='native label')
            ev._relpath(label['path'], 'label path')
            ev._string(label['symbol'], 'label symbol')
            identities.append((label['path'], label['symbol']))
        ev._unique(identities, 'native labels')
        if query['category'] == 'cross_file' and len({p for p, _ in identities}) < 2:
            raise ev.EvaluationError('cross_file needs labels in multiple files')
    ev._unique(ids, 'native query IDs')


def _normalize(config, base):
    ev._object(config, ('format_version', 'name', 'binary', 'root', 'queries', 'profiles'),
               ('candidate_k', 'repeats', 'compare_fp16', 'timeout_seconds',
                'fallback_lines', 'max_file_bytes'), 'baseline config')
    if type(config['format_version']) is not int or config['format_version'] != 1:
        raise ev.EvaluationError('baseline config format_version must be 1')
    ev._string(config['name'], 'baseline name')
    result = dict(config)
    result['binary'] = _path(config['binary'], base)
    if not os.access(result['binary'], os.X_OK):
        raise ev.EvaluationError('binary is not executable')
    result['root'] = _path(config['root'], base, directory=True)
    result['queries'] = _path(config['queries'], base)
    for key, default, minimum in [('candidate_k', 100, 10), ('repeats', 3, 1),
                                   ('fallback_lines', 80, 1), ('max_file_bytes', 2097152, 1)]:
        result[key] = config.get(key, default)
        ev._integer(result[key], key, minimum)
    result['timeout_seconds'] = config.get('timeout_seconds', 600)
    ev._finite_number(result['timeout_seconds'], 'timeout_seconds')
    if not 0 < result['timeout_seconds'] <= 86400:
        raise ev.EvaluationError('timeout_seconds must be in (0, 86400]')
    result['compare_fp16'] = config.get('compare_fp16', False)
    if type(result['compare_fp16']) is not bool:
        raise ev.EvaluationError('compare_fp16 must be boolean')
    if not isinstance(config['profiles'], list) or not config['profiles']:
        raise ev.EvaluationError('profiles must be nonempty')
    profiles = []
    for profile in config['profiles']:
        ev._object(profile, ('name', 'embedding'), ('model_dir', 'encoder_config'), 'profile')
        ev._string(profile['name'], 'profile name')
        kind = profile['embedding']
        if kind not in ('hash', 'minilm', 'process'):
            raise ev.EvaluationError('unsupported embedding profile')
        expected = {'hash': set(), 'minilm': {'model_dir'}, 'process': {'encoder_config'}}[kind]
        if set(profile) - {'name', 'embedding'} != expected:
            raise ev.EvaluationError('profile provider arguments do not match embedding')
        normalized = dict(profile)
        for key in expected:
            normalized[key] = _path(profile[key], base, directory=key == 'model_dir')
        profiles.append(normalized)
    ev._unique([p['name'] for p in profiles], 'profile names')
    result['profiles'] = profiles
    return result


def _profile_inputs(profile):
    files, directories = {}, {}
    if profile['embedding'] == 'minilm':
        directories[profile['model_dir']] = runner.source_inventory(profile['model_dir'])
    if profile['embedding'] == 'process':
        path = Path(profile['encoder_config'])
        files[str(path)] = runner.sha256_file(path)
        launch = ev._load(path)
        ev._object(launch, ('format_version', 'command', 'args'),
                   ('startup_timeout_ms', 'request_timeout_ms'), 'encoder config')
        if type(launch['format_version']) is not int or launch['format_version'] != 1:
            raise ev.EvaluationError('encoder format_version must be 1')
        if not isinstance(launch['args'], list) or not all(isinstance(a, str) for a in launch['args']):
            raise ev.EvaluationError('encoder args must be strings')
        ev._string(launch['command'], 'encoder command')
        command = Path(launch['command'])
        if not command.is_absolute() or not command.is_file() or not os.access(command, os.X_OK):
            raise ev.EvaluationError('encoder command must be an absolute executable')
        for value in [launch['command'], *launch['args']]:
            artifact = Path(value)
            if not artifact.is_absolute():
                continue
            if artifact.is_file():
                files[value] = runner.sha256_file(artifact)
            elif artifact.is_dir():
                directories[value] = runner.source_inventory(artifact)
        for key in ('startup_timeout_ms', 'request_timeout_ms'):
            if key in launch:
                ev._integer(launch[key], key)
                if launch[key] > 600000:
                    raise ev.EvaluationError('encoder timeout exceeds 600000ms')
    return {'files': files, 'directories': directories}


def _inputs(config, queries):
    return {'binary_sha256': runner.sha256_file(config['binary']),
            'queries_sha256': runner.sha256_file(queries),
            'source': runner.source_inventory(config['root']),
            'profiles': [_profile_inputs(p) for p in config['profiles']],
            'runner_sources': {str(Path(m.__file__).resolve()): runner.sha256_file(m.__file__)
                               for m in (runner, ev)},
            'baseline_source_sha256': runner.sha256_file(__file__)}


def _check_output_boundary(output, config, inputs):
    directories = [config['root']]
    for profile in inputs['profiles']:
        directories.extend(profile['directories'])
    for value in directories:
        directory = Path(value).resolve()
        if output == directory or directory in output.parents:
            raise ev.EvaluationError('output must be outside source root and frozen provider directories')


def prepare(config_path, output):
    """Freeze a machine-specific, create-new recipe without executing any tool/model."""
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = _normalize(ev._load(config_path), config_path.parent)
    labels = ev._load(config['queries'])
    validate_queries(labels)
    original = Path(config['queries']).read_bytes()
    inputs = _inputs(config, config['queries'])
    _check_output_boundary(output, config, inputs)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'queries.json').open('xb') as handle:
        handle.write(original)
    if runner.sha256_file(output / 'queries.json') != inputs['queries_sha256']:
        raise ev.EvaluationError('labels changed while preparing')
    recipe = {'format_version': 1, 'config': config, 'inputs': inputs}
    recipe['fingerprint'] = ev._digest(recipe)
    ev._write_new(output / 'recipe.json', recipe)
    return recipe


def _validate_native(raw, dataset, config):
    if not isinstance(raw, dict) or type(raw.get('format_version')) is not int or raw['format_version'] != 1:
        raise ev.EvaluationError('invalid native evaluation wrapper')
    report = raw.get('evaluation')
    canonical_dataset = dict(dataset, queries=[
        dict(q, filter={'language': None, 'path_prefix': None, **q.get('filter', {})})
        for q in dataset['queries']])
    if not isinstance(report, dict) or type(report.get('format_version')) is not int or report['format_version'] != 2 or report.get('dataset') != canonical_dataset:
        raise ev.EvaluationError('native evaluation dataset/version mismatch')
    options = report.get('options', {})
    if not isinstance(options, dict) or any(type(options.get(k)) is not type(config[k]) or options[k] != config[k]
                                             for k in ('candidate_k', 'repeats', 'compare_fp16')):
        raise ev.EvaluationError('native evaluation options mismatch')
    variants = report.get('variants')
    if not isinstance(variants, list) or len(variants) != (2 if config['compare_fp16'] else 1):
        raise ev.EvaluationError('native precision variant count mismatch')
    wanted = [q['id'] for q in dataset['queries']]
    strategies = {'semantic', 'literal', 'experimental_rrf', 'lexical', 'symbol', 'hybrid'}
    for variant in variants:
        if not isinstance(variant, dict) or not isinstance(variant.get('queries'), list):
            raise ev.EvaluationError('invalid native variant/query list')
        if not all(isinstance(q, dict) for q in variant['queries']) or [q.get('id') for q in variant['queries']] != wanted:
            raise ev.EvaluationError('native evaluation query coverage/order mismatch')
        if not isinstance(variant.get('overall'), dict) or set(variant['overall']) != strategies:
            raise ev.EvaluationError('native strategy coverage mismatch')
        if not isinstance(variant.get('snapshot'), dict) or not isinstance(variant.get('by_category'), dict):
            raise ev.EvaluationError('missing native snapshot/category summaries')
        for summary in variant['overall'].values():
            if not isinstance(summary, dict) or type(summary.get('queries')) is not int or summary['queries'] != len(wanted):
                raise ev.EvaluationError('native metric denominator mismatch')
            for key in ('recall_at_5', 'recall_at_10', 'mrr'):
                ev._finite_number(summary.get(key), key)
                if summary[key] > 1:
                    raise ev.EvaluationError('invalid native quality metric')
    expected_precisions = ['fp32', 'fp16'] if config['compare_fp16'] else ['fp32']
    if [v['snapshot'].get('precision') for v in variants] != expected_precisions:
        raise ev.EvaluationError('native precision identities mismatch')
    return report


def run(recipe_dir, output, *, execute=False):
    """Execute frozen profiles; failed conditions stay explicit, without aggregate renormalizing."""
    if not execute:
        raise ev.EvaluationError('run requires --execute to launch local indexing and inference')
    recipe_dir, output = Path(recipe_dir).resolve(), Path(output).resolve()
    recipe = ev._load(recipe_dir / 'recipe.json')
    ev._object(recipe, ('format_version', 'config', 'inputs', 'fingerprint'), where='recipe')
    if type(recipe['format_version']) is not int or recipe['format_version'] != 1:
        raise ev.EvaluationError('recipe format_version must be 1')
    if recipe['fingerprint'] != ev._digest({k: v for k, v in recipe.items() if k != 'fingerprint'}):
        raise ev.EvaluationError('recipe fingerprint mismatch')
    config = recipe['config']
    # Archived query bytes are the execution input, even if the original label file moved.
    checked_config = dict(config, queries=str(recipe_dir / 'queries.json'))
    normalized = _normalize(checked_config, recipe_dir)
    dataset = ev._load(recipe_dir / 'queries.json')
    validate_queries(dataset)
    if _inputs(normalized, recipe_dir / 'queries.json') != recipe['inputs']:
        raise ev.EvaluationError('frozen source/binary/provider/labels changed')
    _check_output_boundary(output, config, recipe['inputs'])
    output.mkdir(parents=True, exist_ok=False)
    ev._write_new(output / 'recipe.json', recipe)
    executions, conditions = [], []
    for index, profile in enumerate(config['profiles']):
        prefix = f'profile-{index:03d}'
        snapshot = output / f'{prefix}.arrow'
        provider = []
        if profile['embedding'] == 'process':
            provider = ['--encoder-config', profile['encoder_config']]
        elif profile['embedding'] == 'minilm':
            provider = ['--model-dir', profile['model_dir']]
        command = [config['binary'], 'index', config['root'], '--output', str(snapshot),
                   '--embedding', profile['embedding'], *provider, '--fallback-lines', str(config['fallback_lines']),
                   '--max-file-bytes', str(config['max_file_bytes']), '--format', 'json']
        built = runner.execute(command, output, f'{prefix}-index', timeout_s=config['timeout_seconds'])
        executions.append(built)
        condition = {'name': profile['name'], 'embedding': profile['embedding'],
                     'query_count': len(dataset['queries']), 'status': built['status']}
        if built['status'] == 'ok':
            command = [config['binary'], 'evaluate', str(snapshot), str(recipe_dir / 'queries.json'),
                       *provider, '--candidate-k', str(config['candidate_k']), '--repeats', str(config['repeats']),
                       '--format', 'json']
            if config['compare_fp16']:
                command.append('--compare-fp16')
            evaluated = runner.execute(command, output, f'{prefix}-evaluate', timeout_s=config['timeout_seconds'])
            executions.append(evaluated)
            condition['status'] = evaluated['status']
            if evaluated['status'] == 'ok':
                try:
                    native = _validate_native(ev._load(output / evaluated['stdout']), dataset, config)
                    condition['variants'] = [{key: variant[key] for key in ('snapshot', 'overall', 'by_category')}
                                             for variant in native['variants']]
                    condition['native_report'] = evaluated['stdout']
                    condition['snapshot_sha256'] = runner.sha256_file(snapshot)
                except (ev.EvaluationError, KeyError, TypeError) as error:
                    condition.update(status='failed', error=str(error))
            else:
                condition['error'] = evaluated.get('error', 'native evaluate failed')
        else:
            condition['error'] = built.get('error', 'native index failed')
        conditions.append(condition)
        ev._write_new(output / f'{prefix}-condition.json', condition)
        ev._write_new(output / f'{prefix}-execution.json', executions[-2:] if built['status'] == 'ok' else [built])
        if _inputs(normalized, recipe_dir / 'queries.json') != recipe['inputs']:
            raise ev.EvaluationError('frozen inputs changed during execution; no completed summary')
    summary = {'format_version': 1, 'recipe_fingerprint': recipe['fingerprint'], 'conditions': conditions,
               'expected_condition_count': len(config['profiles']),
               'successful_condition_count': sum(c['status'] == 'ok' for c in conditions),
               'timing': 'Native evaluate retains one provider/query-vector set per condition; raw native phase timers are preserved. CLI wall time includes startup.',
               'limits': 'Original symbol labels and native metric denominators preserved. No average across conditions; failed conditions have no valid quality score.',
               'runtime': {'platform': platform.platform(), 'python': sys.version, 'logical_cpus': os.cpu_count()}}
    ev._write_new(output / 'summary.json', summary)
    ev._write_new(output / 'execution.json', executions)
    ev._write_new(output / 'manifest.json', {str(p.relative_to(output)): runner.sha256_file(p)
                                            for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--config', required=True)
    prep.add_argument('--output', required=True)
    invoke = sub.add_parser('run')
    invoke.add_argument('--recipe', required=True)
    invoke.add_argument('--output', required=True)
    invoke.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            prepare(args.config, args.output)
        else:
            run(args.recipe, args.output, execute=args.execute)
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f'repository_baseline: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
