"""Opt-in, local two-file incremental-update/MCP-reload acceptance.

No paid model or agent calls. Hash vectors are fixtures, and successful model
queries are lifecycle checks rather than a relevance benchmark.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import repository_collect as rc
import repository_eval as ev
import repository_mcp as mcp

OLD = '/// Load legacy application settings.\npub fn load_legacy_settings() { legacy_settings(); }\n'
NEW = '/// Load current application settings.\npub fn load_current_settings() { current_settings(); }\n'
STABLE = '/// Persist application settings.\npub fn persist_settings() { write_settings(); }\n'


def _rss(pid):
    """Point-in-time parent RSS only; never present this as process-tree peak."""
    try:
        value = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True,
                               text=True, timeout=2, check=True).stdout.strip()
        return int(value) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run(code_binary, mcp_binary, output, encoder_config=None, timeout_s=180):
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (float, int)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout must be finite and positive')
    code_binary, mcp_binary = Path(code_binary).resolve(strict=True), Path(mcp_binary).resolve(strict=True)
    encoder_config = Path(encoder_config).resolve(strict=True) if encoder_config else None
    before = {'code': rc._profile_fingerprints(code_binary, encoder_config, None),
              'mcp': rc._fingerprint_path(mcp_binary),
              'producers':{Path(m.__file__).name:rc.sha256_file(m.__file__) for m in (rc, ev, mcp)},
              'runner_sha256':rc.sha256_file(__file__)}
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    ev._write_new(output/'inputs-before.json', before)
    source = output/'source'
    source.mkdir()
    (source/'loader.rs').write_text(OLD, encoding='utf-8')
    (source/'persist.rs').write_text(STABLE, encoding='utf-8')
    old_snapshot, new_snapshot, target = output/'old.arrow', output/'new.arrow', output/'reload.arrow'
    commands, rpc, rss = [], [], []
    session = None
    def command(argv, stem):
        result = rc.execute(argv, output, stem, timeout_s=timeout_s)
        commands.append(result)
        ev._write_new(output/(stem+'-execution.json'), result)
        if result['status'] != 'ok':
            raise ValueError(f'{stem} failed; retained command evidence')
        return ev._load(output/(stem+'.stdout'))
    try:
        argv = [str(code_binary), 'index', str(source), '--output', str(old_snapshot), '--format', 'json']
        if encoder_config:
            argv += ['--embedding', 'process', '--encoder-config', str(encoder_config)]
        else:
            argv += ['--dimensions', '32']
        initial = command(argv, 'index')
        argv = [str(mcp_binary), str(old_snapshot), '--reload-index', str(target), '--workspace-root', str(source)]
        if encoder_config:
            argv += ['--encoder-config', str(encoder_config)]
        startup = time.monotonic()
        session = mcp.JSONLineSession(argv, output, 'mcp', timeout_s=timeout_s)
        session.request('initialize', {'protocolVersion':'2025-11-25','capabilities':{},
                                      'clientInfo':{'name':'cubeql-reload-smoke','version':'1'}})
        session.notify('notifications/initialized', {})
        startup_ms = (time.monotonic()-startup)*1000
        tools = session.request('tools/list', {})
        if 'reload_snapshot' not in {t['name'] for t in tools['tools']}:
            raise ValueError('reload tool not discovered')
        def call(name, arguments, expected_error=None):
            started = time.monotonic()
            raw = session.request('tools/call', {'name':name, 'arguments':arguments})
            rpc.append({'name':name, 'request_wall_ms':(time.monotonic()-started)*1000, 'raw_result':raw})
            if expected_error:
                if raw.get('isError') is not True or raw.get('structuredContent',{}).get('error',{}).get('code') != expected_error:
                    raise ValueError(f'expected {expected_error}: {raw}')
                return raw['structuredContent']
            return mcp._tool_result(raw)
        def search(symbol):
            value = call('search_code', {'query':symbol,'strategy':'symbol','top_k':1})
            if not value['hits'] or value['hits'][0]['symbol'] != symbol:
                raise ValueError(f'symbol not found: {symbol}')
            return value
        def handle(hit):
            return {'snapshot_id':hit['snapshot_id'], 'chunk_id':hit['hits'][0]['chunk_id']}
        old = search('load_legacy_settings')
        old_handle = handle(old)
        if OLD.strip().split('\n')[-1] not in call('read_code',old_handle)['text']:
            raise ValueError('wrong original source')
        semantic = []
        for _ in range(3):
            semantic.append(call('search_code', {'query':'Load application settings','strategy':'semantic','top_k':1}))
        rss.append({'phase':'before_update','mcp_pid':session.proc.pid,'mcp_rss_bytes':_rss(session.proc.pid)})
        (source/'loader.rs').write_text(NEW, encoding='utf-8')
        if call('check_code',old_handle)['status'] != 'modified':
            raise ValueError('freshness did not detect edit')
        argv = [str(code_binary),'update',str(old_snapshot),'--output',str(new_snapshot),'--format','json']
        if encoder_config:
            argv += ['--encoder-config',str(encoder_config)]
        updated = command(argv,'update')
        if updated['report']['embedded'] != 1 or updated['report']['reused'] != 1:
            raise ValueError('expected exactly one newly encoded and one reused fixture chunk')
        # Updating on disk does not implicitly switch the active session.
        if search('load_legacy_settings')['snapshot_id'] != old['snapshot_id']:
            raise ValueError('session changed before explicit reload')
        target.write_bytes(b'intentionally invalid candidate')
        call('reload_snapshot',{'expected_snapshot_id':old['snapshot_id']},'reload_failed')
        search('load_legacy_settings')
        publish = output/'publish.tmp'
        publish.write_bytes(new_snapshot.read_bytes())
        os.replace(publish,target)  # same-directory atomic publication of a completed snapshot
        changed = call('reload_snapshot',{'expected_snapshot_id':old['snapshot_id']})
        if not changed['changed'] or not changed['provider_reused'] or changed['snapshot_id'] == old['snapshot_id']:
            raise ValueError('expected changed snapshot with retained provider')
        call('read_code',old_handle,'snapshot_mismatch')
        call('check_code',old_handle,'snapshot_mismatch')
        current = search('load_current_settings')
        if current['snapshot_id'] != changed['snapshot_id']:
            raise ValueError('search did not use committed snapshot')
        current_handle = handle(current)
        if 'pub fn load_current_settings()' not in call('read_code',current_handle)['text']:
            raise ValueError('updated source not readable')
        if call('check_code',current_handle)['status'] != 'unchanged':
            raise ValueError('updated file hash did not match')
        for _ in range(3):
            semantic.append(call('search_code', {'query':'Load application settings','strategy':'semantic','top_k':1}))
        noop = call('reload_snapshot',{'expected_snapshot_id':current['snapshot_id']})
        if noop['changed'] or noop['snapshot_id'] != current['snapshot_id']:
            raise ValueError('identical snapshot did not preserve handles')
        rss.append({'phase':'after_reload','mcp_pid':session.proc.pid,'mcp_rss_bytes':_rss(session.proc.pid)})
        session.close()
        if session.proc.returncode != 0 or any('cleanup_error' in r for r in session.records):
            raise ValueError('MCP session cleanup failed; retained traces')
        after = {'code': rc._profile_fingerprints(code_binary, encoder_config, None),
                 'mcp': rc._fingerprint_path(mcp_binary),
                 'producers':{Path(m.__file__).name:rc.sha256_file(m.__file__) for m in (rc, ev, mcp)},
                 'runner_sha256':rc.sha256_file(__file__)}
        ev._write_new(output/'inputs-after.json',after)
        if before != after:
            raise ValueError('binary/model/profile identity changed during acceptance')
        summary = {'format_version':1,'status':'passed','condition':'process-encoder' if encoder_config else 'hash-fixture',
                   'mcp_pid':session.proc.pid,'startup_request_ms':startup_ms,
                   'initial_build':initial['report'],'incremental_build':updated['report'],
                   'changed_reload':changed,'noop_reload':noop,'semantic_lifecycle_queries':semantic,
                   'parent_rss_samples':rss,'rpc_records':session.records,'commands':commands,
                   'limitations':['Two-file lifecycle fixture, not retrieval quality, load-test throughput or agent adoption.',
                                  'RSS samples cover only the MCP parent at two instants, not its peak or the encoder/process tree.',
                                  'provider_reused is the server declaration; independent encoder load/PID evidence requires an instrumented local encoder.',
                                  'Index/update CLI encoder lifetimes are separate from the resident MCP encoder.',
                                  'No hard reload time or decoded-memory bound; no controlled speedup claim.']}
        ev._write_new(output/'responses.json',rpc)
        ev._write_new(output/'summary.json',summary)
        members={f.relative_to(output).as_posix():rc.sha256_file(f) for f in sorted(output.rglob('*')) if f.is_file()}
        ev._write_new(output/'manifest.json',{'format_version':1,'files':members})
        return summary
    except Exception as error:
        if session:
            session.close()
        ev._write_new(output/'failure.json',{'error':str(error),'commands':commands,'rpc_records':rpc})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-binary',required=True)
    parser.add_argument('--mcp-binary',required=True)
    parser.add_argument('--encoder-config')
    parser.add_argument('--output',required=True)
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args(argv)
    try:
        summary=run(args.code_binary,args.mcp_binary,args.output,args.encoder_config,args.timeout)
        print(json.dumps({'status':summary['status'],'output':str(args.output)},sort_keys=True))
        return 0
    except (OSError,ValueError,mcp.MCPError) as error:
        print(f'mcp_reload_smoke: {error}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
