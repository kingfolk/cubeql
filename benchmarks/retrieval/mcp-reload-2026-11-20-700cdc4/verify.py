"""Verify saved lifecycle receipts without models, extraction, or archived execution."""
import hashlib
import json
from pathlib import Path
import tarfile


def verify(root):
    root = Path(root)
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ',1)
        assert Path(name).name == name
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    members = json.loads((root/'evidence-members.json').read_text())
    with tarfile.open(root/'evidence.tar.gz') as archive:
        inventory = archive.getmembers()
        assert len({m.name for m in inventory}) == len(inventory)
        assert {m.name for m in inventory} == set(members)
        data = {}
        for member in inventory:
            assert member.isfile() and not member.name.startswith('/') and '..' not in member.name.split('/')
            raw = archive.extractfile(member).read()
            assert len(raw) == members[member.name]['bytes']
            assert hashlib.sha256(raw).hexdigest() == members[member.name]['sha256'], member.name
            data[member.name] = raw
    def load(name):
        return json.loads(data[name])
    for label in ['hash-attempt1','process-attempt1']:
        prefix = 'runs/'+label+'/'
        manifest = load(prefix+'manifest.json')
        assert {name[len(prefix):] for name in data if name.startswith(prefix)} == set(manifest['files']) | {'manifest.json'}
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(data[prefix+name]).hexdigest() == digest
        before, after = load(prefix+'inputs-before.json'), load(prefix+'inputs-after.json')
        assert before == after
        assert before['runner_sha256'] == hashlib.sha256(data['producers/tools/mcp_reload_smoke.py']).hexdigest()
        for name, digest in before['producers'].items():
            assert digest == hashlib.sha256(data['producers/tools/'+name]).hexdigest()
        summary = load(prefix+'summary.json')
        assert summary['status'] == 'passed'
        assert summary['initial_build']['embedded'] == 2
        assert summary['incremental_build']['embedded'] == 1
        assert summary['incremental_build']['reused'] == 1
        assert summary['changed_reload']['changed'] is True
        assert summary['changed_reload']['provider_reused'] is True
        old, new = summary['changed_reload']['previous_snapshot_id'], summary['changed_reload']['snapshot_id']
        assert old != new
        assert summary['noop_reload']['changed'] is False and summary['noop_reload']['snapshot_id'] == new
        assert len(summary['semantic_lifecycle_queries']) == 6
        assert [q['snapshot_id'] for q in summary['semantic_lifecycle_queries']] == [old]*3+[new]*3
        assert all(c['status']=='ok' and c['returncode']==0 for c in summary['commands'])
        requests = [json.loads(line) for line in data[prefix+'mcp.stdin'].splitlines()]
        outputs = [json.loads(line) for line in data[prefix+'mcp.stdout'].splitlines()]
        replies = {r['id']:r['result'] for r in outputs if 'id' in r}
        calls = [q for q in requests if q['method']=='tools/call']
        responses = load(prefix+'responses.json')
        assert len(calls)==len(responses)
        assert [replies[q['id']] for q in calls] == [r['raw_result'] for r in responses]
        errors = []
        reads = []
        reloads = []
        for call, record in zip(calls,responses):
            assert call['params']['name']==record['name']
            result = record['raw_result']; value = result['structuredContent']
            text = [json.loads(c['text']) for c in result['content'] if c['type']=='text']
            assert text == [value]
            if result['isError']:
                errors.append(value['error']['code'])
            elif record['name']=='read_code':
                reads.append(value)
            elif record['name']=='reload_snapshot':
                reloads.append(value)
        assert errors == ['reload_failed','snapshot_mismatch','snapshot_mismatch']
        assert len(reads)==2 and 'load_legacy_settings' in reads[0]['text'] and 'load_current_settings' in reads[1]['text']
        assert reads[0]['snapshot_id']==old and reads[1]['snapshot_id']==new
        assert reloads == [summary['changed_reload'],summary['noop_reload']]
        traces=[json.loads(line) for line in data[prefix+'mcp.stderr'].splitlines() if line.startswith(b'{')]
        ready=[r for r in traces if r.get('event')=='ready']
        assert len(ready)==1 and ready[0]['snapshot_id']==old and ready[0]['reload_enabled'] is True
        finished=[r for r in traces if r.get('event')=='request_finished']
        assert len(finished)==len(calls)
        assert [r['snapshot_id'] for r in finished if r.get('method')=='reload_snapshot' and r.get('error') is None] == [new,new]
    experiment = json.loads((root/'experiment.json').read_text())
    assert experiment['model'] == load('runs/process-attempt1/index.stdout')['snapshot']['embedding']
    assert experiment['mcp_source_revision'].startswith('700cdc4')
    observations = load('observations/encoder-observations.json')
    assert observations == json.loads((root/'encoder-observations.json').read_text())
    summary=load('runs/process-attempt1/summary.json')
    events=[json.loads(line) for line in data['runs/process-attempt1/encoder-events.jsonl'].splitlines()]
    active=[e for e in events if e['parent_pid']==summary['mcp_pid']]
    ready=[e for e in active if e['event']=='ready']; results=[e for e in active if e['event']=='result']
    assert len(ready)==1 and len(results)==6 and len({e['pid'] for e in active})==1
    assert all(e['role']=='query' for e in results)
    assert observations['encoder_pid']==ready[0]['pid']
    assert observations['mcp_encoder_ready_events']==len(ready)
    assert observations['mcp_query_encoding_events']==len(results)
    assert observations['total_encoder_ready_events_including_separate_index_update']==sum(e['event']=='ready' for e in events)==3
    assert observations['encoder_peak_rss_bytes_first_query']==results[0]['peak_rss_bytes']
    assert observations['encoder_peak_rss_bytes_last_query']==results[-1]['peak_rss_bytes']
    assert observations['query_encoding_ms']==[e['encoding_ms'] for e in results]
    assert observations['changed_reload']==summary['changed_reload']
    assert observations['parent_rss_samples']==summary['parent_rss_samples']
    print('Verified 52 archived artifacts, both lifecycle runs, incremental reuse, failure retention, snapshot handles and encoder PID/load receipts.')


if __name__=='__main__':
    verify(Path(__file__).resolve().parent)
