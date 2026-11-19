"""Verify trace evidence without models, archive extraction, or archived-code execution."""
from pathlib import Path,PurePosixPath
import hashlib,json,statistics,struct,tarfile
ROOT=Path(__file__).resolve().parent
MODES=('semantic','literal','experimental_rrf','lexical','symbol','hybrid')
STAGES=('not_in_filtered_branch_windows','suppressed_by_overlap','truncated_after_deduplication','retained_below_top10','retained_top10')
def check(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def f32(n):return struct.pack('<f',n)
def close(a,b,msg):check(abs(a-b)<1e-12,msg)
def overlap(a,b):
    intersection=max(0,min(a['end_line'],b['end_line'])-max(a['start_line'],b['start_line'])+1)
    union=a['end_line']-a['start_line']+b['end_line']-b['start_line']+2-intersection
    return a['path']==b['path'] and intersection*5>=union*4

def main():
    inventory={}
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        check(Path(name).name==name and name not in inventory,'publication path')
        check(sha((ROOT/name).read_bytes())==digest,'publication hash: '+name);inventory[name]=digest
    check(set(inventory)|{'SHA256SUMS'}=={p.name for p in ROOT.iterdir()},'publication membership')
    expected=json.loads((ROOT/'evidence-members.json').read_bytes());raw={}
    with tarfile.open(ROOT/'evidence.tar.gz','r:gz') as archive:
        entries=archive.getmembers()
        check(len(entries)==len(expected)==len({m.name for m in entries}),'archive membership')
        check(sum(m.size for m in entries)<=128*1024*1024,'archive size bound')
        for member in entries:
            p=PurePosixPath(member.name)
            check(member.isfile() and not p.is_absolute() and '..' not in p.parts,'archive entry')
            data=archive.extractfile(member).read()
            check(member.name in expected and sha(data)==expected[member.name],'archive hash: '+member.name)
            raw[member.name]=data
    load=lambda name:json.loads(raw[name])
    meta=json.loads((ROOT/'experiment.json').read_bytes());summary=json.loads((ROOT/'trace-summary.json').read_bytes())
    for prefix in ('run','diagnostics'):
        for name,digest in load(prefix+'/manifest.json').items():check(sha(raw[prefix+'/'+name])==digest,'stage manifest')
    recipe=load('original/recipe.json');before=load('run/inputs.json');after=load('run/inputs-after.json')
    check(sha(canonical({k:v for k,v in recipe.items() if k!='fingerprint'}))==recipe['fingerprint']==meta['original_recipe_fingerprint'],'recipe')
    check(before['observed_inputs_before']==after,'before/after input identity')
    check({k:v for k,v in after.items() if k!='binary_sha256'}=={k:v for k,v in recipe['inputs'].items() if k!='binary_sha256'},'unexpected frozen-input change')
    check(after['binary_sha256']==meta['binary_sha256'],'executable hash receipt')
    check(before['original_snapshot_sha256']==meta['preserved_snapshot_sha256']==load('original/manifest.json')['profile-000.arrow'],'snapshot receipt')
    check(sha(raw['inputs/queries.json'])==recipe['inputs']['queries_sha256']==meta['queries_sha256'],'labels')
    for name,digest in meta['producer_sha256'].items():check(sha(raw[name])==digest,'producer')
    for name,digest in meta['engine_sources_sha256'].items():check(sha(raw['producer/'+name])==digest,'engine source')
    check(load('run/execution.json')['status']=='ok','execution failed')
    check(sha(raw['previous/evaluate.stdout'])==load('previous/manifest.json')['evaluate.stdout'],'previous rankings')
    comparison=load('run/comparison.json')
    check(comparison==json.loads((ROOT/'comparison.json').read_bytes()),'comparison copy')
    check(not comparison['changed_metric_display_or_ranking_cases'],'changed old results')
    wrapper=load('run/evaluate.stdout');report=wrapper['evaluation'];old=load('previous/evaluate.stdout')['evaluation']
    diagnostic=load('diagnostics/diagnostics.json')
    check(report['format_version']==4 and old['format_version']==3,'versions')
    check(report['dataset']==old['dataset'] and report['options']==old['options']==meta['native_options'],'frozen native condition')
    check(sha(canonical(wrapper))==diagnostic['source']['canonical_sha256']==summary['source']['canonical_sha256'],'diagnostic input')
    check(sha(raw['run/evaluate.stdout'])==diagnostic['source']['raw_bytes_sha256'],'raw digest')
    cases=report['dataset']['queries'];count=0
    for variant,prior,analysis,public in zip(report['variants'],old['variants'],diagnostic['variants'],summary['variants'],strict=True):
        check(variant['snapshot']==prior['snapshot']==meta['snapshot'],'snapshot metadata')
        all_findings=[];quality_observations={mode:[] for mode in MODES}
        for case,q,b,found in zip(cases,variant['queries'],prior['queries'],analysis['queries'],strict=True):
            check(case['id']==q['id']==b['id']==found['id'],'query order')
            for mode in MODES:
                check(all(q[mode][key]==b[mode][key] for key in ('metrics','top_hits','candidate_ranking')),'old ranking/metric/display changed')
                ranking=q[mode]['candidate_ranking'];labels={(a['path'],a['symbol']) for a in case['relevant']}
                check(len(ranking)<=report['options']['candidate_k'],'final budget')
                matches=[h['rank'] for h in ranking if (h['path'],h['symbol']) in labels]
                first=min(matches,default=None);m=q[mode]['metrics']
                check(m['first_relevant_rank']==first,'first rank')
                close(m['reciprocal_rank'],1/first if first else 0,'MRR')
                for k in (5,10):close(m[f'recall_at_{k}'],sum(rank<=k for rank in matches)/len(labels),'recall')
                winclass='missing_from_exported_window' if len(matches)<len(labels) else 'present_below_top10' if any(rank>10 for rank in matches) else 'all_locations_in_top10'
                quality_observations[mode].append((case['category'],len(matches)/len(labels),len(matches)==len(labels),1/first if first else 0,m['recall_at_5'],m['recall_at_10'],winclass))
                count+=1
            trace=q['hybrid_trace'];scores={};metadata={};source_ranks={}
            check([s['source'] for s in trace['sources']]==['semantic','lexical','symbol'],'source coverage')
            check(trace['candidate_k']==trace['top_k']==report['options']['candidate_k'] and trace['rrf_rank_constant']==60,'trace budgets/policy')
            for source in trace['sources']:
                hits=source['candidates'];check(len(hits)<=trace['candidate_k'],'branch budget')
                check([h['rank'] for h in hits]==list(range(1,len(hits)+1)) and len({h['row'] for h in hits})==len(hits),'branch ranks/duplicates')
                source_ranks[source['source']]={(h['path'],h['symbol']):h['rank'] for h in hits}
                for hit in hits:
                    row=hit['row'];identity=tuple(hit[k] for k in ('path','symbol','start_line','end_line'))
                    check(0<=row<variant['snapshot']['chunks'] and 1<=hit['start_line']<=hit['end_line'],'snapshot bounds')
                    check(row not in metadata or tuple(metadata[row][k] for k in ('path','symbol','start_line','end_line'))==identity,'stable geometry')
                    metadata[row]=hit;scores[row]=scores.get(row,0.)+1/(60+hit['rank'])
            fused=trace['fused'];check([h['row'] for h in fused]==sorted(scores,key=lambda row:(-scores[row],row)),'f64 RRF order')
            for rank,h in enumerate(fused,1):
                check(h['rank']==rank and f32(h['score'])==f32(scores[h['row']]),'RRF score/rank')
                check(all(h[k]==metadata[h['row']][k] for k in ('path','symbol','start_line','end_line')),'fused metadata')
            kept=[];edges={}
            for hit in fused:
                witness=next((k for k in kept if overlap(hit,k)),None) if trace['deduplicate'] else None
                if witness is None:kept.append(hit)
                else:edges[hit['row']]=witness['row']
            dedup=trace['deduplicated']
            check([h['row'] for h in dedup]==[h['row'] for h in kept],'greedy dedup order')
            check(len(trace['suppressions'])==len(edges) and {e['row']:e['kept_row'] for e in trace['suppressions']}==edges,'suppression witnesses')
            for rank,(h,k) in enumerate(zip(dedup,kept,strict=True),1):check(h['rank']==rank and all(h[key]==k[key] for key in ('row','path','symbol','start_line','end_line','score')),'kept identity')
            final=kept[:trace['top_k']]
            check(trace['final_rows']==[h['row'] for h in final]==[h['row'] for h in q['hybrid']['candidate_ranking']],'final truncation')
            for actual,expected in zip(q['hybrid']['candidate_ranking'],final,strict=True):
                check(all(actual[key]==expected[key] for key in ('row','path','symbol')) and f32(actual['score'])==f32(expected['score']),'final identity/score')
            for actual,expected in zip(q['hybrid']['top_hits'],final):
                check(all(actual[key]==expected[key] for key in ('start_line','end_line')),'display geometry')
            rank_maps=[{h['row']:i+1 for i,h in enumerate(array)} for array in (fused,dedup,final)]
            by_label={(h['path'],h['symbol']):row for row,h in metadata.items()};expected_anchors=[]
            for label in case['relevant']:
                identity=(label['path'],label['symbol']);row=by_label.get(identity)
                fr,dr,last=[r.get(row) for r in rank_maps]
                stage='not_in_filtered_branch_windows' if row is None else 'suppressed_by_overlap' if row in edges else 'truncated_after_deduplication' if last is None else 'retained_below_top10' if last>10 else 'retained_top10'
                witness_row=edges.get(row);witness=None if witness_row is None else dict(row=witness_row,path=metadata[witness_row]['path'],symbol=metadata[witness_row]['symbol'],final_rank=rank_maps[2].get(witness_row))
                expected_anchors.append(dict(label,stage=stage,branch_ranks={name:r.get(identity) for name,r in source_ranks.items()},fused_rank=fr,deduplicated_rank=dr,final_rank=last,suppressed_by=witness))
            check(found['anchors']==expected_anchors,'per-label stage attribution');all_findings.append(found)
        for mode,observations in quality_observations.items():
            check(analysis['native_overall'][mode]==variant['overall'][mode],'native quality projection')
            for category,group in [('overall',observations)]+[(cat,[o for o in observations if o[0]==cat]) for cat in variant['by_category']]:
                window=analysis['native_final_windows'][mode]['overall'] if category=='overall' else analysis['native_final_windows'][mode]['by_category'][category]
                native_quality=variant['overall'][mode] if category=='overall' else variant['by_category'][category][mode]
                check(window['query_count']==native_quality['queries']==len(group),'quality denominator')
                for key,index in [('recall_at_window',1),('all_required_at_window',2),('mrr_at_window',3)]:close(window[key],statistics.mean(o[index] for o in group),'window quality mean')
                for key,index in [('mrr',3),('recall_at_5',4),('recall_at_10',5)]:close(native_quality[key],statistics.mean(o[index] for o in group),'native quality mean')
                check(window['failure_class_counts']=={name:sum(o[6]==name for o in group) for name in window['failure_class_counts']},'window classes')
        for category,group in [('overall',all_findings)]+[(c,[q for q in all_findings if q['category']==c]) for c in analysis['by_category']]:
            stats=analysis['overall'] if category=='overall' else analysis['by_category'][category]
            anchors=[a for q in group for a in q['anchors']]
            check(stats['query_count']==len(group) and stats['anchor_judgment_count']==len(anchors),'stage denominators')
            check(stats['anchor_stage_counts']=={stage:sum(a['stage']==stage for a in anchors) for stage in STAGES},'anchor counts')
            check(stats['queries_with_stage']=={stage:sum(any(a['stage']==stage for a in q['anchors']) for q in group) for stage in STAGES},'overlapping query counts')
        check(all(public[key]==analysis[key] for key in ('precision','overall','by_category','native_overall','native_final_windows')),'summary projection')
        gaps=[q for q in analysis['queries'] if any(a['stage'] in STAGES[:3] for a in q['anchors'])]
        check(public['missing_final_window_queries']==gaps,'gap projection')
        check(public['retained_regression']==next(q for q in analysis['queries'] if q['id']=='v2-behavior-biased-first-timing'),'regression projection')
    print(f'OK: {len(entries)} evidence members; {len(cases)} Hybrid traces; {count} unchanged full strategy rankings and metrics.')
    print('RRF order/scores, greedy overlap witnesses, truncation and every labeled first-loss stage independently verified.')
if __name__=='__main__':main()
