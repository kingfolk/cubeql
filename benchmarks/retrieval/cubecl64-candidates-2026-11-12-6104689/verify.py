"""Verify retained-window evidence without models or archive extraction."""
from pathlib import Path, PurePosixPath
import hashlib,json,math,statistics,tarfile
ROOT=Path(__file__).resolve().parent
MODES=('semantic','literal','experimental_rrf','lexical','symbol','hybrid')
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def check(ok,message):
    if not ok:raise ValueError(message)
def close(a,b,message):check(abs(a-b)<1e-12,message)
def main():
    inventory={}
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        check(Path(name).name==name and name not in inventory,'publication path')
        check(sha((ROOT/name).read_bytes())==digest,'publication hash: '+name)
        inventory[name]=digest
    check(set(inventory)|{'SHA256SUMS'}=={p.name for p in ROOT.iterdir()},'publication membership')
    expected=json.loads((ROOT/'evidence-members.json').read_bytes());raw={}
    with tarfile.open(ROOT/'evidence.tar.gz','r:gz') as archive:
        entries=archive.getmembers()
        check(len(entries)==len(expected)==len({m.name for m in entries}),'archive membership')
        check(sum(m.size for m in entries)<=64*1024*1024,'archive uncompressed size')
        for member in entries:
            p=PurePosixPath(member.name)
            check(member.isfile() and not p.is_absolute() and '..' not in p.parts,'archive path/type')
            content=archive.extractfile(member).read()
            check(member.name in expected and sha(content)==expected[member.name],'archive hash: '+member.name)
            raw[member.name]=content
    load=lambda name:json.loads(raw[name])
    meta=json.loads((ROOT/'experiment.json').read_bytes())
    top=json.loads((ROOT/'window-summary.json').read_bytes())
    for prefix in ('run','diagnostics'):
        for name,digest in load(prefix+'/manifest.json').items():
            check(sha(raw[prefix+'/'+name])==digest,'stage manifest: '+name)
    check(load('run/execution.json')['status']=='ok','failed execution')
    before=load('run/inputs.json');after=load('run/inputs-after.json')
    check(before['observed_inputs_before']==after,'inputs changed')
    recipe=load('original/recipe.json')
    check(sha(canonical({k:v for k,v in recipe.items() if k!='fingerprint'}))==recipe['fingerprint']==meta['original_recipe_fingerprint'],'recipe fingerprint')
    check({k:v for k,v in after.items() if k!='binary_sha256'}=={k:v for k,v in recipe['inputs'].items() if k!='binary_sha256'},'unintended model/source/labels/runner change')
    check(after['binary_sha256']==meta['binary_sha256'],'new executable receipt')
    check(before['original_snapshot_sha256']==meta['preserved_snapshot_sha256']==load('original/manifest.json')['profile-000.arrow'],'snapshot receipt')
    check(sha(raw['inputs/queries.json'])==meta['queries_sha256']==recipe['inputs']['queries_sha256'],'original labels')
    check(sha(raw['original/profile-000-evaluate.stdout'])==load('original/manifest.json')['profile-000-evaluate.stdout'],'original report')
    for name,digest in meta['producer_sha256'].items():check(sha(raw[name])==digest,'producer hash')
    wrapper=load('run/evaluate.stdout'); native=wrapper['evaluation']
    old=load('original/profile-000-evaluate.stdout')['evaluation']
    diagnostic=load('diagnostics/diagnostics.json')
    check(native['format_version']==3 and native['options']==old['options']==meta['native_options'],'version/options')
    check(native['dataset']==old['dataset'],'dataset echo')
    check(sha(canonical(wrapper))==diagnostic['source']['canonical_sha256']==top['source']['canonical_sha256'],'diagnostic input')
    check(sha(raw['run/evaluate.stdout'])==diagnostic['source']['raw_bytes_sha256'],'raw report digest')
    check(load('run/comparison.json')==json.loads((ROOT/'comparison.json').read_bytes()),'comparison copy')
    check(not load('run/comparison.json')['changed_metric_or_display_cases'],'retained regression delta')
    cases=native['dataset']['queries'];count=0
    for variant,original,analysis in zip(native['variants'],old['variants'],diagnostic['variants'],strict=True):
        check(variant['snapshot']==original['snapshot']==meta['snapshot'],'snapshot metadata')
        check([q['id'] for q in variant['queries']]==[q['id'] for q in cases],'query coverage/order')
        for mode in MODES:
            observations=[]
            for case,query,prior,detail in zip(cases,variant['queries'],original['queries'],analysis['candidate_window'][mode]['queries'],strict=True):
                strategy=query[mode];ranking=strategy['candidate_ranking']; metrics=strategy['metrics']
                check(metrics==prior[mode]['metrics'] and strategy['top_hits']==prior[mode]['top_hits'],'old metric/display changed')
                check(len(ranking)<=native['options']['candidate_k'],'candidate budget')
                check([h['rank'] for h in ranking]==list(range(1,len(ranking)+1)),'rank order')
                check(len({h['row'] for h in ranking})==len(ranking),'duplicate rows')
                check(all(type(h['row']) is int and 0<=h['row']<variant['snapshot']['chunks'] and math.isfinite(h['score']) for h in ranking),'candidate bounds/score')
                displayed=strategy['top_hits']
                check(len(displayed)==min(10,len(ranking)),'display count')
                check(all(all(h[k]==r[k] for k in ('row','path','symbol','score')) for h,r in zip(displayed,ranking)),'display prefix')
                labels={(a['path'],a['symbol']) for a in case['relevant']}
                ranks={(h['path'],h['symbol']):h['rank'] for h in ranking if (h['path'],h['symbol']) in labels}
                first=min(ranks.values(),default=None);rr=1/first if first else 0
                check(metrics['first_relevant_rank']==first,'full first relevant rank')
                close(metrics['reciprocal_rank'],rr,'independent reciprocal rank')
                for k in (5,10):close(metrics[f'recall_at_{k}'],sum(rank<=k for rank in ranks.values())/len(labels),'independent recall')
                window=len(ranks)/len(labels);complete=len(ranks)==len(labels)
                cls='missing_from_exported_window' if not complete else 'present_below_top10' if any(r>10 for r in ranks.values()) else 'all_locations_in_top10'
                check(detail['id']==case['id'] and detail['failure_class']==cls,'gap class')
                check(detail['anchor_ranks']==[dict(path=a['path'],symbol=a['symbol'],rank=ranks.get((a['path'],a['symbol']))) for a in case['relevant']],'anchor ranks')
                close(detail['recall_at_window'],window,'query window recall')
                observations.append((case['category'],window,complete,rr,cls));count+=1
            for category,group in [('overall',observations)]+[(c,[o for o in observations if o[0]==c]) for c in variant['by_category']]:
                summary=analysis['candidate_window'][mode]['overall' if category=='overall' else 'by_category']
                if category!='overall':summary=summary[category]
                check(summary['query_count']==len(group),'window denominator')
                for key,index in [('recall_at_window',1),('all_required_at_window',2),('mrr_at_window',3)]:close(summary[key],statistics.mean(o[index] for o in group),'window aggregate')
                check(summary['failure_class_counts']=={c:sum(o[4]==c for o in group) for c in summary['failure_class_counts']},'class counts')
            check(top['strategies'][mode]=={k:v for k,v in analysis['candidate_window'][mode].items() if k!='queries'},'summary projection')
        check(top['hybrid_incomplete_top10']==[q for q in analysis['candidate_window']['hybrid']['queries'] if q['failure_class']!='all_locations_in_top10'],'incomplete projection')
    print(f'OK: {len(entries)} evidence members; {len(cases)} queries; {count} strategy cases with independently recomputed full-window MRR and recall.')
    print('All original metrics and ten-hit displays match; internal candidate-generation causes are not established.')
if __name__=='__main__':main()
