"""Source-bound official result analysis; cross-run normalization is not a paired A/B."""
import json, math
from pathlib import Path
from assess_p062 import records, normalized_latency
ROOT=Path(__file__).resolve().parent
EXP='P2-11-p065-explicit-consumer'
SHA='9379aaaa9fb36761b5ec1b5c46ef12bb9b5b918ddaa4085d1c9100a7276810ff'
SELECTED_SHA='001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e'

def assess(candidate, selected, old_encoded):
    if candidate['admission']['submission_sha256']!=SHA or candidate['admission']['mode']!='check':raise ValueError('Source/mode mismatch')
    if candidate['status'] not in {'done','failed','rejected','duplicate'}:raise ValueError('Pending is not a result')
    if selected['admission']['job_id']!='41ba778763a5433b' or selected['admission']['submission_sha256']!=SELECTED_SHA:raise ValueError('Selected identity mismatch')
    if old_encoded['admission']['job_id']!='128378586ebe4730':raise ValueError('Historical identity mismatch')
    out=dict(experiment=EXP,job_id=candidate['admission']['job_id'],source_sha256=SHA,
             selected_job=selected['admission']['job_id'],selected_preserved=True,
             matched_gluon_raw_gpu_timing=False,dedicated_aba_concurrency_gpu=False)
    result=candidate['result']
    if result.get('passed') is not True:
        return dict(**out,passed=False,decision='Reject native candidate: compilation/correctness not passed. No performance conclusion.',error=result.get('error'))
    if candidate['prerequisite_tests']!={'passed':9,'total':9}:raise ValueError('Missing prerequisites')
    cr,sr,er=records(candidate),records(selected),records(old_encoded)
    if not set(cr)==set(sr)==set(er):raise ValueError('Case set mismatch')
    rows=[];loads={}
    for key,c in cr.items():
        s,e=sr[key],er[key];case=c['case']
        if case['weight']!=62 or s['case']['weight']!=62:raise ValueError('Weight mismatch')
        target=case['layer']=='pooler' and case['m'] in (8,16,32)
        ratio=normalized_latency(c['paired_ratio'],s['paired_ratio'])
        row=dict(load=key[0],layer=key[1],m=case['m'],target=target,
                 candidate_calibrated_us=c['median_ns']/1000,selected_calibrated_us=s['median_ns']/1000,
                 candidate_raw_graph_us=c['median_raw_ns']/1000,selected_raw_graph_us=s['median_raw_ns']/1000,
                 candidate_vs_selected_normalized_latency=ratio,
                 candidate_vs_p043_normalized_latency=normalized_latency(c['paired_ratio'],e['paired_ratio']),
                 candidate_paired_sglang_speedup=c['paired_ratio'])
        rows.append(row)
        ld=loads.setdefault(key[0],dict(selected_weighted_us=0,rebased_candidate_weighted_us=0,target_only_weighted_us=0))
        base=s['median_ns']/1000*62
        ld['selected_weighted_us']+=base;ld['rebased_candidate_weighted_us']+=base*ratio
        ld['target_only_weighted_us']+=base*(ratio if target else 1)
    targets=sorted((r for r in rows if r['target']),key=lambda r:r['m'])
    controls=[r for r in rows if not r['target']]
    if len(targets)!=3 or len(controls)!=27:raise ValueError('Wrong target/control count')
    profiles=[r for r in result['records'] if r.get('event')=='profile' and r['case']['layer']=='pooler' and r['case']['m'] in (8,16,32)]
    executed=sorted(p['case']['m'] for p in profiles if '_pooler_codec_partial' in p['report'] and '_reduce' in p['report'])==[8,16,32]
    gm=math.exp(sum(math.log(r['candidate_vs_selected_normalized_latency']) for r in targets)/3)
    drift=[r for r in controls if abs(r['candidate_vs_selected_normalized_latency']-1)>.05]
    same_evaluator=result['evaluator']==selected['result']['evaluator']
    score_ratio=result['speedup']/selected['result']['speedup']
    screen=executed and all(r['candidate_vs_selected_normalized_latency']<1 for r in targets) and gm<=.95 and score_ratio>=1.01 and not drift and same_evaluator
    if not executed:decision='No profile-confirmed codec route; performance mechanism remains unresolved. Preserve P046.'
    elif any(r['candidate_vs_selected_normalized_latency']>1.05 for r in targets):decision='Reject P065: at least one intended target materially regresses versus P046. Compiler capability is real but this consumer is not competitive.'
    elif screen:decision='Promising only: passes predeclared screen, requires independent paired confirmation and matched raw timing before promotion.'
    else:decision='Does not pass the predeclared promotion screen; preserve P046 and do not infer a win from aggregate point score.'
    for l in loads.values():
        l['rebased_ratio']=l['selected_weighted_us']/l['rebased_candidate_weighted_us']
        l['target_only_modeled_ratio']=l['selected_weighted_us']/l['target_only_weighted_us']
    modeled=math.exp(sum(math.log(x['target_only_modeled_ratio']) for x in loads.values())/6)
    out.update(passed=True,score=result['speedup'],score_ci=result['speedup_ci'],score_ratio_vs_selected=score_ratio,
               target_geomean_latency_ratio=gm,targets=targets,all30shapes=rows,controls_over5pct=drift,
               same_evaluator=same_evaluator,codec_execution_verified=executed,promotion_screen=screen,
               weighted_loads=loads,target_only_modeled_score_ratio=modeled,decision=decision,
               qualifications=['Ratios <1 mean faster latency; reference-normalized historical comparisons are not same-input paired trials.',
                 'The target-only score ratio holds P046 controls constant and is a model, not an achieved score.',
                 'Offline compiler artifacts are not proven identical to judge codegen. No hardware counter causality.',
                 'Gluon direct loads change the prior pipeline; old compressed comparison cannot isolate shared conversion.',
                 'Official checks cover their graph/fresh-input suite; no dedicated all-bit GPU, ABA or overlapping-stream proof.'])
    return out

if __name__=='__main__':
    c=json.loads((ROOT/f'results/remote-{EXP}.json').read_text())
    s=json.loads((ROOT/'results/remote-P2-09-pooler-column-batch2-hybrid.json').read_text())
    e=json.loads((ROOT/'results/remote-P2-07-pooler-lossless-codec.json').read_text())
    out=assess(c,s,e)
    (ROOT/'results/projection-p065-assessment.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['job_id','passed','decision']}))
