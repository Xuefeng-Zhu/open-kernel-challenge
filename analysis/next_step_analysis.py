"""Offline saved-receipt analysis only. No CUDA, network, credentials or candidate edits."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
PROJECTION_REF='75d1dc117aa40c401cddab015402ee75f8a2f934'


def projection_bytes(path):
    return subprocess.check_output(['git','show',PROJECTION_REF+':'+path],cwd=ROOT)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def resource_keys(value,path=''):
    found=[]
    if isinstance(value,dict):
        for k,v in value.items():
            loc=path+'.'+k
            if any(s in k.lower() for s in ['register','spill','shared','occupancy','dram','stall','ptx','ttgir']):
                found.append(loc)
            found.extend(resource_keys(v,loc))
    elif isinstance(value,list):
        for i,v in enumerate(value):found.extend(resource_keys(v,path+f'[{i}]'))
    return found


def main():
    paths=['reviews/attention-v2/E5-official-result.json',
           'reviews/attention-v2/mixed_fa3.py',
           'reviews/attention-v2/official/baseline.py',
           'reviews/attention-v2/attention-cycle-budget.json']
    original={p:(ROOT/p).read_bytes() for p in paths}
    ledger_bytes=projection_bytes('experiments/projection-cycle2-budget.json')
    receipt=json.loads(original[paths[0]]);r=receipt['result']
    assert receipt['id']=='e0b3cd9474084069' and r['passed']
    records=[x for x in r['records'] if x['event']=='benchmark']
    assert len(records)==12
    shapes=[];loads={}
    for x in records:
        c=x['case'];w=c['weight'];assert w==(46 if c['layer']=='sliding' else 16)
        C=x['median_ns']/1000;S=x['baseline_ns']/1000
        assert math.isclose(S/C,x['paired_ratio'],rel_tol=1e-8)
        assert abs(statistics.median(x['runs_ns'])-x['median_raw_ns'])<2
        row=dict(case=c,weight=w,raw_candidate_us=x['median_raw_ns']/1000,
                 raw_sglang_us=x['baseline_raw_ns']/1000,adjusted_candidate_us=C,
                 adjusted_sglang_us=S,paired_sglang_over_candidate=x['paired_ratio'],
                 weighted_signed_gap_us=w*(C-S),runs=x['runs'],
                 replay_overhead_ns=x['replay_overhead']['overhead_ns'])
        shapes.append(row)
        load=loads.setdefault(c['load'],dict(candidate_us=0.,sglang_us=0.,shapes=[]))
        load['candidate_us']+=w*C;load['sglang_us']+=w*S;load['shapes'].append(c['layer'])
    assert len(loads)==6 and all(set(v['shapes'])=={'sliding','full'} for v in loads.values())
    for load in loads.values():
        load['signed_gap_us']=load['candidate_us']-load['sglang_us']
        load['speedup']=load['sglang_us']/load['candidate_us']
    G=math.exp(statistics.mean(math.log(x['speedup']) for x in loads.values()))
    assert math.isclose(G,r['speedup'],rel_tol=1e-10)
    ranking=[]
    for s in shapes:
        old=loads[s['case']['load']]['candidate_us'];new=old-s['weighted_signed_gap_us']
        ranking.append(dict(load=s['case']['load'],layer=s['case']['layer'],
            weighted_gap_us=s['weighted_signed_gap_us'],
            score_if_only_this_shape_matches_sglang=G*(old/new)**(1/6),
            relative_score_change=(old/new)**(1/6)-1))
    ranking.sort(key=lambda x:x['relative_score_change'],reverse=True)
    target=next(s for s in shapes if s['case']['load']=='verify 1' and s['case']['layer']=='sliding')
    load=loads['verify 1'];scenarios=[]
    for name,new in [('target_latency_minus20pct',target['adjusted_candidate_us']*.8),
                     ('target_matches_same_receipt_sglang',target['adjusted_sglang_us']),
                     ('unattainable_zero_target_time_ceiling',0.)]:
        newload=load['candidate_us']-46*(target['adjusted_candidate_us']-new)
        modeled=G*(load['candidate_us']/newload)**(1/6)
        scenarios.append(dict(kind=name,assumed_target_us=new,modeled_verify1_load_us=newload,
            modeled_score=modeled,relative_score_change=modeled/G-1,
            label='Fixed-other-times arithmetic scenario; NOT achieved or forecast timing'))

    projpaths=['results/remote-P2-02-full-composition.json',
               'results/remote-P2-03-pooler-m256-tma.json',
               'results/remote-modal-M34-resource-probe-fixed.json',
               'candidates/projection_full_composition.py',
               'candidates/projection_pooler_m256_tma.py',
               'candidates/projection_pooler_m256_copy.py']
    pb={p:projection_bytes(p) for p in projpaths};best=json.loads(pb[projpaths[0]]);new=json.loads(pb[projpaths[1]])
    assert new['result']['evaluator']==best['result']['evaluator']
    find=lambda d,event:next(x for x in d['result']['records'] if x['event']==event
                            and x.get('case',{}).get('m')==256 and x['case']['layer']=='pooler')
    a,b=find(best,'benchmark'),find(new,'benchmark')
    this_fields=resource_keys(new['result']);assert not this_fields,this_fields
    earlier=json.loads(pb[projpaths[2]])['result']['records']['resource-probe.json']
    this_sha=digest(pb[projpaths[4]])
    assert this_sha not in earlier['source_sha256'].values()
    previous=[dict(runner=c['runner'],source_sha256=earlier['source_sha256'][c['runner']],
                   bm=c['bm'],bn=c['bn'],m=c['m'],registers=c['n_regs_per_thread'],
                   shared_bytes=c['metadata']['shared'],spills=c['n_spills_driver_raw'])
              for c in earlier['compiled'] if c['m']==256]
    controls=[]
    lookup={(x['case']['load'],x['case']['layer']):x for x in best['result']['records'] if x['event']=='benchmark'}
    for x in new['result']['records']:
        if x['event']!='benchmark' or (x['case']['m']==256 and x['case']['layer']=='pooler'):continue
        p=lookup[x['case']['load'],x['case']['layer']]
        controls.append(x['paired_ratio']/p['paired_ratio']-1)
    result=dict(
        method='Saved official receipts only; six weighted loads. All future scenarios explicitly modeled, never achieved.',
        attention=dict(job_id=receipt['id'],source_sha256=digest(original[paths[1]]),shapes=shapes,loads=loads,
            score=G,one_shape_matching_ranking=ranking,target_scenarios=scenarios,
            raw_note='Raw event medians are not official paired ratios; use baseline_ns and median_ns for score arithmetic. No guessed overhead subtraction.'),
        projection=dict(best_job=best['admission']['job_id'],candidate_job=new['admission']['job_id'],
            best_score=best['result']['speedup'],candidate_score=new['result']['speedup'],
            score_ratio=new['result']['speedup']/best['result']['speedup'],
            target_best_adjusted_us=a['median_ns']/1000,target_candidate_adjusted_us=b['median_ns']/1000,
            target_best_raw_us=a['median_raw_ns']/1000,target_candidate_raw_us=b['median_raw_ns']/1000,
            target_reference_normalized_speedup_ratio=b['paired_ratio']/a['paired_ratio'],
            unchanged_max_abs_ratio_change=max(abs(v) for v in controls),
            best_profile=find(best,'profile')['report'],candidate_profile=find(new,'profile')['report'],
            compiled_resources_for_THIS_candidate=dict(registers=None,shared_bytes=None,spills=None,
                ptx=None,sass=None,available_resource_fields=this_fields,
                qualification='Unavailable, not zero. Native path/correctness/timing passed; profiles give GPU durations only.'),
            available_gpu_telemetry=new['result'].get('machine'),
            earlier_different_kernel_resources_not_transferable=previous,
            confounders=['cuBLAS library algorithm versus new hand-written descriptor GEMM; library internal schedule unknown.',
                'Best reads prepared contiguous[K,N] pooler_t; prototype reads contiguous[N,K] original W then transposes each tile.',
                'Prototype64/64/64,2D launch,4warps,3stages; library tile/pipeline/resources are not established by its kernel name.',
                'Both target profiles show one kernel; no separate split-K reduction observed, and new source has none. Internal library details remain opaque.',
                'The same-geometry ordinary-copy control has only CPU evidence, so TMA alone is not isolated.',
                'Separate jobs/seeds and conditions; same-run SGLang normalization reduces but does not eliminate cross-run confounding.'],
            absent_evidence=['Achieved occupancy','DRAM/cache traffic','Stall breakdown','THIS candidate register/shared/spill metrics','PTX/SASS instruction mix']),
        next_experiment=dict(problem='muse_attention_decode',parent='mixed_fa3.py at51d7fca04ccc38a972039ef1b086011d8e456264',
            target='Only batch1 q_len8 sliding-window attention',
            hypothesis='An official-baseline FA3 path can reduce this target normalized latency by at least20%, without changing the other11shapes.',
            isolated_change='Add only this shape dispatch, with official QK norm+RoPE/config-only cos-sin table, cache store, native-required FP8 Q, causal window_size=(2047,0), and final RMSNorm/gate. Leave existing E5 full paths and fallback helpers unchanged.',
            falsification='Reject goal if correctness/cache/replay fails, target reference-normalized latency reduction is below20%, or another shape regresses>5%; overlapping aggregate intervals must remain inconclusive, not a proven score win.',
            unverified=['No new source, native compile or benchmark in this analysis','RoPE-table initialization/memory and graph replay lifetime','Target page64/nonunit scales, fresh-position ABA and cache invariants','Single-run aggregate CI and unchanged-control noise'],
            authorization='Recommendation only. No new benchmark authorization; existing attention1/projection7 budgets untouched.'),
        inputs=dict(root_sha256={p:digest(v) for p,v in original.items()},projection_git_commit=PROJECTION_REF,
                    projection_sha256={p:digest(v) for p,v in pb.items()}))
    (OUT/'next_step_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Offline next-step evidence analysis','',
          'No GPU/check/install/credential/push action. Candidate code and budget bytes unchanged.',
          'Input receipts: attention e0b3cd9474084069; projection best48b1357016134e52 vs6799753227ed463e.',
          'All source hashes and exact pinned Git input commit are in next_step_evidence.json.','',
          '## Attention: all12shapes (microseconds)','',
          'C=our candidate, S=same-receipt SGLang. Raw is unadjusted event median;',
          'official adjusted C/S and paired S/C determine score. Raw medians must not',
          'be averaged or divided to substitute for official paired/calibrated ratios.','',
          '| Load | Layer | Raw C / S | Adjusted C / S | Paired S/C | w×(C−S) µs |',
          '|---|---|---:|---:|---:|---:|']
    for s in shapes:
        text.append(f"| {s['case']['load']} | {s['case']['layer']} | {s['raw_candidate_us']:.3f} / {s['raw_sglang_us']:.3f} | {s['adjusted_candidate_us']:.3f} / {s['adjusted_sglang_us']:.3f} | {s['paired_sglang_over_candidate']:.6f} | {s['weighted_signed_gap_us']:+.3f} |")
    text+=['','Negative gap means we win. Sliding weight46, full weight16; all rows are measured.',
           '', '## Six loads and exact score arithmetic','',
           '`C_load=46*C_sliding+16*C_full; S_load=46*S_sliding+16*S_full`',
           '`G=exp(mean(log(S_load/C_load)))=0.8919466283178157`','',
           '| Load | Candidate load µs | SGLang load µs | Signed gap µs | S/C |',
           '|---|---:|---:|---:|---:|']
    for name,l in loads.items():text.append(f"| {name} | {l['candidate_us']:.3f} | {l['sglang_us']:.3f} | {l['signed_gap_us']:+.3f} | {l['speedup']:.6f} |")
    text+=['', 'Largest load deficit: verify1,1616.784µs. Its sliding component969.701µs',
           'is the largest single-path deficit; full verify1 adds647.083µs. Full decode16',
           'adds891.472µs, but lives in a larger load. Equal-load geometric scoring means',
           'absolute gap alone is insufficient: matching one path gives relative score',
           'opportunities6.091% (verify1 sliding),3.774% (verify1 full),3.479% (decode16 full).',
           '', '## One-target scenarios, NOT achieved speedups','',
           '`verify1 current =46*43.976461+16*76.490695=3246.768337µs`',
           '`new_score=G*(3246.768337/new_verify1_load)^(1/6)`','',
           '| Assumption for verify1 sliding only | Target µs | New load µs | Modeled score | Relative change |',
           '|---|---:|---:|---:|---:|']
    for s in scenarios:text.append(f"| {s['kind']} | {s['assumed_target_us']:.6f} | {s['modeled_verify1_load_us']:.6f} | {s['modeled_score']:.9f} | {s['relative_score_change']*100:+.4f}% |")
    text+=['','All other11shape times and all SGLang times are frozen. Matching SGLang is a',
           'conditional scenario, not a hard speed limit or performance forecast. The zero-',
           'time row is an unattainable algebraic ceiling for changing only this path.',
           'For the matching scenario:46*22.896000+16*76.490695=2277.067134µs;',
           '0.891946628*(3246.768337/2277.067134)^(1/6)=0.946276626 (+6.091%).',
           '20% latency reduction gives35.181169µs,2842.184895µs/load,0.911952106 (+2.243%).',
           'Neither scenario by itself establishes an overall win over SGLang.','',
           '## Projection: what THIS failed candidate establishes','',
           'Official adjusted target30.262861→54.191390µs; normalized relative speedup',
           '0.548359; overall1.148649952→1.112041018 (-3.187%).29controls move at most4.691%.',
           'Both jobs passed9prerequisites+30shapes. Same evaluator/image, separate runs;',
           'this is strong rejection evidence for the whole implementation, not TMA alone.','',
           'Target profiles: best one library kernel29.2µs; candidate one _pooler_pipeline',
           '56.5µs. These are separate single instrumented samples, not steady-state',
           'medians. They confirm the new GPU path executed. Host descriptor creation',
           'and compilation are not replayed GPU-kernel time; neither is an established',
           'explanation for the measured kernel regression.','',
           '**THIS candidate registers/shared memory/spill counts: unavailable, not zero.**',
           'No compiled-resource fields, PTX/SASS, hardware counters, actual HBM traffic,',
           'cache hit rate, achieved occupancy or stalls appear in its saved receipt.',
           'GPU identity/driver/clock/temperature telemetry is available; it does not',
           'provide those per-kernel performance counters or compiled resources.',
           'Earlier M34 shows95regs/40960Bshared/0spills for BM32/BN128 and45regs/',
           '16384Bshared/0spills for BM32/BN32; their source hashes differ from44312859.',
           'Those values cannot be assigned to current BM64/BN64 TMA code or cuBLAS.','',
           'Confounders: new backend/geometry/pipeline versus opaque cuBLAS algorithm;',
           'original[N,K] descriptors versus prepared contiguous[K,N] library layout;',
           'compiler scheduling/accumulation order/resources. Both target profiles show',
           'one kernel and no separate split-K reduce; m128 reduction evidence is not',
           'evidence for m256. The ordinary-copy same-geometry control was never GPU-',
           'tested. No conclusion that TMA itself, spills or bandwidth caused the loss.','',
           '## One recommended next experiment card (not implemented or run)','',
           '- Problem/parent: attention, adopted mixed_fa3.py SHA708d3f06/source51d7fca.',
           '- Hypothesis: official FA3 can cut batch1/L8/sliding normalized latency≥20%.',
           '- Change only this dispatch. Use official QK norm+RoPE, correct new-cache',
           '  writes/scales, native-required FP8 query, causal window=(2047,0), and output',
           '  RMSNorm/gate. Preserve all existing E5 paths/helpers, no competitor code.',
           '- Expected affected shapes:1of12; every other11 is a control. Config-only',
           '  RoPE table follows official baseline (~512MiB), not cached input/output.',
           '- Value:20% cut models+2.243% score; matching reference models+6.091%.',
           '- Reject predeclared goal if target cut<20%, correctness/cache/replay fails',
           '  or another path regresses>5%. If global CI overlaps, global improvement',
           '  remains inconclusive even when target improves. No guaranteed score.',
           '- Before any future GPU admission: preserve BF16/FP8 order, no host value',
           '  sync, verify RoPE positions, page64/nonunit scales, live cache row0,',
           '  untouched rows, CPU argument/dispatch tests, fresh same-address A/B/A.',
           '  Check actual native graph/table lifetime and all official cases if approved.',
           '- Source profile alone does not diagnose registers/occupancy here either.',
           '- Only one attention check remains: complete source/CPU review before any',
           '  future admission; no spare correction check is assumed. This task authorizes',
           '  analysis only, so the recommendation is not an instruction to launch it.',
           '  Attention1/projection7 user checks remain untouched by this analysis.',
           '', '## Exact saved sources and reproduction', '',
           '- Attention receipt: `reviews/attention-v2/E5-official-result.json`, job',
           '  `e0b3cd9474084069`; candidate `reviews/attention-v2/mixed_fa3.py`, source',
           '  commit `51d7fca04ccc38a972039ef1b086011d8e456264`.',
           '- Official implementation contract: `reviews/attention-v2/official/baseline.py`.',
           '- Projection inputs are read with `git show` from immutable commit',
           '  `75d1dc117aa40c401cddab015402ee75f8a2f934`: receipts',
           '  `results/remote-P2-02-full-composition.json` and',
           '  `results/remote-P2-03-pooler-m256-tma.json`; candidate sources',
           '  `candidates/projection_full_composition.py`,',
           '  `candidates/projection_pooler_m256_tma.py`, and',
           '  `candidates/projection_pooler_m256_copy.py`.',
           '- Earlier, different-kernel resource receipt: the same pinned commit\'s',
           '  `results/remote-modal-M34-resource-probe-fixed.json`.',
           '- `next_step_evidence.json` retains full-precision numbers, profile text,',
           '  telemetry, source SHA256s, and every input SHA256.', '',
           'Reproduce offline: `.venv/bin/python analysis/next_step_analysis.py`.',
           'Checks: twelve shape ratios, raw sample medians, six weighted loads, exact',
           'official score, evaluator identity, resource-source identity, and unchanged',
           'candidate/budget bytes are asserted. No native kernels execute.']
    (OUT/'NEXT_STEP_EVIDENCE.md').write_text('\n'.join(text)+'\n')
    assert all((ROOT/p).read_bytes()==v for p,v in original.items())
    assert projection_bytes('experiments/projection-cycle2-budget.json')==ledger_bytes
    print(f'Reproduced G={G:.12f}; wrote12shape/six-load tables and one-target models. No candidate/budget changes.')


if __name__=='__main__':main()
