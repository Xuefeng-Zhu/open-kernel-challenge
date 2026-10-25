"""Recompute the bounded representative result; pure stdlib, no GPU/network.

No performance summary is admitted unless the complete recorded run and all
18 official-oracle replays pass, with all planned paired samples validated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
NAMES=['fa3','bf16_custom']
EXPECTED_CASES=['full_verify16_page64_boundary','full_verify16_scored_family','full_verify32_scored_family']
FROZEN={
    'worker.py':'3818221cb069ed1d605c37871183aebc5cf711bdcf22fd28fcc6768b3dc120c7',
    'candidatecustomhelper.py':'59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82',
    'candidate.py':'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',
    'modal_worker.py':'9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py':'386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py':'9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
LOCAL={'worker.py':HERE/'worker.py','candidatecustomhelper.py':HERE/'candidatecustomhelper.py',
    'candidate.py':HERE.parent/'probe/candidate.py','modal_worker.py':HERE.parents[1]/'modal_worker.py',
    'official/reference.py':HERE.parents[1]/'official/reference.py',
    'official/checks.py':HERE.parents[1]/'official/checks.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_positive(x):
    return isinstance(x,(int,float)) and math.isfinite(x) and x>0


def close(x,y):
    return isinstance(x,(int,float)) and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-9)


def summarize(samples):
    ratios=[r['raw_us']['fa3']/r['raw_us']['bf16_custom'] for r in samples]
    pairs=[math.sqrt(ratios[i]*ratios[i+1]) for i in range(0,len(ratios),2)]
    ratio=statistics.median(pairs)
    return dict(raw_median_us={name:statistics.median(r['raw_us'][name] for r in samples) for name in NAMES},
        raw_min_max_us={name:[min(r['raw_us'][name] for r in samples),max(r['raw_us'][name] for r in samples)] for name in NAMES},
        fa3_over_bf16_custom_ratios=ratios,reverse_order_pair_geomeans=pairs,
        median_reverse_order_pair_ratio=ratio,
        paired_candidate_latency_change_fraction=1/ratio-1,
        custom_faster_measured_comparisons=sum(x>1 for x in ratios),
        qualification='Within-session whole-call measurements, two input versions, six reverse-order pair ratios; no confidence interval or official score')


def analyze(data,receipt,result_path,receipt_path=HERE/'launch-receipt.json'):
    issues=[]
    provenance={}
    for name,expected in FROZEN.items():
        actual=sha(LOCAL[name])
        provenance[name]=dict(expected_sha256=expected,local_sha256=actual,
            result_sha256=data.get('payload_sha256',{}).get(name),
            launch_sha256=receipt.get('payload_sha256',{}).get(name))
        if any(x!=expected for x in (actual,data.get('payload_sha256',{}).get(name),receipt.get('payload_sha256',{}).get(name))):
            issues.append('Frozen payload mismatch or missing evidence: '+name)
    if data.get('source_sha256')!=FROZEN['candidate.py']:
        issues.append('Result source hash mismatch')
    if receipt.get('report_sha256')!=sha(result_path):
        issues.append('Result transport receipt hash mismatch or unavailable')
    expected_order=[(label,name) for label in ('A1','B','A2') for name in NAMES]
    cases=data.get('cases',[])
    oracle_count=0
    checked_count=0
    returns_count=0
    failures=[]
    case_audits=[]
    for case in cases:
        name=case.get('case',{}).get('name')
        rows=case.get('records',[])
        input_lengths=case.get('input_lengths',[])
        if len(input_lengths)!=2 or input_lengths[0]==input_lengths[1]:
            issues.append('Missing/unchanged two length versions: '+str(name))
        positions=[(r.get('label'),r.get('runner')) for r in rows]
        if positions!=expected_order and data.get('status')=='complete':
            issues.append('ABA replay sequence incomplete or wrong: '+str(name))
        for r in rows:
            oracle_count+=r.get('official_oracle_called') is True
            passed=all(r.get(k) is True for k in ('passed','official_oracle_called','stable_addresses','immutable_inputs_and_untouched_rows'))
            if r.get('label')=='A2':
                passed=passed and r.get('exact_output_newkv_return') is True
                returns_count+=r.get('exact_output_newkv_return') is True
            version=1 if r.get('label')=='B' else 0
            if len(input_lengths)==2:
                passed=passed and r.get('actual_lengths')==input_lengths[version]
            checked_count+=passed
            if not passed:
                failures.append(dict(case=name,arm=r.get('runner'),replay=r.get('label'),
                    oracle_message=r.get('oracle_message'),record=r))
        case_audits.append(dict(case=name,passed=case.get('passed') is True,records=len(rows),
            oracle_calls=sum(r.get('official_oracle_called') is True for r in rows),
            errors=[dict(arm=r.get('runner'),replay=r.get('label'),message=r.get('oracle_message')) for r in rows],
            input_lengths=input_lengths,page_table_shape=case.get('page_table_shape')))
    complete_cases=[c.get('case',{}).get('name') for c in cases]==EXPECTED_CASES
    oracle_gate=complete_cases and oracle_count==18 and checked_count==18 and returns_count==6 and all(c.get('passed') is True for c in cases)
    if data.get('correctness_complete') is True and not oracle_gate:
        issues.append('Correctness-complete flag contradicts 18-oracle/ABA audit')
    timing_audits=[]
    expected_timing_names=EXPECTED_CASES[1:]
    for timing in data.get('timings',[]):
        name=timing.get('case',{}).get('name')
        samples=timing.get('samples',[])
        warmups=timing.get('warmups',[])
        valid=len(samples)==12 and len(warmups)==4
        timed_checks=0
        for row in warmups+samples:
            i=row.get('index',-1)
            expected=NAMES if i%2==0 else list(reversed(NAMES))
            okay=row.get('order')==expected and row.get('input_version')==(i//2)%2
            okay=okay and set(row.get('raw_us',{}))==set(NAMES)
            for arm in NAMES:
                check=row.get('checks',{}).get(arm,{})
                part=all(check.get(k) is True for k in ('passed','stable_addresses','immutable_inputs_and_untouched_rows','exact_output_newkv_return'))
                part=part and check.get('official_oracle_called') is False and finite_positive(row.get('raw_us',{}).get(arm))
                okay=okay and part
                timed_checks+=part
                if not part:
                    failures.append(dict(case=name,arm=arm,timing_sample=i,record=check))
            valid=valid and okay
        valid=valid and [r.get('index') for r in warmups]==list(range(4)) and [r.get('index') for r in samples]==list(range(4,16))
        row=dict(case=name,complete_and_valid=valid,warmup_two_arm_samples=len(warmups),
            measured_two_arm_samples=len(samples),validated_replays=timed_checks)
        if valid and oracle_gate and data.get('status')=='complete' and data.get('passed') is True and data.get('timing_complete') is True:
            summary=summarize(samples)
            saved=timing.get('summary',{})
            for arm in NAMES:
                if not close(saved.get('raw_median_us',{}).get(arm),summary['raw_median_us'][arm]):
                    issues.append('Saved median disagrees: '+str(name)+'/'+arm)
            if not close(saved.get('median_reverse_order_pair_ratio'),summary['median_reverse_order_pair_ratio']):
                issues.append('Saved paired ratio disagrees: '+str(name))
            stored_pairs=saved.get('reverse_order_pair_geomeans',[])
            if len(stored_pairs)!=6 or not all(close(a,b) for a,b in zip(stored_pairs,summary['reverse_order_pair_geomeans'])):
                issues.append('Saved six reverse-order ratios disagree: '+str(name))
            calibration=timing.get('calibration_us',[])
            if len(calibration)!=6 or not all(finite_positive(v) for v in calibration):
                issues.append('Calibration incomplete/nonpositive: '+str(name))
            else:
                offset=statistics.median(calibration)
                if not close(timing.get('empty_graph_median_us'),offset):
                    issues.append('Calibration median differs: '+str(name))
                row['separate_empty_graph_median_us']=offset
            row['recomputed_summary']=summary
        timing_audits.append(row)
    timing_gate=([x['case'] for x in timing_audits]==expected_timing_names and
                 all(x['complete_and_valid'] for x in timing_audits))
    admitted=(not issues and oracle_gate and timing_gate and data.get('status')=='complete' and
              data.get('passed') is True and data.get('correctness_complete') is True and data.get('timing_complete') is True)
    if not admitted:
        for row in timing_audits:
            row.pop('recomputed_summary',None)
            row.pop('separate_empty_graph_median_us',None)
    if not failures and data.get('status')!='complete':
        last=cases[-1] if cases else None
        failures.append(dict(case=last.get('case',{}).get('name') if last else None,
            arm='unavailable: worker exception did not record current arm',
            completed_records_in_case=len(last.get('records',[])) if last else 0,
            error_type=data.get('error_type'),error=data.get('error'),
            qualification='Do not infer a failed arm from capture or missing records.'))
    profile=data.get('fa3_profiler') or {}
    scheduler_names=[]
    for name in profile.get('kernel_names',[]):
        matches=re.findall(r'flash::VarlenDynamicPersistentTileScheduler<[^>]+>',name)
        for match in matches:
            scheduler_names.append(dict(literal_scheduler_template=match,
                full_event_name_sha256=hashlib.sha256(name.encode()).hexdigest()))
    return dict(kind='Offline audit of bounded BF16 custom/unchanged FA3 representative run',
        analyzer_sha256=sha(Path(__file__)),result_sha256=sha(result_path),launch_receipt_sha256=sha(receipt_path),
        source_commit=receipt.get('source_commit'),app_id=receipt.get('app_id'),payload_provenance=provenance,
        recorded_status=data.get('status'),recorded_passed=data.get('passed'),
        official_oracle_calls=oracle_count,oracle_replays_passed=checked_count,
        exact_returning_a_checks=returns_count,all_18_oracles_admitted=oracle_gate,
        timing_complete_gate=timing_gate,performance_results_admitted=admitted,
        audit_issues=issues,failures=failures,cases=case_audits,timings=timing_audits,
        environment=data.get('environment'),phases=data.get('phases'),worker_wall_seconds=data.get('worker_wall_seconds'),
        unchanged_fa3_control=dict(exact_candidate_hash=FROZEN['candidate.py'],
            frozen_worker_function_binding='fa3 = candidate.kernel_fn; bf16_custom = separate helper',
            formal_dispatch_modified=False),
        fa3_profiler=data.get('fa3_profiler'),observed_named_schedulers=scheduler_names,
        limitations=['Representative cases only: not full official prerequisite suite, all shapes, concurrency, or independent-session confirmation.',
          'Raw complete-call graph timings primary; separate empty-graph calibration is not official judge calibration.',
          'No official score, confidence interval, selected-candidate promotion, or scheduler novelty claim.'])


def markdown(audit):
    lines=['# BF16 full-verify follow-through: '+('complete representative run' if audit['performance_results_admitted'] else 'failed or incomplete; no performance claim'),'',
        f"Frozen execution commit `{audit['source_commit']}`. Result SHA256 `{audit['result_sha256']}`.",
        f"Published launch record SHA256 `{audit['launch_receipt_sha256']}`; default analysis reads `launch-receipt.json` and requires no ignored local record.",
        f"Recorded status `{audit['recorded_status']}`; official-oracle calls {audit['official_oracle_calls']}, passed replays {audit['oracle_replays_passed']}/18, exact returning-A checks {audit['exact_returning_a_checks']}/6.",'',
        'The FA3 arm is unchanged exact A2-13 `candidate.kernel_fn`; the experimental arm uses a separate BF16-QK helper. Formal routing and selected A2-11 remain unchanged.']
    if not audit['performance_results_admitted']:
        lines+=['','**No timing result is admitted.** Failure details:','']
        for failure in audit['failures']:
            lines.append('- '+json.dumps(failure,ensure_ascii=False))
        for issue in audit['audit_issues']:
            lines.append('- Audit issue: '+issue)
        lines+=['','Partial observations and error records remain in `analysis.json` and `result.json`; a compile pass does not establish valid runtime behavior. Stop this variant until its exact failure is understood.']
    else:
        lines+=['','Both families completed all planned samples after all three cases passed the unchanged official oracle. All measured and warmup replays checked exact same-arm output/new-KV bytes, stable addresses, immutable inputs and untouched cache rows.','',
            '| Family | FA3 raw median us | BF16 custom raw median us | Median FA3/custom pair ratio | Custom latency change |',
            '|---|---:|---:|---:|---:|']
        for timing in audit['timings']:
            s=timing['recomputed_summary']
            lines.append(f"| {timing['case']} | {s['raw_median_us']['fa3']:.6f} | {s['raw_median_us']['bf16_custom']:.6f} | {s['median_reverse_order_pair_ratio']:.9f} | {s['paired_candidate_latency_change_fraction']:+.3%} |")
        lines+=['','Positive latency change means slower than FA3. Each ratio summarizes six adjacent reverse-order pairs; these are within-session observations from two input versions, without confidence intervals or an official score.','']
        for timing in audit['timings']:
            s=timing['recomputed_summary']
            lines.append(timing['case']+' reverse-order pair ratios: '+', '.join(f'{x:.9f}' for x in s['reverse_order_pair_geomeans'])+'.')
        if all(t['recomputed_summary']['median_reverse_order_pair_ratio']<1 for t in audit['timings']):
            lines+=['','**Decision: reject this complete custom BF16 route as a performance candidate.** Both families are slower than unchanged FA3; an official check is not justified by these results. WGMMA feasibility passed compilation, but that is insufficient to overcome the complete-path deficit. This run did not time the old custom FP8 route, so it cannot isolate BF16 widening\'s own runtime effect.']
    lines+=['','The optional profiler is outside paired timing.']
    if audit['observed_named_schedulers']:
        lines+=['','The actual B32/L8/D64/page1 FA3 event name contains the complete scheduler type:',
            '',*['`'+s['literal_scheduler_template']+'`' for s in audit['observed_named_schedulers']],
            '','Thus the existing B32 verify specialization already names a dynamic persistent scheduler. Exact build/source identity, boolean meanings, B16 scheduling and native instructions remain unverified. See `FA3_PROFILE.md`; no scheduler novelty is claimed.']
    else:
        lines+=['No precise scheduler template was recovered. Full kernel names require source/template evidence before identifying the installed scheduler. No novelty is inferred from generic FlashAttention names.']
    lines+=['',
        'Scope: page64/scale1/64 boundary lengths plus full B16/B32 L8 80K page1 coherent families; same-address A→B→A replay. This is not all official shapes, the full prerequisite suite, concurrent-call testing or independent-session confirmation. No official check or selected-candidate promotion follows from this report.','',
        'Exact payload hashes, all recomputed values and failures are preserved in `analysis.json`; deliverable hashes are in `analysis-manifest.json`.']
    return '\n'.join(lines)+'\n'


def self_test(data,receipt,result_path,receipt_path):
    audit=analyze(data,receipt,result_path,receipt_path)
    assert audit['performance_results_admitted'] and audit['official_oracle_calls']==18
    bad=copy.deepcopy(data)
    bad['cases'][1]['records'][1]['passed']=False
    audit=analyze(bad,receipt,result_path,receipt_path)
    assert not audit['performance_results_admitted'] and all('recomputed_summary' not in t for t in audit['timings'])
    bad=copy.deepcopy(data)
    bad['timings'][0]['samples'].pop()
    audit=analyze(bad,receipt,result_path,receipt_path)
    assert not audit['performance_results_admitted'] and all('recomputed_summary' not in t for t in audit['timings'])
    bad=copy.deepcopy(data)
    bad['timings'][0]['summary']['median_reverse_order_pair_ratio']=1.5
    audit=analyze(bad,receipt,result_path,receipt_path)
    assert not audit['performance_results_admitted'] and audit['audit_issues']
    verification=dict(positive_complete_gate=True,negative_failed_oracle_suppresses_timings=True,
        negative_missing_sample_suppresses_timings=True,negative_mismatched_summary_suppresses_timings=True,
        result_sha256=sha(result_path),launch_receipt_sha256=sha(receipt_path),
        analyzer_sha256=sha(Path(__file__)),gpu_network_browser_calls=0)
    (HERE/'analysis-local-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('4 admission/rejection guards passed with published launch receipt; immutable result unchanged')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,default=HERE/'result.json')
    parser.add_argument('--receipt',type=Path,default=HERE/'launch-receipt.json',
        help='Published launch record; alternate records require an explicit path. No ignored local-file fallback.')
    parser.add_argument('--self-test',action='store_true',help='Run four admission/rejection guards against the immutable recorded run.')
    args=parser.parse_args()
    data=json.loads(args.result.read_text())
    receipt=json.loads(args.receipt.read_text())
    if args.self_test:
        self_test(data,receipt,args.result,args.receipt)
        return 0
    audit=analyze(data,receipt,args.result,args.receipt)
    (HERE/'analysis.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'ANALYSIS.md').write_text(markdown(audit))
    names=['analyze_result.py','analysis.json','ANALYSIS.md','result.json','launch-receipt.json']
    names += [n for n in ('FA3_PROFILE.md','analysis-local-verification.json') if (HERE/n).is_file()]
    (HERE/'analysis-manifest.json').write_text(json.dumps({n:sha(HERE/n) for n in names},indent=2)+'\n')
    print(json.dumps(dict(performance_results_admitted=audit['performance_results_admitted'],
        oracle_calls=audit['official_oracle_calls'],failures=audit['failures'],audit_issues=audit['audit_issues'],
        analysis_sha256=sha(HERE/'analysis.json')),indent=2))
    return 0 if not audit['audit_issues'] else 2


if __name__=='__main__':
    raise SystemExit(main())
