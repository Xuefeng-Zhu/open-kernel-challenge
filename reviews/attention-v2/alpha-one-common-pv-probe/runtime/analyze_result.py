"""Offline evidence reconstruction; valid negative outcomes are not launch errors."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
PROBE, REPO = HERE.parent, HERE.parents[3]
FREEZE = '032c2ffaf494c4f847e0b9f4bc97b7dd23e627fc'
APP = 'ap-wE6ZSXzC1dnP253vX0kM5l'
WORKER = 'c3e9193d8c7dbb8cd68c06ea363b2f77cc8b4d64657be6a20ad832ac3e1739bb'
CONTROLLER = 'ab4b88f430ab2934f4cd0e75ce5bfa1b16fd1f6c1ca3801631a3f0b77ef7f0e3'
DIAGNOSTIC = 'ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60'
SOURCES = {'baseline.py':'3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'candidate.py':'14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5'}
SUPPORT = {'modal_worker.py':'9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py':'386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py':'9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055'}
TARGET = dict(name='full_B32_L1_scored',batch=32,q_len=1,ctx=80000,heads='tp8',layer='full',page=1,kv_scale=1.,spread='agentic',dist='coherent',boundary=False,force=None)
FIRST = dict(name='full_B32_L1_first_max',batch=32,q_len=1,ctx=8193,heads='tp8',layer='full',page=64,kv_scale=1/64,spread='fixed',dist='soft',boundary=True,force='first')
CASES = [TARGET,FIRST,dict(FIRST,name='full_B32_L1_middle_max',force='middle'),dict(FIRST,name='full_B32_L1_last_max',force='last'),dict(FIRST,name='full_B32_L1_DP_boundary',heads='dp',force=None),dict(TARGET,name='full_B16_L8_fusion_unchanged',batch=16,q_len=8)]
BASE_COUNTS = dict(original_oracle_calls=39,exact_return_A_checks=12,cross_arm_cache_byte_comparisons=36,
    concurrent_original_oracle_calls=2,independent_setup_oracle_calls=1,post_concurrent_isolation_checks=4,
    extra_candidate_replays=2,diagnostic_calls=10,diagnostic_reference_replays=10,cross_arm_output_newkv_ABA_comparisons=18)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def checked(row,oracle,exact=None):
    assert all(row[k] is True for k in ['passed','stable_addresses','immutable_inputs_and_untouched_rows','finite_output'])
    assert row['official_oracle_called'] is oracle and row['exact_output_newkv_return'] is exact


def byte_check(row,name=None,size=None):
    assert row['exact'] is True and row['unequal_byte_count']==0 and type(row['bytes']) is int and row['bytes']>0
    if name is not None: assert row['name']==name
    if size is not None: assert row['bytes']==size


def counter_model(lengths,hkv,splits,force):
    """Enumerate tile starts and record maxima independently of chunk ceil formulas."""
    expected=[]
    for seq in lengths:
        starts=list(range(0,seq,64))
        leader=None if not starts or force is None else starts[{'first':0,'middle':(len(starts)-1)//2,'last':len(starts)-1}[force]]
        for _ in range(hkv):
            for split in range(splits):
                tiles=[at for at in starts if at//8192==split]
                slow=None
                if force:
                    old=-math.inf;slow=0
                    for at in tiles:
                        new=max(old,1 if at==leader else 0)
                        slow+=new!=old;old=new
                expected.append((len(tiles),slow))
    return expected


def timing_summary(samples):
    assert len(samples)==12
    for index,row in enumerate(samples,4):
        assert row['index']==index and row['input_version']==(index//2)%2
        assert row['order']==(['baseline','candidate'] if index%2==0 else ['candidate','baseline'])
        assert all(math.isfinite(row['raw_us'][key]) and row['raw_us'][key]>0 for key in ['baseline','candidate'])
    ratios=[row['raw_us']['baseline']/row['raw_us']['candidate'] for row in samples]
    pairs=[math.sqrt(a*b) for a,b in zip(ratios[::2],ratios[1::2])]
    versions=[row['input_version'] for row in samples[::2]]
    per={}
    for version in [0,1]:
        values=[ratio for ratio,v in zip(pairs,versions) if v==version];assert len(values)==3
        median=statistics.median(values)
        per[str(version)]=dict(pair_ratios=values,median_ratio=median,latency_reduction_percent=100*(1-1/median),slowdown_percent=100*(1/median-1),positive_pairs=sum(x>1 for x in values))
    return dict(raw_median_us={key:statistics.median(r['raw_us'][key] for r in samples) for key in ['baseline','candidate']},reverse_order_pair_geomeans=pairs,pair_input_versions=versions,median_reverse_order_pair_ratio=statistics.median(pairs),positive_reverse_order_pairs=sum(x>1 for x in pairs),per_fixture=per)


def analyze():
    raw=(HERE/'result.json').read_bytes(); launch=(HERE/'launch-receipt.json').read_bytes(); close=(HERE/'reconciliation.json').read_bytes()
    report,receipt,closure=map(json.loads,[raw,launch,close])
    assert len(raw)<=500000 and receipt['report_sha256']==sha(raw)
    assert receipt['source_commit']==FREEZE and receipt['app_id']==APP
    assert receipt['worker_status']==report['status'] and receipt['worker_passed'] is report['passed']
    assert receipt['paid_limit_usd']=='0' and Decimal(receipt['cost_reserve_usd'])==Decimal('.58')
    assert receipt['resources']==dict(gpu='H200',cpu=[2,2],memory_mib=[8192,8192],max_containers=1,worker_deadline=160,child_timeout=170,function_timeout=185,startup_timeout=20,watchdog=210,retry_policy=0,explicit_invocations=1)
    assert receipt['performance_admission'] is False and receipt['official_check'] is False and receipt['formal_entry'] is False
    assert receipt['performance_measurement'] is bool(report['timings'])
    v2=HERE.parents[1]
    fixed={str((PROBE/name).relative_to(REPO)):value for name,value in SOURCES.items()}
    fixed.update({str((v2/name).relative_to(REPO)):value for name,value in SUPPORT.items()})
    fixed.update({str((HERE/'worker.py').relative_to(REPO)):WORKER,str((HERE/'controller.py').relative_to(REPO)):CONTROLLER,str((HERE/'diagnostic.py').relative_to(REPO)):DIAGNOSTIC})
    assert set(receipt['payload_sha256'])==set(fixed)|{'LICENSE','THIRD_PARTY_NOTICES.md'}
    for name,value in receipt['payload_sha256'].items():
        assert name not in fixed or fixed[name]==value
        assert sha((REPO/name).read_bytes())==value and sha(subprocess.check_output(['git','show',FREEZE+':'+name],cwd=REPO))==value,name
    assert report['source_sha256']==SOURCES and report['support_sha256']==SUPPORT and report['diagnostic_sha256']==DIAGNOSTIC and report['worker_sha256']==WORKER
    assert closure['app']['app_id']==APP and closure['app']['state']=='stopped' and int(closure['app']['tasks'])==0
    assert closure['stage_closed'] is True and closure['all_apps_stopped_zero_tasks'] is True
    assert all(x['state']=='stopped' and int(x['tasks'])==0 for x in closure['all_apps']) and closure['app'] in closure['all_apps']
    assert Decimal(closure['billing']['billed_cost'])==Decimal(closure['paid_limit_usd'])==0
    common=dict(source_freeze=FREEZE,app_id=APP,source_sha256=SOURCES,raw_result_sha256=sha(raw),launch_receipt_sha256=sha(launch),reconciliation_sha256=sha(close),stage_closed=True,app_stopped=True,app_tasks=0,closure_observed_utc=closure['observed_utc'],actual_paid_usd=str(Decimal(closure['billing']['billed_cost'])),automatic_performance_admission=False,official_check=False)
    if report['passed'] is not True:
        return dict(**common,native_run_complete=False,execution_failed=True,status=report['status'],error_type=report.get('error_type'),error=report.get('error'),reported_partial_counts=report.get('counts'),benefit_gate_passed=None,decision='Stop: incomplete execution/evidence, not a completed performance negative.')
    assert receipt['exit_code']==0 and report['status']=='complete' and report['correctness_complete'] is True and report['diagnostic_complete'] is True and report['official_check_admitted'] is False
    assert report['verified_hashes']=={**SOURCES,**SUPPORT,'diagnostic.py':DIAGNOSTIC}
    protocol=report['protocol'];assert protocol['seeds']==[135731,135732] and protocol['cases']==CASES
    assert protocol['original_oracle_and_tolerances'] is True and protocol['exact_cross_arm_output_newkv'] is True and protocol['profiler_used'] is False and protocol['instrumented_timings_reported'] is False
    assert protocol['warmup_samples']==4 and protocol['measured_samples']==12 and protocol['cold_l2_eviction_bytes']==256*1024*1024
    assert len(report['cases'])==6
    counts={key:0 for key in BASE_COUNTS};counts.update(checked_warmup_measured_replays=0,cross_arm_output_newkv_timing_comparisons=0)
    case_map={}
    for item,case in zip(report['cases'],CASES):
        assert item['case']==case and item['passed'] is True
        cfg=item['config'];b,l,hq,hkv=case['batch'],case['q_len'],128 if case['heads']=='dp' else 16,16 if case['heads']=='dp' else 2
        assert [cfg[k] for k in ['batch','q_len','num_q_heads','num_kv_heads','head_dim','page_size','window','rope_theta']]==[b,l,hq,hkv,64,case['page'],0,0.]
        assert cfg['kv_scale']==case['kv_scale']
        columns=((86000 if not case['boundary'] else 8193)+l+case['page']-1)//case['page'];assert item['page_table_shape']==[b,columns]
        lengths=item['input_lengths'];assert len(lengths)==2 and lengths[0]!=lengths[1]
        assert all(len(v)==b and all(type(x)is int and 0<=x<columns*case['page'] for x in v) for v in lengths)
        if case['boundary']: assert lengths==[[0,1,63,64,65,8191,8192,8193]*4,([0,1,63,64,65,8191,8192,8193]*4)[::-1]]
        cache_size=b*columns*case['page']*hkv*64; raw_size=b*l*(hq*64*2+2*hkv*64)
        assert len(item['records'])==6
        for row,(label,version,name) in zip(item['records'],[(label,v,name)for label,v in [('A1',0),('B',1),('A2',0)]for name in ['baseline','candidate']]):
            assert (row['label'],row['input_version'],row['runner'])==(label,version,name);checked(row,True,True if label=='A2' else None)
            counts['original_oracle_calls']+=1;counts['exact_return_A_checks']+=label=='A2'
        assert len(item['cache_cross_arm'])==len(item['output_newkv_cross_arm'])==3
        for cache,out,(label,version) in zip(item['cache_cross_arm'],item['output_newkv_cross_arm'],[('A1',0),('B',1),('A2',0)]):
            assert cache['passed'] is True and (cache['label'],cache['input_version'])==(label,version) and len(cache['comparisons'])==2
            for row,name in zip(cache['comparisons'],['k_cache','v_cache']):byte_check(row,name,cache_size);counts['cross_arm_cache_byte_comparisons']+=1
            assert (out['label'],out['input_version'])==(label,version);byte_check(out,'output_newkv',raw_size);counts['cross_arm_output_newkv_ABA_comparisons']+=1
        for version,leader in enumerate(item['leader_evidence']):
            if not case['force']:assert leader is None;continue
            positions=[None if x==0 else list(range(0,x,64))[{'first':0,'middle':((x+63)//64-1)//2,'last':(x+63)//64-1}[case['force']]] for x in lengths[version]]
            assert leader['historical_leader_positions']==positions and leader['placement']==case['force']
            assert all(leader[k] is True for k in ['encoded_query_dim0_positive','reference_query_dim0_positive','query_other_components_zero','new_keys_zero','nonleader_history_keys_zero'])
            assert leader['cache_nonzero_components']==leader['expected_nonzero_components']==sum(x is not None for x in positions)*hkv and leader['leader_raw_e4m3_value']==32.
        case_map[case['name']]=(item,case,hkv,cache_size,raw_size)
    assert len(report['independent_setup_checks'])==len(report['concurrent_submissions'])==1
    setup=report['independent_setup_checks'][0];assert setup['case']==CASES[2]['name'] and setup['input_version']==0;checked(setup,True)
    counts['independent_setup_oracle_calls']+=1;counts['original_oracle_calls']+=1
    concurrent=report['concurrent_submissions'][0];assert concurrent['case']==CASES[2]['name'] and concurrent['passed'] is True
    assert all(concurrent[k] is True for k in ['disjoint_writable_caches','disjoint_outputs','separate_graph_captures','separate_streams']) and concurrent['actual_hardware_overlap_measured'] is False and concurrent['direct_scratch_pointers_observed'] is False
    assert len(concurrent['checks'])==2;checked(concurrent['checks'][0],True,True);checked(concurrent['checks'][1],True)
    counts['original_oracle_calls']+=2;counts['concurrent_original_oracle_calls']+=2
    assert len(concurrent['isolation_checks'])==4
    for row,key in zip(concurrent['isolation_checks'],[('replay_active',1),('check_inactive_without_replay',1),('replay_active',0),('check_inactive_without_replay',1)]):
        assert (row['action'],row['input_version'])==key;checked(row,False,True);counts['post_concurrent_isolation_checks']+=1;counts['extra_candidate_replays']+=key[0]=='replay_active'
    assert len(report['diagnostics'])==10; frequencies=[]
    for row,(case,version) in zip(report['diagnostics'],[(case,v)for case in CASES[:5]for v in [0,1]]):
        assert row['case']==case['name'] and row['input_version']==version and row['passed'] is True and row['timed'] is False and row['original_oracle_called'] is False and row['designated_output']=='alpha_stats'
        item,_,hkv,cache_size,raw_size=case_map[case['name']];splits=(item['page_table_shape'][1]*case['page']+8191)//8192
        expected=counter_model(item['input_lengths'][version],hkv,splits,case['force']);actual=row['actual_stats']
        assert row['splits']==splits and row['stats_shape']==[len(expected),3] and len(actual)==len(expected)
        for (total,fast,slow),(visited,forced_slow) in zip(actual,expected):
            assert all(type(x)is int and x>=0 for x in [total,fast,slow]) and total==visited and fast+slow==total
            assert total==0 or slow>=1
            if forced_slow is not None:assert slow==forced_slow
        summary=dict(passed=True,total=sum(x[0]for x in actual),fast=sum(x[1]for x in actual),slow=sum(x[2]for x in actual),expected_total_by_cta=[x[0]for x in expected],expected_slow_by_cta=[x[1]for x in expected])
        assert row['stats_validation']==summary;checked(row['control_replay_check'],False,True)
        byte_check(row['output_newkv_comparison'],'diagnostic_output_newkv',raw_size)
        assert len(row['cache_comparisons'])==2
        for check,name in zip(row['cache_comparisons'],['k_cache','v_cache']):byte_check(check,name,cache_size)
        assert row['immutable_inputs_and_untouched_rows'] is True and row['stable_addresses'] is True and row['finite_output'] is True
        counts['diagnostic_calls']+=1;counts['diagnostic_reference_replays']+=1
        frequencies.append(dict(case=case['name'],input_version=version,total=summary['total'],fast=summary['fast'],slow=summary['slow'],fast_fraction=summary['fast']/summary['total'] if summary['total'] else None))
    fast=[next(x['fast']for x in frequencies if x['case']==TARGET['name'] and x['input_version']==v)for v in [0,1]];opportunity=all(x>0 for x in fast)
    assert report['frequency_gate']['scored_fast_counts']==fast and report['frequency_gate']['opportunity'] is opportunity
    summaries=[]
    if not opportunity:
        assert report['timing_complete'] is False and report['timings']==[] and report['benefit']['admitted'] is False and report['outcome']=='no_scored_fast_opportunity'
        admitted=False;decision='Reject: at least one scored fixture has zero fast opportunity; no timing was performed.'
    else:
        assert report['timing_complete'] is True and len(report['timings'])==2 and report['outcome']=='timed'
        for row,case in zip(report['timings'],[TARGET,CASES[5]]):
            assert row['case']==case and len(row['warmups'])==4 and len(row['samples'])==12
            raw_size=case_map[case['name']][4]
            for index,sample in enumerate(row['warmups']+row['samples']):
                assert sample['index']==index and sample['input_version']==(index//2)%2 and sample['order']==(['baseline','candidate'] if index%2==0 else ['candidate','baseline'])
                for name in ['baseline','candidate']:checked(sample['checks'][name],False,True);counts['checked_warmup_measured_replays']+=1
                byte_check(sample['cross_arm_output_newkv'],'output_newkv',raw_size);counts['cross_arm_output_newkv_timing_comparisons']+=1
            summary=timing_summary(row['samples'])
            assert all(row['summary'][key]==value for key,value in summary.items());summaries.append(dict(case=case['name'],**summary))
        target_ok=all(x['latency_reduction_percent']>=2 for x in summaries[0]['per_fixture'].values()) and summaries[0]['positive_reverse_order_pairs']==6
        control_ok=all(x['slowdown_percent']<=2 for x in summaries[1]['per_fixture'].values());admitted=target_ok and control_ok
        assert report['benefit']['admitted'] is admitted and report['benefit']['target_passed'] is target_ok and report['benefit']['unchanged_control_passed'] is control_ok
        decision='Supports separately reviewed fresh-input confirmation only.' if admitted else 'Reject performance: completed valid timing experiment missed a predeclared per-fixture gate.'
    assert counts==report['counts'] and all(type(v)is int and v>=0 for v in report['counts'].values())
    assert all(counts[key]==value for key,value in BASE_COUNTS.items())
    assert counts['checked_warmup_measured_replays']==(64 if opportunity else 0) and counts['cross_arm_output_newkv_timing_comparisons']==(32 if opportunity else 0)
    return dict(**common,native_run_complete=True,execution_failed=False,outcome=report['outcome'],counts=counts,diagnostic_frequencies=frequencies,scored_fast_counts=fast,scored_opportunity=opportunity,timings=summaries,benefit_gate_passed=admitted,decision=decision,actual_hardware_overlap_measured=False,direct_scratch_pointers_observed=False,worker_wall_seconds=report['worker_wall_seconds'],environment=report['environment'],qualifications=['One session and two fresh fixtures; no score, interval or general serving-safety claim.','Diagnostic equality is uint8-exact observed output/newKV plus entire cache comparison; instrumented calls are never timed.','Fast frequency is not a cycle-saving estimate; predicated instruction issue costs remain a timing question.','A valid negative benefit/frequency result is a completed experiment, not an execution error.','No automatic official check, selection promotion or further launch follows.'])


if __name__=='__main__':
    result=analyze();(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result.get(key)for key in ['native_run_complete','execution_failed','outcome','counts','scored_fast_counts','benefit_gate_passed','decision','stage_closed']},indent=2))
