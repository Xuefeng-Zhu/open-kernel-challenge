"""Offline reconstruction of the exact V-stage runtime evidence. No services/GPU."""
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
PROBE, REPO = HERE.parent, HERE.parents[3]
WORKER = 'bc93f136f4943e28ce3fd6be91cf67639ee4172ea9195b715bc762ac80a13d83'
CONTROLLER = '1c9e664635fc31637e5f5629fc6bf37a03f6121eef69952ef4a8ed6cf401d062'
FREEZE = 'c55d14c507b9d5cc2254441531d4f0e1a18714d5'
APP = 'ap-Hy5O3TPhVTVKu0XyfsgJWT'
EXPECTED_COUNTS = dict(original_oracle_calls=30, exact_return_A_checks=9,
    cross_arm_cache_byte_comparisons=36, concurrent_original_oracle_calls=2,
    independent_setup_oracle_calls=1, post_concurrent_isolation_checks=4,
    extra_candidate_replays=2, prototype_output_newkv_ABA_comparisons=9,
    cross_arm_newkv_ABA_comparisons=18, checked_warmup_measured_replays=48,
    prototype_output_newkv_timing_comparisons=16,
    cross_arm_newkv_timing_comparisons=32)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def worker_module():
    path = HERE/'worker.py'
    assert sha(path.read_bytes()) == WORKER
    spec = importlib.util.spec_from_file_location('frozen_vstage_cpu_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Only the hash-bound stdlib top level.
    return module


def checked(row, oracle, exact):
    assert all(row[key] is True for key in
        ('passed', 'stable_addresses', 'immutable_inputs_and_untouched_rows', 'finite_output'))
    assert row['official_oracle_called'] is oracle
    assert row['exact_output_newkv_return'] is exact


def byte_check(row, name, size):
    assert row['name'] == name and row['exact'] is True
    assert type(row['bytes']) is int and row['bytes'] == size
    assert type(row['unequal_byte_count']) is int and row['unequal_byte_count'] == 0


def evidence_disjoint(rows):
    for key in ('output_address', 'k_cache_address', 'v_cache_address'):
        addresses = [row[key] for row in rows]
        assert all(type(value) is int and value > 0 for value in addresses)
        assert len(set(addresses)) == len(addresses)
    pools = [tuple(row['graph_private_pool']) for row in rows if row['graph_private_pool'] is not None]
    assert len(set(pools)) == len(pools)
    assert all(row['internal_scratch_pointers_observed'] is False for row in rows)


def independent_timing(samples, worker):
    """Independent arithmetic cross-check of the frozen worker's CPU helper."""
    assert len(samples) == 12
    for index, row in enumerate(samples, 4):
        order, version = worker.order_and_version(index)
        assert (row['index'], row['order'], row['input_version']) == (index, order, version)
        assert set(row['raw_us']) == set(worker.NAMES)
        assert all(type(value) in (int, float) and math.isfinite(value) and value > 0
                   for value in row['raw_us'].values())
    comparisons = {}
    for control in ('v_after_wait', 'fa3'):
        ratios = [row['raw_us'][control]/row['raw_us']['v_before_wait'] for row in samples]
        pairs = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, 12, 2)]
        versions = [row['input_version'] for row in samples[::2]]
        per_fixture = {}
        for version in (0, 1):
            values = [ratio for ratio, fixture in zip(pairs, versions) if fixture == version]
            assert len(values) == 3
            median = statistics.median(values)
            per_fixture[str(version)] = dict(pair_ratios=values, median_ratio=median,
                latency_reduction_percent=100*(1-1/median), positive_pairs=sum(value > 1 for value in values))
        comparisons['v_before_wait_vs_'+control] = dict(reverse_order_pair_geomeans=pairs,
            pair_input_versions=versions, positive_reverse_order_pairs=sum(value > 1 for value in pairs),
            per_fixture=per_fixture)
    summary = worker.timing_summary(samples)
    assert comparisons == summary['comparisons']
    assert summary['raw_median_us'] == {name: statistics.median(row['raw_us'][name] for row in samples)
                                      for name in worker.NAMES}
    return summary


def failure_rows(value, path='report'):
    found = []
    if isinstance(value, dict):
        if path != 'report' and value.get('passed') is False:
            found.append(dict(path=path, runner=value.get('runner'), label=value.get('label'),
                input_version=value.get('input_version'), oracle_message=value.get('oracle_message')))
        for key, item in value.items():
            found.extend(failure_rows(item, path+'.'+key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(failure_rows(item, path+'['+str(index)+']'))
    return found


def analyze():
    worker = worker_module()
    raw, launch, close = [(HERE/name).read_bytes() for name in
                          ('result.json', 'launch-receipt.json', 'reconciliation.json')]
    report, receipt, closure = map(json.loads, (raw, launch, close))
    assert len(raw) <= 500000 and sha(raw) == receipt['report_sha256']
    assert re.fullmatch(r'[0-9a-f]{40}', receipt['source_commit'])
    assert receipt['source_commit'] == FREEZE and receipt['app_id'] == APP
    assert receipt['worker_status'] == report['status'] and receipt['worker_passed'] is report['passed']
    assert report['source_sha256'] == worker.SOURCE_HASHES and report['support_sha256'] == worker.SUPPORT_HASHES
    assert report['worker_sha256'] == WORKER
    expected_paths = {
        str((HERE/'worker.py').relative_to(REPO)): WORKER,
        str((HERE/'controller.py').relative_to(REPO)): CONTROLLER,
        str((PROBE/'kernel.py').relative_to(REPO)): worker.SOURCE_HASHES['kernel.py'],
        str((PROBE/'config.py').relative_to(REPO)): worker.SOURCE_HASHES['config.py'],
        str((HERE.parents[1]/'fusion-lookahead-probe/candidate.py').relative_to(REPO)):
            worker.SOURCE_HASHES['fusion-lookahead-probe/candidate.py'],
        **{str((HERE.parents[1]/name).relative_to(REPO)): value for name, value in worker.SUPPORT_HASHES.items()},
        **worker.LICENSE_HASHES,
    }
    assert receipt['payload_sha256'] == expected_paths
    for name, expected in expected_paths.items():
        assert sha((REPO/name).read_bytes()) == expected, name
    assert worker.source_gate() == {**worker.SOURCE_HASHES, **worker.SUPPORT_HASHES, **worker.LICENSE_HASHES}
    assert receipt['resources'] == dict(gpu='H200', cpu=[2,2], memory_mib=[8192,8192],
        max_containers=1, worker_deadline=160, child_timeout=170, function_timeout=185,
        startup_timeout=20, watchdog=210, retry_policy=0, explicit_invocations=1)
    assert Decimal(receipt['paid_limit_usd']) == 0 and Decimal(receipt['cost_reserve_usd']) == Decimal('.58')
    assert receipt['performance_admission'] is False and receipt['official_check'] is False
    assert receipt['formal_entry'] is False
    assert report['official_check_admitted'] is False and report['production_replacement_admitted'] is False
    app = closure['app']
    assert app['app_id'] == receipt['app_id'] and app['state'] == 'stopped' and int(app['tasks']) == 0
    apps = closure.get('apps', closure.get('all_apps'))
    assert isinstance(apps, list) and app in apps
    assert all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in apps)
    assert Decimal(closure['billing']['billed_cost']) == 0
    result = dict(source_freeze=receipt['source_commit'], app_id=receipt['app_id'],
        source_sha256=worker.SOURCE_HASHES, frozen_payload_sha256=expected_paths,
        raw_result_sha256=sha(raw),
        launch_receipt_sha256=sha(launch), reconciliation_sha256=sha(close),
        stage_closed=True, app_stopped=True, tasks=0, closure_observed_utc=closure['observed_utc'],
        actual_paid_usd=str(Decimal(closure['billing']['billed_cost'])),
        automatic_performance_admission=False, official_check=False,
        production_replacement_admitted=False)
    if report['passed'] is not True:
        return dict(**result, native_run_complete=False, execution_failed=True,
            status=report['status'], correctness_complete=report['correctness_complete'],
            timing_complete=report['timing_complete'], error_type=report.get('error_type'),
            error=report.get('error'), last_completed_phase=(report.get('phases') or [None])[-1],
            last_recorded_case=(report.get('cases') or [{}])[-1].get('case'),
            failure_observations=failure_rows(report), reported_partial_counts=report['counts'],
            partial_timing_rows=len(report['timings']), benefit_gate_passed=None,
            decision='Stop: incomplete execution/evidence, not a completed performance result.')
    assert receipt['exit_code'] == 0 and report['status'] == 'complete'
    assert report['correctness_complete'] is True and report['timing_complete'] is True
    assert report['verified_hashes'] == {**worker.SOURCE_HASHES, **worker.SUPPORT_HASHES, **worker.LICENSE_HASHES}
    protocol = report['protocol']
    assert protocol['seeds'] == worker.SEEDS and protocol['cases'] == worker.CASES and protocol['arms'] == worker.NAMES
    assert protocol['original_oracle_and_tolerances'] is True and protocol['no_production_dispatch_change'] is True
    assert protocol['profiler_used'] is False and protocol['official_score_predicted'] is False
    assert protocol['page_table_stride'] == 86008 and protocol['max_nominal_context'] == 4096
    counts = {key:0 for key in EXPECTED_COUNTS}
    assert len(report['cases']) == 3
    raw_sizes = {}
    for item, case in zip(report['cases'], worker.CASES):
        assert item['case'] == case and item['passed'] is True
        cfg = item['config']
        assert [cfg[key] for key in ('batch','q_len','num_q_heads','num_kv_heads','head_dim','page_size','window','rope_theta')] == [32,8,16,2,64,1,0,0.]
        assert cfg['kv_scale'] == case['kv_scale'] and cfg['sm_scale'] == .125 and cfg['qk_scale_factor'] == 5.4730064863875
        lengths = item['input_lengths']
        assert len(lengths) == 2 and lengths[0] != lengths[1]
        assert all(len(vector) == 32 and all(type(value) is int and value >= 0 for value in vector) for vector in lengths)
        if case['boundary']:
            assert lengths == [worker.boundary_lengths(0), worker.boundary_lengths(1)]
        width = 137 if case['boundary'] else math.ceil(4096*1.075)+8
        assert len(item['padding_evidence']) == 2
        for evidence, vector in zip(item['padding_evidence'], lengths):
            assert evidence['original_page_table_width'] == width
            assert evidence['padded_page_table_shape'] == [32,86008] and evidence['padding_value'] == -1
            assert evidence['physical_cache_rows'] == 32*width
            assert evidence['seq_lens_dtype'] == evidence['page_table_dtype'] == 'torch.int32'
            assert all(evidence[key] is True for key in ('valid_prefix_preserved','cache_loc_matches_live_mapping',
                'all_live_page_entries_in_physical_range','all_unused_suffix_entries_negative_one'))
            assert all(value+8 <= width for value in vector)
        assert set(item['capture_evidence']) == set(worker.NAMES)
        evidence_disjoint(list(item['capture_evidence'].values()))
        raw_size, new_size, cache_size = 32*8*(16*64*2+2*2*64), 32*8*2*2*64, 32*width*2*64
        raw_sizes[case['name']] = (raw_size,new_size)
        plan = [(label,version,name) for label,version in (('A1',0),('B',1),('A2',0)) for name in worker.NAMES]
        assert len(item['records']) == len(plan)
        for row, (label,version,name) in zip(item['records'], plan):
            assert (row['label'],row['input_version'],row['runner']) == (label,version,name)
            checked(row, True, True if label == 'A2' else None)
            counts['original_oracle_calls'] += 1
            counts['exact_return_A_checks'] += label == 'A2'
        cross_plan = [(label,version,name) for label,version in (('A1',0),('B',1),('A2',0)) for name in worker.NAMES[1:]]
        assert len(item['cache_cross_arm']) == len(item['newkv_cross_arm']) == 6
        for row, newkv, (label,version,name) in zip(item['cache_cross_arm'], item['newkv_cross_arm'], cross_plan):
            assert (row['label'],row['input_version'],row['other']) == (label,version,name)
            assert len(row['comparisons']) == 2
            for byte, key in zip(row['comparisons'], ('k_cache','v_cache')):
                byte_check(byte, key, cache_size); counts['cross_arm_cache_byte_comparisons'] += 1
            assert (newkv['label'],newkv['other']) == (label,name)
            byte_check(newkv, 'newKV', new_size); counts['cross_arm_newkv_ABA_comparisons'] += 1
        assert len(item['prototype_output_cross_arm']) == 3
        for row, (label,version) in zip(item['prototype_output_cross_arm'], (('A1',0),('B',1),('A2',0))):
            assert (row['label'],row['input_version']) == (label,version)
            byte_check(row, 'prototype_output_newkv', raw_size); counts['prototype_output_newkv_ABA_comparisons'] += 1
    assert len(report['independent_setup_checks']) == len(report['concurrent_submissions']) == 1
    setup = report['independent_setup_checks'][0]
    assert setup['input_version'] == 0; checked(setup, True, None)
    counts['independent_setup_oracle_calls'] += 1; counts['original_oracle_calls'] += 1
    concurrent = report['concurrent_submissions'][0]
    assert concurrent['passed'] is True and concurrent['disjoint_outputs_and_writable_caches'] is True
    assert concurrent['actual_hardware_overlap_measured'] is False and concurrent['internal_scratch_pointers_observed'] is False
    assert len(concurrent['capture_evidence']) == 2; evidence_disjoint(concurrent['capture_evidence'])
    assert len(concurrent['checks']) == 2
    checked(concurrent['checks'][0], True, True); checked(concurrent['checks'][1], True, None)
    counts['original_oracle_calls'] += 2; counts['concurrent_original_oracle_calls'] += 2
    assert len(concurrent['isolation_checks']) == 4
    for row, action in zip(concurrent['isolation_checks'], ('replay_active','check_inactive_without_replay')*2):
        assert row['action'] == action; checked(row, False, True)
        counts['post_concurrent_isolation_checks'] += 1; counts['extra_candidate_replays'] += action == 'replay_active'
    assert len(report['timings']) == 1
    timing = report['timings'][0]
    assert timing['case'] == worker.TARGET and len(timing['warmups']) == 4 and len(timing['samples']) == 12
    raw_size, new_size = raw_sizes[worker.TARGET['name']]
    for index, sample in enumerate(timing['warmups']+timing['samples']):
        order, version = worker.order_and_version(index)
        assert (sample['index'],sample['order'],sample['input_version']) == (index,order,version)
        assert set(sample['checks']) == set(sample['raw_us']) == set(worker.NAMES)
        for name in worker.NAMES:
            assert math.isfinite(sample['raw_us'][name]) and sample['raw_us'][name] > 0
            checked(sample['checks'][name], False, True); counts['checked_warmup_measured_replays'] += 1
        byte_check(sample['prototype_pair'], 'prototype_output_newkv', raw_size)
        counts['prototype_output_newkv_timing_comparisons'] += 1
        assert len(sample['newkv_cross_arm']) == 2
        for row,name in zip(sample['newkv_cross_arm'], worker.NAMES[1:]):
            assert row['other'] == name; byte_check(row, 'newKV', new_size)
            counts['cross_arm_newkv_timing_comparisons'] += 1
    assert counts == report['counts'] == EXPECTED_COUNTS
    summary = independent_timing(timing['samples'], worker)
    benefit = worker.benefit_gate(timing['samples'])
    assert report['benefit'] == benefit and benefit['summary'] == summary
    for field in ('mechanism_passed','actual_fa3_passed','prospective_followup_supported'):
        assert receipt['worker_'+field] is benefit[field]
    assert receipt['performance_measurement'] is True
    return dict(**result, native_run_complete=True, execution_failed=False, outcome='timed', counts=counts,
        timing_summary=summary, mechanism_gate_passed=benefit['mechanism_passed'],
        actual_fa3_gate_passed=benefit['actual_fa3_passed'],
        prospective_followup_supported=benefit['prospective_followup_supported'],
        decision='Separate mechanism and actual-FA3 point-estimate gates; no automatic production/official admission.',
        worker_wall_seconds=report['worker_wall_seconds'], environment=report['environment'],
        actual_hardware_overlap_measured=False, internal_scratch_pointers_observed=False,
        qualifications=['Padded nearby ctx4096 geometry only; not an unpadded/80K or official score result.',
            'Two fresh fixtures and one native session; no population confidence interval or serving-safety claim.',
            'Adapters include preparation/new-KV/cast/core/output gate; actual FA3 entrypoint remains unchanged.'])


if __name__ == '__main__':
    result = analyze()
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result.get(key) for key in ('source_freeze','app_id','native_run_complete',
        'execution_failed','mechanism_gate_passed','actual_fa3_gate_passed','stage_closed')}, indent=2))
