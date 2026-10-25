"""Offline verification of exact-source DP multichunk correctness evidence."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FREEZE = '8d38550d061af14df3257afe0d438dbf0fc82f5f'
APP = 'ap-gXUKhAM11JBfvrA3h5rip9'
WORKER_SHA = '4507a19cdaea450415b6d7a40315e9e0021741919f5a33842fed290e0f2af59d'
CONTROLLER_SHA = '0299612f835e68b26ece31f5d5d031c959953988ce4d998d95941ddd9d778f17'
SOURCES = {
    'baseline.py': '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0',
    'candidate.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
}
SUPPORT = {
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
CASES = [dict(name='DP_page64_ctx8193_soft_scaled', batch=32, q_len=1, ctx=8193,
    heads='dp', layer='full', page=64, kv_scale=1/64, spread='fixed', dist='soft'),
    dict(name='DP_page1_ctx16385_coherent', batch=32, q_len=1, ctx=16385,
    heads='dp', layer='full', page=1, kv_scale=1., spread='fixed', dist='coherent')]
COUNTS = dict(original_oracle_calls=15, independent_setup_oracle_calls=1,
    exact_return_A_checks=4, cross_arm_cache_byte_comparisons=12,
    concurrent_original_oracle_calls=2, post_concurrent_isolation_checks=4,
    extra_candidate_replays=2)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(row, oracle, exact=None):
    for key in ['passed', 'stable_addresses', 'immutable_inputs_and_untouched_rows', 'finite_output']:
        assert row[key] is True, key
    assert row['official_oracle_called'] is oracle
    assert row['exact_output_newkv_return'] is exact


def evidence(row):
    assert row['output_shape'] == [32, 1, 8192]
    for key in ['output_address', 'k_cache_address', 'v_cache_address']:
        assert type(row[key]) is int and row[key] > 0
    assert row['direct_scratch_pointers_observed'] is False
    observed = row['graph_pool_handle_observed']
    assert type(observed) is bool
    pool = row['graph_private_pool']
    if observed:
        assert type(pool) is list and len(pool) == 2
        assert all(type(value) is int and value >= 0 for value in pool)
    else:
        assert pool is None
    return pool


def disjoint_ranges(rows, cache_bytes):
    ranges = []
    for index, row in enumerate(rows):
        for key, size in [('output_address', 32*8192*2),
                          ('k_cache_address', cache_bytes), ('v_cache_address', cache_bytes)]:
            ranges.append((index, key, row[key], row[key]+size))
    for index, left in enumerate(ranges):
        for right in ranges[index+1:]:
            assert max(left[2], right[2]) >= min(left[3], right[3]), (left[:2], right[:2])


def analyze():
    raw = (HERE/'result.json').read_bytes()
    launch_raw = (HERE/'launch-receipt.json').read_bytes()
    closure_raw = (HERE/'reconciliation.json').read_bytes()
    report, receipt, closure = map(json.loads, [raw, launch_raw, closure_raw])
    assert len(raw) <= 400000 and receipt['report_sha256'] == sha(raw)
    assert receipt['source_commit'] == FREEZE and receipt['app_id'] == APP
    assert receipt['exit_code'] == 0 and receipt['worker_passed'] is True and receipt['worker_status'] == 'passed'
    assert receipt['base_image'] == 'im-cPTbGpNiX3RarKResPgX7h'
    assert receipt['correctness_only'] is True and receipt['performance_measurement'] is False
    assert receipt['official_check'] is False and receipt['formal_entry'] is False
    assert Decimal(receipt['paid_limit_usd']) == 0 and Decimal(receipt['cost_reserve_usd']) == Decimal('.58')
    assert receipt['resources'] == dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192],
        max_containers=1, worker_deadline=160, child_timeout=170, function_timeout=185,
        startup_timeout=20, watchdog=210, retry_policy=0, explicit_invocations=1)
    v2 = HERE.parents[1]
    files = {str((v2/'fusion-lookahead-probe'/name).relative_to(REPO)): value for name, value in SOURCES.items()}
    files.update({str((v2/name).relative_to(REPO)): value for name, value in SUPPORT.items()})
    files.update({str((HERE/'worker.py').relative_to(REPO)): WORKER_SHA,
                  str((HERE/'controller.py').relative_to(REPO)): CONTROLLER_SHA})
    payload = receipt['payload_sha256']
    assert set(payload) == set(files) | {'LICENSE', 'THIRD_PARTY_NOTICES.md'}
    assert all(payload[name] == value for name, value in files.items())
    for name, value in payload.items():
        assert sha((REPO/name).read_bytes()) == value, name
        assert sha(subprocess.check_output(['git', 'show', FREEZE+':'+name], cwd=REPO)) == value, name
    assert report['worker_sha256'] == WORKER_SHA
    assert report['source_sha256'] == SOURCES and report['support_sha256'] == SUPPORT
    assert report['verified_hashes'] == {**SOURCES, **SUPPORT}
    assert report['status'] == 'passed' and report['passed'] is True and report['correctness_complete'] is True
    assert report['official_check_admitted'] is False and report['timings'] == []
    protocol = report['protocol']
    assert protocol['correctness_only'] is True and protocol['original_tolerances'] is True
    assert protocol['no_performance_claim'] is True and protocol['seeds'] == [135721, 135722]
    assert protocol['cases'] == CASES and protocol['replay_labels'] == ['A1', 'B', 'A2']
    assert len(report['cases']) == 2
    counts = dict(original_oracle_calls=0, independent_setup_oracle_calls=0,
        exact_return_A_checks=0, cross_arm_cache_byte_comparisons=0,
        concurrent_original_oracle_calls=0, post_concurrent_isolation_checks=0, extra_candidate_replays=0)
    captures, geometries = [], []
    for item, case in zip(report['cases'], CASES):
        assert item['case'] == case and item['passed'] is True
        lengths = ([0, 1, 63, 64, 65, 8191, 8192, 8193]*4 if case['ctx'] == 8193 else
            [0, 1, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193, 16320, 16321, 16383, 16384, 16385]*2)
        assert item['input_lengths'] == [lengths, lengths[::-1]]
        width = (case['ctx']+1+case['page']-1)//case['page']
        span, splits = width*case['page'], (width*case['page']+8191)//8192
        n = 32*16*splits*16
        geometry = item['geometry']
        expected = dict(page_table_shape=[32, width], shape_bound_tokens=span, chunk_size=8192,
            splits=splits, merge_bs=1 << (splits-1).bit_length(), producer_grid=[32, 16, splits],
            merge_grid=[32, 128], block_m=16, partial_shapes=dict(o=[n, 64], m=[n], l=[n]),
            source_derived_partial_bytes=n*66*4, cache_bytes_per_tensor=32*span*16*64)
        assert {key: geometry[key] for key in expected} == expected
        assert (splits, geometry['merge_bs']) == ((2, 2) if case['page'] == 64 else (3, 4))
        geometries.append(dict(case=case['name'], **expected))
        assert len(item['scheduler_metadata_shapes']) == 2
        assert item['scheduler_metadata_shapes'][0] == item['scheduler_metadata_shapes'][1]
        assert all(type(d) is int and d > 0 for d in item['scheduler_metadata_shapes'][0])
        cap = item['capture_evidence']
        assert set(cap) == {'baseline', 'candidate'}
        pools = [evidence(cap[name]) for name in ['baseline', 'candidate']]
        disjoint_ranges([cap[name] for name in ['baseline', 'candidate']], expected['cache_bytes_per_tensor'])
        assert cap['baseline']['output_address'] != cap['candidate']['output_address']
        if all(pool is not None for pool in pools):
            assert pools[0] != pools[1]
        captures.append(dict(case=case['name'], graph_pools=pools))
        expected_rows = [(name, label, version) for label, version in [('A1', 0), ('B', 1), ('A2', 0)]
                         for name in ['baseline', 'candidate']]
        assert len(item['records']) == 6
        for row, key in zip(item['records'], expected_rows):
            assert (row['runner'], row['label'], row['input_version']) == key
            checked(row, True, True if key[1] == 'A2' else None)
            counts['original_oracle_calls'] += 1
            counts['exact_return_A_checks'] += key[1] == 'A2'
        assert len(item['cache_cross_arm']) == 3
        for row, (label, version) in zip(item['cache_cross_arm'], [('A1', 0), ('B', 1), ('A2', 0)]):
            assert row['passed'] is True and (row['label'], row['input_version']) == (label, version)
            assert len(row['comparisons']) == 2
            for byte_row, name in zip(row['comparisons'], ['k_cache', 'v_cache']):
                assert byte_row['name'] == name and byte_row['exact'] is True
                assert byte_row['unequal_byte_count'] == 0 and byte_row['bytes'] == expected['cache_bytes_per_tensor']
                counts['cross_arm_cache_byte_comparisons'] += 1
    assert len(report['independent_setup_checks']) == 1
    setup = report['independent_setup_checks'][0]
    assert (setup['case'], setup['runner'], setup['input_version']) == (CASES[0]['name'], 'candidate_independent', 0)
    checked(setup, True)
    counts['original_oracle_calls'] += 1
    counts['independent_setup_oracle_calls'] += 1
    assert len(report['concurrent_submissions']) == 1
    group = report['concurrent_submissions'][0]
    assert group['case'] == CASES[0]['name']
    for key in ['passed', 'disjoint_writable_caches', 'disjoint_outputs', 'separate_graph_captures', 'separate_streams']:
        assert group[key] is True, key
    assert group['actual_hardware_overlap_measured'] is False and group['direct_scratch_pointers_observed'] is False
    cap = group['capture_evidence']
    assert len(cap) == 2 and cap[0] == report['cases'][0]['capture_evidence']['candidate']
    independent_pools = [evidence(row) for row in cap]
    disjoint_ranges(cap, geometries[0]['cache_bytes_per_tensor'])
    for key in ['output_address', 'k_cache_address', 'v_cache_address']:
        assert cap[0][key] != cap[1][key]
    observed = all(pool is not None for pool in independent_pools)
    assert group['distinct_default_graph_pools'] is (True if observed else None)
    if observed:
        assert independent_pools[0] != independent_pools[1]
    assert len(group['checks']) == 2
    checked(group['checks'][0], True, True)
    checked(group['checks'][1], True)
    counts['original_oracle_calls'] += 2
    counts['concurrent_original_oracle_calls'] += 2
    schedule = [('replay_active', 1), ('check_inactive_without_replay', 1),
                ('replay_active', 0), ('check_inactive_without_replay', 1)]
    assert len(group['isolation_checks']) == 4
    for row, key in zip(group['isolation_checks'], schedule):
        assert (row['action'], row['input_version']) == key
        checked(row, False, True)
        counts['post_concurrent_isolation_checks'] += 1
        counts['extra_candidate_replays'] += key[0] == 'replay_active'
    assert counts == report['counts'] == COUNTS
    assert all(type(value) is int and value >= 0 for value in report['counts'].values())
    assert report['environment']['gpu'] == 'NVIDIA H200' and report['environment']['capability'] == [9, 0]
    assert 0 < report['worker_wall_seconds'] < 160
    assert closure['app']['app_id'] == APP and closure['app']['state'] == 'stopped' and int(closure['app']['tasks']) == 0
    assert closure['stage_closed'] is True and closure['all_apps_stopped_zero_tasks'] is True
    assert all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in closure['all_apps'])
    assert any(row == closure['app'] for row in closure['all_apps'])
    assert Decimal(closure['billing']['billed_cost']) == Decimal(closure['paid_limit_usd']) == 0
    return dict(source_freeze=FREEZE, app_id=APP, source_sha256=SOURCES,
        raw_result_sha256=sha(raw), launch_receipt_sha256=sha(launch_raw), reconciliation_sha256=sha(closure_raw),
        validation_passed=True, counts=counts, fresh_seeds=[135721, 135722],
        source_derived_geometry=geometries, graph_pool_observations=captures,
        independent_candidate_graph_pools=independent_pools, independent_graph_checks_passed=True,
        observable_output_cache_address_ranges_disjoint=True,
        actual_hardware_overlap_measured=False, direct_scratch_pointers_observed=False,
        environment=report['environment'], worker_wall_seconds=report['worker_wall_seconds'],
        stage_closed=True, app_stopped=True, app_tasks=0, closure_observed_utc=closure['observed_utc'],
        actual_paid_usd=str(Decimal(closure['billing']['billed_cost'])), official_check=False, performance_measurement=False,
        qualifications=['Two targeted nearby DP cases, not universal serving/concurrency certification.',
            'Shape/partial/grid descriptions are derived from exact executed source; no achieved CTA-residency trace.',
            'Distinct graph pool handles and repeated output/cache isolation support these cases; internal scratch pointers and hardware overlap unmeasured.',
            'No latency measurement, new official score or extra official check follows from this validation.',
            'Shutdown and billed0 refer to the retained independent cutoff, not a fresh live workspace query.'])


if __name__ == '__main__':
    result = analyze()
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
