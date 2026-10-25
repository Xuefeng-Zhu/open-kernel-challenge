"""Offline verification of a compile-gated common-PV artifact probe; no execution admission."""
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent
REPO = HERE.parents[3]
FREEZE = '6df535d096aec0e76379b6795bb97b5956d0474b'
APP = 'ap-dUTLyzmcwfCtweeterml8r'
WORKER_SHA = 'bef01dd950ce3955fb45316c5f5ee8937ea85a86c471b30c8639aeb0647e02e2'
CONTROLLER_SHA = '1c9256c05ba907e0f5044599a584304eb1cd7c32fe67fc9c6867481e8c75caaa'
SOURCES = {
    'baseline.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'candidate.py': '14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5',
}
LICENSES = {
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
FORMATS = ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass']
ARMS = ['baseline', 'candidate']
UNITS = ['producer', 'merge']
FILES = {f'{arm}.{unit}.{suffix}' for arm in ARMS for unit in UNITS for suffix in FORMATS}
PRODUCER = dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16, BLOCK_N=64, BLOCK_L=16,
    SPLITS=11, ROPE=False, WINDOWED=False, CHUNK_SIZE=8192, SEPARATE_MERGE=True,
    PAGE_LOOKAHEAD=True, num_warps=4, num_stages=3)
MERGE = dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16, SPLITS=11, BS=16, num_warps=4, num_stages=3)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def static_inventory(ttgir, ptx, sass):
    loop = next(line for line in ttgir.splitlines() if 'scf.for' in line and 'iter_args' in line)
    return dict(two_stage_e4m3_shared_allocations=re.findall(
        r'^.*ttg\.local_alloc : \(\) -> !ttg\.memdesc<2x64x64xf8E4M3FN.*$', ttgir, re.M),
        historical_loop_signature=loop,
        loop_carried_async_tokens=loop.count('!ttg.async.token'),
        ttgir_async_copy_operations=len(re.findall(r'\bttg\.async_copy_global_to_local\b', ttgir)),
        ttgir_async_commit_operations=len(re.findall(r'\bttg\.async_commit_group\b', ttgir)),
        ptx_wait_group_parameters=[int(value) for value in re.findall(r'cp\.async\.wait_group\s+(\d+)\s*;', ptx)],
        ptx_bf16_mma_instructions=len(re.findall(r'\bmma\.sync\.aligned\.[^\s]+\.f32\.bf16\.bf16\.f32\b', ptx)),
        ptx_shuffle_instructions=len(re.findall(r'\bshfl\.sync\.', ptx)),
        ptx_barrier_instructions=len(re.findall(r'\bbar\.sync\b', ptx)),
        sass_predicated_P0_fmul_addresses=re.findall(r'/\*([0-9a-f]+)\*/\s+@P0\s+FMUL\b', sass),
        ptx_local_accesses=len(re.findall(r'\b(?:ld|st)\.local\b', ptx)),
        sass_local_accesses=len(re.findall(r'\b(?:LDL|STL)(?:\.|\s)', sass)),
        highest_named_sass_register=max(int(value) for value in re.findall(r'\bR(\d+)\b', sass)),
        qualification='Static generated-code inventory; no dynamic frequency, throughput, achieved occupancy or causal compiler attribution')


def analyze():
    raw = (HERE/'result.json').read_bytes()
    launch_raw = (HERE/'launch-receipt.json').read_bytes()
    closure_raw = (HERE/'reconciliation.json').read_bytes()
    report, receipt, closure = map(json.loads, [raw, launch_raw, closure_raw])
    assert len(raw) <= 2*1024*1024 and receipt['report_sha256'] == sha(raw)
    assert receipt['source_commit'] == FREEZE and receipt['app_id'] == APP
    assert receipt['exit_code'] == 0 and receipt['worker_passed'] is True and receipt['worker_status'] == 'compiled'
    assert receipt['base_image'] == 'im-cPTbGpNiX3RarKResPgX7h'
    assert receipt['compile_only'] is True and receipt['kernel_launches'] == 0
    for key in ['correctness_measurement', 'performance_measurement', 'official_check', 'formal_entry']:
        assert receipt[key] is False, key
    assert receipt['artifact_files'] == 24 and receipt['artifacts_retrieved'] is True
    assert Decimal(receipt['paid_limit_usd']) == 0 and Decimal(receipt['cost_reserve_usd']) == Decimal('.40')
    assert receipt['resources'] == dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192],
        max_containers=1, worker_deadline=90, child_timeout=100, function_timeout=115,
        startup_timeout=20, watchdog=140, retry_policy=0, explicit_invocations=1)
    expected = {str((PROBE/name).relative_to(REPO)): value for name, value in SOURCES.items()}
    expected.update(LICENSES)
    expected.update({str((HERE/'worker.py').relative_to(REPO)): WORKER_SHA,
                     str((HERE/'controller.py').relative_to(REPO)): CONTROLLER_SHA})
    assert receipt['payload_sha256'] == expected
    for name, value in expected.items():
        assert sha((REPO/name).read_bytes()) == value, name
        assert sha(subprocess.check_output(['git', 'show', FREEZE+':'+name], cwd=REPO)) == value, name
    assert report['source_sha256'] == SOURCES and report['worker_sha256'] == WORKER_SHA
    assert report['verified_hashes'] == {**SOURCES, **LICENSES}
    assert report['status'] == 'compiled' and report['passed'] is True and report['compile_complete'] is True
    assert type(report['kernel_launches']) is int and report['kernel_launches'] == 0
    for key in ['correctness_tested', 'performance_tested', 'semantic_admission']:
        assert report[key] is False, key
    assert not report.get('timings') and not report.get('official_check_admitted')
    protocol = report['protocol']
    assert protocol['compile_only'] is True and protocol['kernel_launches'] == 0 and protocol['fixed_units'] == 4
    assert protocol['semantic_admission'] is False and protocol['deadline_seconds'] == 90
    assert protocol['cuobjdump_timeout_seconds'] == 10 and protocol['artifact_formats'] == FORMATS
    assert protocol['logical_fixture'] == dict(batch=32, q_len=1, heads='tp8', layer='full', ctx=80000,
        spread='agentic', page=1, page_table_columns=86001, qkv_width=2304,
        producer_grid=[32, 2, 11], merge_grid=[32, 16])
    assert report['producer_runtime_scalars'] == [1., 1e-5, 5.4730064863875, .125, 0, 1, 2304, 86001]
    assert report['environment']['gpu'] == 'NVIDIA H200' and report['environment']['capability'] == [9, 0]
    assert 0 < report['worker_wall_seconds'] < 90
    for value in report['pointer_metadata'].values():
        assert value['elements'] == 1 and value['alignment_mod16'] == 0
    spec = importlib.util.spec_from_file_location('common_pv_result_offline_decoder', HERE/'controller.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)  # Stdlib-only definitions; launch/auth are never called.
    bundle = report['artifact_bundle']
    decoded = controller.decode_bundle(bundle)
    assert set(decoded) == FILES and len(decoded) == 24
    directory = HERE/'artifacts'
    assert {p.name for p in directory.iterdir()} == FILES | {'manifest.json'}
    catalog = json.loads((directory/'manifest.json').read_bytes())
    assert catalog == {key: bundle[key] for key in ['encoding', 'files', 'compressed_sha256', 'uncompressed_sha256', 'uncompressed_size']}
    for name, value in decoded.items():
        path = directory/name
        assert not path.is_symlink() and path.read_bytes() == value, name
        assert bundle['files'][name] == dict(sha256=sha(value), size=len(value)), name
    assert len(report['kernels']) == 4
    units = {}
    for row in report['kernels']:
        key = row['arm'], row['unit']
        assert key in {(arm, unit) for arm in ARMS for unit in UNITS} and key not in units
        assert row['options'] == (PRODUCER if key[1] == 'producer' else MERGE)
        assert row['driver_module_loaded'] is True and row['kernel_launched'] is False and row['cuobjdump_exit_code'] == 0
        assert set(row['artifacts']) == set(FORMATS)
        for suffix, info in row['artifacts'].items():
            name = f'{key[0]}.{key[1]}.{suffix}'
            assert info == dict(file=name, size=len(decoded[name]), sha256=sha(decoded[name]))
        assert row['shared_memory_bytes'] == row['metadata']['shared']
        assert row['metadata']['num_warps'] == 4
        assert all(type(row[field]) is int and row[field] >= 0 for field in ['registers', 'shared_memory_bytes', 'driver_spills_raw'])
        units[key] = row
    resources = {arm: {unit: {field: units[(arm, unit)][field]
        for field in ['registers', 'shared_memory_bytes', 'driver_spills_raw']} for unit in UNITS} for arm in ARMS}
    assert resources['baseline']['producer'] == dict(registers=96, shared_memory_bytes=20480, driver_spills_raw=0)
    assert resources['candidate']['producer'] == resources['baseline']['producer']
    assert resources['baseline']['merge'] == resources['candidate']['merge'] == dict(registers=27, shared_memory_bytes=1024, driver_spills_raw=0)
    merge_same = decoded['baseline.merge.sass'] == decoded['candidate.merge.sass']
    assert merge_same and sha(decoded['baseline.merge.sass']) == '69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4'
    inventory = {arm: static_inventory(*(decoded[f'{arm}.producer.{suffix}'].decode() for suffix in ['ttgir', 'ptx', 'sass'])) for arm in ARMS}
    assert [len(inventory[arm]['two_stage_e4m3_shared_allocations']) for arm in ARMS] == [2, 2]
    assert [inventory[arm]['ttgir_async_copy_operations'] for arm in ARMS] == [6, 6]
    assert [inventory[arm]['ttgir_async_commit_operations'] for arm in ARMS] == [6, 6]
    assert [inventory[arm]['ptx_wait_group_parameters'] for arm in ARMS] == [[2, 0], [2, 0]]
    assert [inventory[arm]['loop_carried_async_tokens'] for arm in ARMS] == [4, 4]
    assert [inventory[arm]['ptx_bf16_mma_instructions'] for arm in ARMS] == [10, 10]
    assert [inventory[arm]['ptx_barrier_instructions'] for arm in ARMS] == [18, 18]
    assert [inventory[arm]['ptx_shuffle_instructions'] for arm in ARMS] == [38, 41]
    assert inventory['candidate']['sass_predicated_P0_fmul_addresses'] == [
        '2770', '2790', '27a0', '27b0', '27d0', '27e0', '27f0', '2810']
    candidate_ttgir = decoded['candidate.producer.ttgir'].decode()
    condition = re.search(r'(?P<value>%\w+) = scf\.if %all_alpha_one_\d+.*?^\s+} loc', candidate_ttgir, re.M | re.S)
    assert condition and 'arith.mulf %arg23' in condition.group() and 'scf.yield %arg23' in condition.group()
    assert 'tt.dot' not in condition.group()
    common_pv = next(line for line in candidate_ttgir[condition.end():].splitlines() if 'tt.dot ' in line)
    assert re.search(re.escape(condition['value'])+r', inputPrecision', common_pv)
    assert 'bf16' in common_pv
    assert 'arith.cmpf une, %alpha_' in candidate_ttgir and 'arith.andi %row_ok, %all_alpha_one' in candidate_ttgir
    assert all(inventory[arm]['ptx_local_accesses'] == inventory[arm]['sass_local_accesses'] == 0 for arm in ARMS)
    increased_registers = resources['candidate']['producer']['registers'] > resources['baseline']['producer']['registers']
    assert not increased_registers
    assert closure['app']['app_id'] == APP and closure['app']['state'] == 'stopped' and int(closure['app']['tasks']) == 0
    assert closure['stage_closed'] is True and closure['all_apps_stopped_zero_tasks'] is True
    assert all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in closure['all_apps'])
    assert closure['app'] in closure['all_apps']
    assert Decimal(closure['billing']['billed_cost']) == Decimal(closure['paid_limit_usd']) == 0
    return dict(source_freeze=FREEZE, app_id=APP, source_sha256=SOURCES,
        raw_result_sha256=sha(raw), launch_receipt_sha256=sha(launch_raw), reconciliation_sha256=sha(closure_raw),
        compilation_complete=True, artifact_files_verified=24, artifact_file_bytes=sum(map(len, decoded.values())),
        raw_bundle_json_bytes=bundle['uncompressed_size'], compressed_bundle_bytes=bundle['compressed_size'],
        resources=resources, same_geometry=True, merge_sass_byte_identical=merge_same,
        merge_sass_sha256=sha(decoded['baseline.merge.sass']), producer_static_inventory=inventory,
        resource_gate_passed=True, compile_mechanism_gate_passed=True,
        onward_admission=False, execution_requires_separate_review=True, candidate_rejected=False,
        rejection_reasons=[], predication_issue_cycle_savings_established=False,
        kernel_launches=0, correctness_tested=False, frequency_tested=False, performance_tested=False, official_check=False,
        semantic_equivalence_established=False, selected_working_source_sha256=SOURCES['baseline.py'],
        environment=report['environment'], worker_wall_seconds=report['worker_wall_seconds'],
        stage_closed=True, app_stopped=True, app_tasks=0, closure_observed_utc=closure['observed_utc'],
        actual_paid_usd=str(Decimal(closure['billing']['billed_cost'])),
        qualifications=['The compile mechanism/resource gates pass, while numerical/frequency/performance stages remain untested and require separate review.',
            'Registers/shared/spills and generated-code counts are static, not measured latency, achieved occupancy or causal allocator proof.',
            'Both generated pipelines retain two-stage K/V buffers, four loop-carried async tokens, six copy/commit sites and wait parameters[2,0].',
            'Unchanged merge SASS is byte-identical; other debug-bearing merge artifact bytes need not match.',
            'Eight predicated FMULs suppress arithmetic/destination updates on the fast edge; issue/control-cycle savings are unproven and three additional shuffles add static overhead.',
            'Shutdown/billed0 refer to the retained independent cutoff, not a fresh service query.'])


if __name__ == '__main__':
    result = analyze()
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['compilation_complete', 'artifact_files_verified', 'resources',
        'merge_sass_byte_identical', 'resource_gate_passed', 'onward_admission', 'candidate_rejected', 'kernel_launches', 'stage_closed']}, indent=2))
