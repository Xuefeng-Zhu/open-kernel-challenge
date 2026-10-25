"""Offline verification of a rejected, source-bound compile-only alpha probe."""
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
FREEZE = '99185609d7924d18bb07de950c3d689a9ec9e649'
APP = 'ap-uY5K1rBNFRR2K8rdo2F8tw'
WORKER_SHA = '9e0377dc696518225b5091f65fe7f72ac5e96e6c2b539c399f13bf337a80a585'
CONTROLLER_SHA = '2ddfab210d7a589826674990a380d657a078b794ac73e175fb0fc5d27e9978bf'
SOURCES = {
    'baseline.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'candidate.py': 'e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764',
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
    return dict(two_stage_e4m3_shared_allocations=re.findall(
        r'^.*ttg\.local_alloc : \(\) -> !ttg\.memdesc<2x64x64xf8E4M3FN.*$', ttgir, re.M),
        ttgir_async_copy_operations=len(re.findall(r'\bttg\.async_copy_global_to_local\b', ttgir)),
        ttgir_async_commit_operations=len(re.findall(r'\bttg\.async_commit_group\b', ttgir)),
        ptx_wait_group_parameters=[int(value) for value in re.findall(r'cp\.async\.wait_group\s+(\d+)\s*;', ptx)],
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
    spec = importlib.util.spec_from_file_location('alpha_result_offline_decoder', HERE/'controller.py')
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
    assert resources['candidate']['producer'] == dict(registers=128, shared_memory_bytes=20480, driver_spills_raw=0)
    assert resources['baseline']['merge'] == resources['candidate']['merge'] == dict(registers=27, shared_memory_bytes=1024, driver_spills_raw=0)
    merge_same = decoded['baseline.merge.sass'] == decoded['candidate.merge.sass']
    assert merge_same and sha(decoded['baseline.merge.sass']) == '69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4'
    inventory = {arm: static_inventory(*(decoded[f'{arm}.producer.{suffix}'].decode() for suffix in ['ttgir', 'ptx', 'sass'])) for arm in ARMS}
    assert [len(inventory[arm]['two_stage_e4m3_shared_allocations']) for arm in ARMS] == [2, 1]
    assert [inventory[arm]['ttgir_async_copy_operations'] for arm in ARMS] == [6, 3]
    assert [inventory[arm]['ttgir_async_commit_operations'] for arm in ARMS] == [6, 3]
    assert [inventory[arm]['ptx_wait_group_parameters'] for arm in ARMS] == [[2, 0], [1, 0]]
    assert all(inventory[arm]['ptx_local_accesses'] == inventory[arm]['sass_local_accesses'] == 0 for arm in ARMS)
    increased_registers = resources['candidate']['producer']['registers'] > resources['baseline']['producer']['registers']
    assert increased_registers
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
        resource_gate_passed=False, onward_admission=False, candidate_rejected=True,
        rejection_reasons=['Producer register count increases96to128, violating the predeclared no-increase gate.',
            'Generated two-stage E4M3 shared allocations/copy pipeline differ; static buffering preservation is not established.'],
        kernel_launches=0, correctness_tested=False, frequency_tested=False, performance_tested=False, official_check=False,
        semantic_equivalence_established=False, selected_working_source_sha256=SOURCES['baseline.py'],
        environment=report['environment'], worker_wall_seconds=report['worker_wall_seconds'],
        stage_closed=True, app_stopped=True, app_tasks=0, closure_observed_utc=closure['observed_utc'],
        actual_paid_usd=str(Decimal(closure['billing']['billed_cost'])),
        qualifications=['Compile completion is separate from a failed resource-admission gate.',
            'Registers/shared/spills and generated-code counts are static, not measured latency, achieved occupancy or causal allocator proof.',
            'Wait-group parameter changes2to1; both PTX files still contain two wait-group instructions including final wait0.',
            'Unchanged merge SASS is byte-identical; other debug-bearing merge artifact bytes need not match.',
            'No frequency, native correctness, timing or official check was performed; do not execute this rejected variant.',
            'Shutdown/billed0 refer to the retained independent cutoff, not a fresh service query.'])


if __name__ == '__main__':
    result = analyze()
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['compilation_complete', 'artifact_files_verified', 'resources',
        'merge_sass_byte_identical', 'resource_gate_passed', 'onward_admission', 'candidate_rejected', 'kernel_launches', 'stage_closed']}, indent=2))
