"""One selected B32/L8 FA3 baseline: graph timings and separate eager counters.

No candidate, permission, credential, network, install or official-run operation.
All vendor imports and CUDA work are deferred until source/CLI gates pass.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import sys
import time
import traceback

CASE = dict(batch=32, q_len=8, ctx=80000, heads='tp8', layer='full',
    page=1, kv_scale=1., spread='agentic', dist='coherent', seed=135741)
CASE_NAME = 'full_verify_b32'
LABEL = 'PROFILE_SELECTED_FULL_VERIFY_B32'
METRICS = (
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
)
METRIC_BASES = (
    'sm__pipe_tensor_cycles_active', 'dram__throughput',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active', 'sm__warps_active',
)
EXPECTED = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'counter_helper.py': 'd984cbd44691cfead1c889cbb2ae8b0bde2e11e0a1d703bbdd78a7ecf1b30637',
    'cli_contract.py': '7391280b4577a2d1f242bdffdc42777a26a180b1cdcd6e021dc47c2866c1137a',
    'trace_helpers.py': 'd89e2f6dac4d8afd9105b713ce255407df8e9253c330a1e727d8193a220c5ec8',
    'cpu_identity.py': '29e4f87f294142a38f926a7bf6f57e8032d615be7ab0da4a17e2f0962a50ec92',
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
FILES = (*EXPECTED, 'csv_helpers.py')
MAX_SUMMARY = 1024*1024
MAX_TRACE = 32*1024*1024
MAX_TRACE_GZ = 6*1024*1024
MAX_FINGERPRINT = 1536*1024*1024
WRAPPER_SHA = '610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bind_sources(root):
    result = {}
    for name in FILES:
        path = root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 256*1024:
            raise ValueError('Payload source path/size refused: '+name)
        result[name] = sha(path)
        if name in EXPECTED and result[name] != EXPECTED[name]:
            raise ValueError('Frozen source binding differs: '+name)
    return result


def selected_route(raw):
    """Bind the first scored dispatch branch without executing/patching it."""
    tree = ast.parse(raw)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'kernel_fn')
    branch = next(node for node in fn.body if isinstance(node, ast.If))
    if len(branch.body) != 1 or not isinstance(branch.body[0], ast.Return):
        raise ValueError('Unexpected selected dispatch')
    if ast.unparse(branch.body[0].value) != '_fa3_full_verify_fused(data)':
        raise ValueError('Selected first branch no longer uses fused full FA3')
    test = ast.unparse(branch.test)
    if '(32, 8)' not in test or "cfg['window'] == 0" not in test or "cfg['rope_theta'] == 0" not in test:
        raise ValueError('Selected B32/L8 full dispatch gate differs')
    return dict(helper='_fa3_full_verify_fused', condition=test,
        function_ast_sha256=hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest(),
        source_only=True, native_pack_split_override=False)


def query_argv(ncu, base):
    return [str(ncu), '--query-metrics', '--query-metrics-mode', 'suffix', '--metrics', base]


def profile_argv(ncu, root):
    return [str(ncu), '--replay-mode', 'kernel', '--target-processes', 'application-only',
        '--kernel-name-base', 'demangled', '--kernel-name', 'regex:.*FlashAttnFwdSm90.*',
        '--launch-skip', '1', '--launch-count', '1', '--metrics', ','.join(METRICS),
        '--page', 'raw', '--csv', '--clock-control', 'none',
        sys.executable, str(root/'worker.py'), '--counter-child']


def metric_supported(text, name):
    return bool(re.search(r'(?<![\w.])'+re.escape(name)+r'(?![\w.])', text))


def progress_rows(text):
    rows = []
    for line in text.splitlines():
        if line.startswith('FA3_PROGRESS_JSON='):
            rows.append(json.loads(line.split('=', 1)[1]))
    return rows


def mapped_flash_files():
    """Record only vendor flash mappings, omitting ASLR addresses and other maps."""
    path = Path('/proc/self/maps')
    with path.open('rb') as stream:
        raw = stream.read(2*1024*1024+1)
    if len(raw) > 2*1024*1024:
        raise ValueError('Process mapping byte bound')
    result = {}
    for line in raw.decode().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        name = parts[5]
        if not name.startswith('/opt/sglang/') or 'flash_ops' not in Path(name).name or not name.endswith('.so'):
            continue
        p = Path(name)
        stat = p.stat()
        result[name] = dict(path=name, mapped_inode=int(parts[4]), file_inode=stat.st_ino,
            inode_matches=int(parts[4]) == stat.st_ino, bytes=stat.st_size)
    if not result or not all(v['inode_matches'] for v in result.values()):
        raise ValueError('Loaded FA3 mapping/file identity unresolved')
    return sorted(result.values(), key=lambda v: v['path'])


def tensor_digest(tensor):
    import torch
    flat = tensor.contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), 8*1024*1024):
        digest.update(flat[offset:offset+8*1024*1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def fixture_fingerprint(data):
    import torch
    tensors = {key: value for key, value in data.items() if isinstance(value, torch.Tensor)}
    size = sum(value.numel()*value.element_size() for value in tensors.values())
    if size > MAX_FINGERPRINT:
        raise ValueError('Input fingerprint byte bound')
    return dict(config=data['config'], tensor_bytes_hashed=size,
        tensors={key: dict(shape=list(value.shape), stride=list(value.stride()),
            dtype=str(value.dtype), sha256=tensor_digest(value)) for key, value in tensors.items()},
        scheduler_metadata_is_none=data['scheduler_metadata'] is None,
        actual_lengths=data['seq_lens'].cpu().tolist())


def runtime_modules(root):
    import torch
    import triton
    sys.path.insert(0, str(root))
    helpers = load(root/'modal_worker.py', 'fa3_counter_harness')
    reference = load(root/'reference.py', 'fa3_counter_reference')
    selected = load(root/'selected.py', 'fa3_counter_selected')
    import sgl_kernel.flash_attn as wrapper
    if sha(wrapper.__file__) != WRAPPER_SHA or selected.flash_attn_with_kvcache is not wrapper.flash_attn_with_kvcache:
        raise ValueError('Selected public FA3 wrapper binding changed')
    props = torch.cuda.get_device_properties(0)
    if 'H200' not in props.name or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('H200 SM90 required')
    meta = dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        gpu=props.name, sm_count=props.multi_processor_count,
        wrapper_path=wrapper.__file__, wrapper_sha256=sha(wrapper.__file__),
        native_schema=str(torch.ops.sgl_kernel.fwd.default._schema),
        mapped_flash_files=mapped_flash_files())
    return helpers, reference, selected, meta


def check_output(reference, helpers, work, fresh, out, expected=None, oracle=False):
    import torch
    raw = reference.collect(work, out)
    intact, detail = helpers.unchanged(work, fresh)
    if raw.dtype != torch.uint8:
        raise ValueError('Original reference.collect must return raw uint8 bytes')
    exact = expected is None or torch.equal(raw.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8))
    okay, message = reference.check_implementation(work, raw) if oracle else (True, 'Byte invariance against original-oracle-checked result')
    result = dict(passed=bool(okay and intact and exact), original_oracle_called=oracle,
        original_oracle_passed=bool(okay), oracle_message=message,
        immutable_inputs_and_nonnew_cache_unchanged=intact, immutable_message=detail,
        exact_output_and_newkv_bytes=bool(exact), output_newkv_sha256=tensor_digest(raw))
    if not result['passed']:
        raise AssertionError('Selected baseline oracle/cache/byte check failed: '+str(result))
    return result, raw


def counter_child(root):
    def emit(phase, **extra):
        print('FA3_PROGRESS_JSON='+json.dumps(dict(phase=phase, **extra)), flush=True)
    emit('child_started')
    bind_sources(root)
    emit('source_bound')
    helpers, reference, selected, environment = runtime_modules(root)
    emit('vendor_and_cuda_ready', environment=environment)
    fresh = reference.generate_input(**CASE, device='cuda')
    fingerprint = fixture_fingerprint(fresh)
    emit('fixture_ready', fingerprint=fingerprint)
    work = helpers.clone_inputs(fresh)
    expected = None
    for phase in ('pre', 'measured', 'post'):
        emit(phase+'_starting')
        helpers.restore(work, fresh)
        out = selected.kernel_fn(work)
        import torch
        torch.cuda.synchronize()
        check, raw = check_output(reference, helpers, work, fresh, out, expected,
            oracle=phase != 'measured')
        if expected is None:
            expected = raw.clone()
        emit(phase, **check)
    return 0


def run(root, output, seconds):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='Selected B32/L8 full FA3 baseline graph/counter diagnostic',
        status='starting', passed=False, worker_sha256=sha(__file__), case=CASE,
        selected_candidate_changed=False, official_check=False, formal_entry=False,
        security_changes=False, performance_variant=False, activity_sessions=0,
        counter_sessions_attempted=0, commands=[], checks=[], before_us=[], after_us=[], post_counter_us=[],
        metrics=list(METRICS), benchmark_claim='Unprofiled graph samples only; no official-score or optimization claim')
    def save():
        report['worker_elapsed_seconds'] = time.monotonic()-started
        raw = json.dumps(report, separators=(',', ':'))+'\n'
        if len(raw.encode()) > MAX_SUMMARY:
            raise ValueError('Summary exceeds 1MiB; preserve no truncated claim')
        path = output/'summary.tmp'
        path.write_text(raw)
        path.replace(output/'summary.json')
    def mark(phase):
        report['phase'] = phase
        report.setdefault('progress', []).append(dict(phase=phase,
            elapsed_seconds=time.monotonic()-started, utc=datetime.now(timezone.utc).isoformat()))
        save()
        if time.monotonic()-started >= seconds:
            raise TimeoutError('Worker deadline')
    def alarm(*_):
        raise TimeoutError('190-second worker deadline')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(seconds)
    save()
    try:
        report['verified_hashes'] = bind_sources(root)
        report['selected_route'] = selected_route((root/'selected.py').read_text())
        helper = load(root/'counter_helper.py', 'fa3_command_helper')
        cli = load(root/'cli_contract.py', 'fa3_cli_contract')
        csv_reader = load(root/'csv_helpers.py', 'fa3_csv_helpers')
        ncu = shutil.which('ncu')
        if ncu is None:
            raise ValueError('Existing NCU unavailable; no installation')
        mark('ncu_version_help')
        for name, argv, timeout in [('version', [ncu, '--version'], 3), ('help', [ncu, '--help'], 5)]:
            call = helper._command(argv, output, name, timeout)
            report['commands'].append(call)
            if call['state'] != 'complete':
                raise ValueError('Existing NCU version/help failed')
        help_text = report['commands'][-1].get('stdout', '')
        report['installed_cli_contract'] = cli.validate_installed_help(help_text,
            profile_argv(ncu, root), query_argv(ncu, METRIC_BASES[0]))
        mark('metric_support_queries')
        report['metric_support'] = {}
        for i, (metric, base) in enumerate(zip(METRICS, METRIC_BASES)):
            call = helper._command(query_argv(ncu, base), output, 'query-'+str(i), 5)
            report['commands'].append(call)
            supported = call['state'] == 'complete' and metric_supported(call.get('stdout', ''), metric)
            report['metric_support'][metric] = supported
            save()
            if not supported:
                report.update(status='required_metric_unsupported', reason=metric+' was not confirmed; no fallback metric or NCU session')
                save()
                return 1
        mark('cpu_binary_identity')
        identity = load(root/'cpu_identity.py', 'fa3_cpu_identity')
        report['cpu_file_identity'] = identity.run(root/'selected.py', root/'reference.py', seconds=25)
        import torch
        helpers, reference, selected, environment = runtime_modules(root)
        report['environment'] = environment
        report['precision'] = dict(qkv_input='BF16', external_q_k_v='FP8 E4M3 per unchanged source and wrapper inputs',
            output='BF16', internal_qk_pv='UNKNOWN machine-instruction/accumulation sequence',
            pack_gqa_input=None, num_splits_input=0, scheduler_metadata_input=None,
            resolved_pack_gqa='UNKNOWN', resolved_split_count='UNKNOWN')
        mark('fixture_and_graph_capture')
        fresh = reference.generate_input(**CASE, device='cuda')
        if fresh['scheduler_metadata'] is not None:
            raise ValueError('Verify metadata must remain None')
        report['fixture'] = fixture_fingerprint(fresh)
        work = helpers.clone_inputs(fresh)
        graph, out = helpers.capture(selected.kernel_fn, work, fresh)
        addresses = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
        output_address = out.data_ptr()
        helpers.restore(work, fresh)
        graph.replay()
        torch.cuda.synchronize()
        check, raw = check_output(reference, helpers, work, fresh, out, oracle=True)
        expected = raw.clone()
        report['checks'].append(dict(phase='initial_graph_oracle', **check))
        report['graph_output_newkv_sha256'] = check['output_newkv_sha256']
        eviction = torch.ones(256*1024*1024//4, device='cuda')
        def graph_check(phase, oracle=False):
            item, _ = check_output(reference, helpers, work, fresh, out, expected, oracle)
            item['stable_addresses'] = addresses == {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)} and output_address == out.data_ptr()
            if not item['stable_addresses']:
                raise AssertionError('Graph addresses changed')
            report['checks'].append(dict(phase=phase, **item))
            save()
        mark('unprofiled_before')
        for i in range(5):
            helpers.restore(work, fresh)
            value = helpers.timed(graph, eviction)
            report.setdefault('warmup_us', []).append(value) if i < 2 else report['before_us'].append(value)
            graph_check('warmup-'+str(i) if i < 2 else 'before-'+str(i-2))
        mark('one_graph_activity_session')
        report['activity_sessions'] = 1
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False) as profile:
            helpers.restore(work, fresh)
            eviction.sum()
            torch.cuda.synchronize()
            with torch.profiler.record_function(LABEL):
                graph.replay()
                torch.cuda.synchronize()
        trace_path = output/'activity-trace.json'
        profile.export_chrome_trace(str(trace_path))
        trace_raw = trace_path.read_bytes()
        if len(trace_raw) > MAX_TRACE:
            raise ValueError('Trace exceeds32MiB')
        compressed = gzip.compress(trace_raw, mtime=0)
        if len(compressed) > MAX_TRACE_GZ:
            raise ValueError('Compressed trace exceeds6MiB')
        (output/'activity-trace.json.gz').write_bytes(compressed)
        report['trace_artifacts'] = {name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
            for name, value in [('activity-trace.json', trace_raw), ('activity-trace.json.gz', compressed)]}
        graph_check('post_activity_graph_oracle', oracle=True)
        mark('unprofiled_after')
        for i in range(3):
            helpers.restore(work, fresh)
            report['after_us'].append(helpers.timed(graph, eviction))
            graph_check('after-'+str(i))
        report['whole_call'] = dict(before_median_us=statistics.median(report['before_us']),
            after_median_us=statistics.median(report['after_us']),
            before_range_us=[min(report['before_us']), max(report['before_us'])],
            after_range_us=[min(report['after_us']), max(report['after_us'])],
            eviction_bytes=256*1024*1024, empty_graph_subtraction=False,
            before_after_scope='Before and after the one Torch activity session; both precede NCU',
            qualification='Whole unchanged kernel_fn CUDA graph replays. Instrumented event/counter durations are separate.')
        trace_helpers = load(root/'trace_helpers.py', 'fa3_trace_helpers')
        report['graph_activity'] = trace_helpers.attribute(json.loads(trace_raw), CASE_NAME)
        activity = report['graph_activity']
        if len(activity['full_mainloop_symbols']) != 1 or len(activity['mainloop_resources']) != 1:
            raise ValueError('One exact selected FA3 mainloop identity required')
        remaining = seconds-(time.monotonic()-started)
        if remaining < 85:
            report.update(status='insufficient_remaining_for_counter', remaining_seconds=remaining,
                reason='No NCU child started; preserve complete graph evidence and deadline')
            save()
            return 1
        mark('eager_ncu_session')
        report['counter_sessions_attempted'] = 1
        save()
        command = helper._command(profile_argv(ncu, root), output, 'ncu-baseline', 75)
        report['commands'].append(command)
        save()
        remaining = seconds-(time.monotonic()-started)
        if remaining < 5:
            report.update(status='post_counter_confirmation_not_reached',
                remaining_seconds=remaining,
                reason='NCU has terminated; insufficient remaining time for original-oracle and graph confirmation')
            save()
            return 1
        mark('post_counter_graph_confirmation')
        helpers.restore(work, fresh)
        graph.replay()
        torch.cuda.synchronize()
        graph_check('post_counter_graph_oracle', oracle=True)
        for i in range(3):
            helpers.restore(work, fresh)
            report['post_counter_us'].append(helpers.timed(graph, eviction))
            graph_check('post_counter-'+str(i))
        report['whole_call'].update(post_counter_median_us=statistics.median(report['post_counter_us']),
            post_counter_range_us=[min(report['post_counter_us']), max(report['post_counter_us'])],
            post_counter_confirmation_complete=True)
        save()
        rows = progress_rows(command.get('stdout', ''))
        checks = [row for row in rows if row.get('phase') in ('pre', 'measured', 'post')]
        report['counter_child_progress'] = rows
        report['counter_permission_denial_observed'] = 'ERR_NVGPUCTRPERM' in command.get('stdout', '')+'\n'+command.get('stderr', '')
        if command['state'] != 'complete' or command['returncode'] != 0:
            report.update(status='counter_session_incomplete', reason=command['state'])
            save()
            return 1
        if [row.get('phase') for row in checks] != ['pre', 'measured', 'post'] or not all(row.get('passed') is True for row in checks):
            raise ValueError('NCU child correctness pre/measured/post incomplete')
        parsed = csv_reader.parse_ncu_csv(command.get('stdout', ''), METRICS)
        report['counter_report'] = parsed
        resource = activity['mainloop_resources'][0]
        proxy = csv_reader.compare_kernel_identity(parsed, activity['full_mainloop_symbols'][0], resource['grid'], resource['block'])
        child_fixture = next(row['fingerprint'] for row in rows if row.get('phase') == 'fixture_ready')
        child_environment = next(row['environment'] for row in rows if row.get('phase') == 'vendor_and_cuda_ready')
        proxy.update(exact_symbol_grid_block_match=proxy['matches'],
            full_input_fingerprint_match=child_fixture == report['fixture'],
            eager_graph_output_newkv_bytes_match=all(row['output_newkv_sha256'] == report['graph_output_newkv_sha256'] for row in checks),
            wrapper_and_mapped_binary_match=all(child_environment[k] == environment[k] for k in
                ('wrapper_sha256', 'mapped_flash_files', 'torch', 'triton', 'cuda', 'gpu', 'sm_count')),
            opaque_split_policy='UNKNOWN; identity checks do not prove identical eager/graph internal scheduling state')
        proxy['necessary_identity_checks_passed'] = all(proxy[k] for k in
            ('exact_symbol_grid_block_match', 'full_input_fingerprint_match', 'eager_graph_output_newkv_bytes_match', 'wrapper_and_mapped_binary_match'))
        report['graph_proxy_identity'] = proxy
        report['conclusion'] = dict(status='requires_counter_review' if proxy['necessary_identity_checks_passed'] else 'inconclusive_eager_graph_proxy_mismatch',
            qualification='Four counters on one eager replayed mainloop; no register-only diagnosis, precise residency trace, or official-score claim. Even identity parity is not proof of resolved split equality.',
            next_measurement='Choose a distinct measurement after reviewing actual counter values; do not introduce a candidate from static resource counts alone.')
        mark('complete')
        report.update(status='complete', passed=True)
        save()
        return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed', error_type=type(exc).__name__,
            error=str(exc)[:1200], traceback=traceback.format_exc()[-5000:])
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counter-child', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--deadline', type=int, default=190)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.counter_child:
        raise SystemExit(counter_child(root))
    if args.output_dir is None or not 1 <= args.deadline <= 190:
        parser.error('Output directory and deadline1..190 required')
    raise SystemExit(run(root, args.output_dir, args.deadline))
