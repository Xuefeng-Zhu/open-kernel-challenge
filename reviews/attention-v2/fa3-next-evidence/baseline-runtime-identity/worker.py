"""Bounded selected-baseline identity/activity diagnostic, never a candidate.

No Modal/auth/network/install operations. The remote controller supplies
unchanged source files next to this worker; import is deferred until source and
installed-tool inventory have passed. CPU metadata precedes CUDA initialization.
An independently reviewed optional counter capability gate runs only after core
checks, with its own eager child targets, one metric and remaining-time gate.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

SOURCES = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
}
CASES = {f'full_verify_b{b}': dict(batch=b, q_len=8, ctx=80000,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic',
    dist='coherent', seed=135741) for b in (16, 32)}
LABELS = {name: 'PROFILE_SELECTED_' + name.upper() for name in CASES}
MAX_SUMMARY = 1024 * 1024
MAX_TRACE = 32 * 1024 * 1024
MAX_TRACE_GZ = 6 * 1024 * 1024
GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def interval_union(intervals):
    rows = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    if not rows:
        return 0.
    total, lo, hi = 0., rows[0][0], rows[0][1]
    for a, b in rows[1:]:
        if a > hi:
            total += hi-lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + hi-lo


def stage(name):
    for token, label in (('_fa3_prepare_fused', 'fused_preparation'),
        ('prepare_varlen_num_blocks', 'scheduler_preparation'),
        ('FlashAttnFwdSm90', 'fa3_mainloop'),
        ('FlashAttnFwdCombine', 'fa3_combine'), ('norm_gate', 'output_norm_gate')):
        if token in name:
            return label
    return 'other'


def compact_trace(trace):
    """Keep complete symbols/resources, but infer no opaque template semantics."""
    events = trace.get('traceEvents')
    if not isinstance(events, list):
        raise ValueError('Raw traceEvents missing')
    complete = [e for e in events if e.get('ph') == 'X'
        and isinstance(e.get('ts'), (int, float))
        and isinstance(e.get('dur'), (int, float)) and e['dur'] >= 0]
    result = {'trace_event_count': len(events), 'cases': {},
        'hardware_counters_collected': False, 'exact_cta_residency_trace': False,
        'qualification': 'Instrumented activity only. No DRAM/tensor/stall/achieved occupancy inference.'}
    for name, label in LABELS.items():
        ranges = [e for e in complete if e.get('name') == label
            and e.get('cat') in {'user_annotation', 'cpu_op'}]
        if len(ranges) != 1:
            raise ValueError('Require one fenced range: ' + name)
        low, high = ranges[0]['ts'], ranges[0]['ts'] + ranges[0]['dur']
        within = [e for e in complete if e.get('cat') in GPU_CATEGORIES
            and e['ts'] >= low and e['ts']+e['dur'] <= high]
        if not within or len(within) > 128:
            raise ValueError('Missing or too many case GPU events: ' + name)
        kernels = [e for e in within if e.get('cat') == 'kernel']
        mainloops = [e for e in kernels if 'FlashAttnFwdSm90' in str(e.get('name', ''))]
        if len(mainloops) != 1:
            raise ValueError('Expected one actual FA3 mainloop: ' + name)
        stages = {}
        rows = []
        for event in within:
            symbol = str(event.get('name', ''))
            category = event['cat']
            label_stage = stage(symbol)
            stages.setdefault(label_stage, []).append((event['ts'], event['ts']+event['dur']))
            rows.append(dict(name=symbol, category=category, stage=label_stage,
                ts_us=event['ts'], duration_us=event['dur'], pid=event.get('pid'),
                tid=event.get('tid'), args=event.get('args', {})))
        intervals = [(e['ts'], e['ts']+e['dur']) for e in within]
        result['cases'][name] = dict(range_start_us=low, range_end_us=high,
            cpu_annotated_span_us=high-low,
            gpu_activity_span_us=max(b for _, b in intervals)-min(a for a, _ in intervals),
            gpu_activity_union_us=interval_union(intervals),
            stage_union_us={k: interval_union(v) for k, v in stages.items()},
            events=rows, full_mainloop_symbols=[e['name'] for e in mainloops],
            resolved_tile='UNKNOWN until full symbol template is matched to exact definition',
            resolved_pack_gqa='UNKNOWN until template/API semantics are proven',
            resolved_split_count='UNKNOWN: no interpretation from grid or opaque dimensions',
            internal_qk_pv_dtype='UNKNOWN: callable dtype and type names are evidence, not disassembly')
    return result


def command_version(executable):
    path = shutil.which(executable)
    if path is None:
        return {'installed_on_path': False}
    resolved = Path(path).resolve()
    result = {'installed_on_path': True, 'path': str(resolved),
        'bytes': resolved.stat().st_size}
    try:
        call = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=3)
        text = call.stdout+call.stderr
        if len(text.encode()) > 4096:
            raise ValueError('Version output exceeds bound')
        result.update(version_exit_code=call.returncode, version_output=text)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        result['version_status'] = type(exc).__name__
    return result


def tool_inventory():
    result = {'collected_before_cuda_initialization': True,
        'tools': {name: command_version(name) for name in ('ncu', 'nsys', 'cuobjdump', 'nvdisasm')},
        'packages': {}, 'hardware_counter_sessions': 0,
        'counter_policy': 'This inventory runs no NCU session and makes no permission change or installation; two historical timeouts are not a permission-denial finding.'}
    for name in ('torch', 'triton', 'sglang', 'sgl-kernel', 'nvidia-cuda-cupti-cu12', 'nvidia-cuda-cupti-cu13'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    path = Path('/proc/driver/nvidia/params')
    if path.is_file() and path.stat().st_size <= 65536:
        with path.open() as stream:
            raw = stream.read(65537)
        if len(raw) <= 65536:
            matches = [line for line in raw.splitlines() if 'RmProfilingAdminOnly' in line]
            result['profiling_admin_only_setting'] = matches
        else:
            result['profiling_admin_only_setting'] = 'UNKNOWN: procfile exceeds 64KiB bound'
    result['counter_permission_tested'] = False
    return result


def mapped_flash_binaries():
    """Read only bounded mappings of the vendor extension, omitting ASLR addresses."""
    path = Path('/proc/self/maps')
    if not path.is_file():
        return {'status': 'unavailable', 'binaries': []}
    with path.open() as stream:
        raw = stream.read(2*1024*1024+1)
    if len(raw) > 2*1024*1024:
        return {'status': 'over_2MiB_bound', 'binaries': []}
    binaries = {}
    for line in raw.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or 'flash_ops' not in Path(parts[5]).name:
            continue
        candidate = Path(parts[5])
        if 'site-packages' not in candidate.parts or not candidate.is_file():
            continue
        status = candidate.stat()
        inode_matches = str(status.st_ino) == parts[4]
        binaries[str(candidate)] = dict(path=str(candidate), mapped_inode=int(parts[4]),
            file_inode=status.st_ino, inode_matches=inode_matches,
            bytes=status.st_size, qualification='Loaded mapping/file identity; not judge identity or source revision')
    return {'status': 'bounded_maps_read', 'binaries': list(binaries.values())}


def make_aux_adapter(selected_source, namespace):
    """Derive a NEW untimed adapter. Never replace candidate/wrapper functions.

    The original function's AST is retained except its name, the single public
    wrapper return flag/assignment, and the final Python return container.
    The wrapper AST must separately prove that flag never reaches native fwd.
    """
    tree = ast.parse(selected_source)
    source = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == '_fa3_full_verify_fused')
    fn = ast.parse(ast.unparse(source)).body[0]
    fn.name = '_diagnostic_aux_full'
    count = 0
    body = []
    for node in fn.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'flash_attn_with_kvcache'):
            if len(node.targets) != 1 or ast.unparse(node.targets[0]) != 'out':
                raise ValueError('Unexpected selected FA3 assignment')
            if any(k.arg == 'return_softmax_lse' for k in node.value.keywords):
                raise ValueError('Selected return flag unexpectedly supplied')
            node.targets = [ast.Name(id='_diagnostic_aux_return', ctx=ast.Store())]
            node.value.keywords.append(ast.keyword(arg='return_softmax_lse', value=ast.Constant(value=True)))
            body += [node, ast.parse('out = _diagnostic_aux_return[0]').body[0]]
            count += 1
        elif isinstance(node, ast.Return):
            node.value = ast.Tuple(elts=[node.value,
                ast.Name(id='_diagnostic_aux_return', ctx=ast.Load())], ctx=ast.Load())
            body.append(node)
        else:
            body.append(node)
    if count != 1:
        raise ValueError('Expected one selected public FA3 call')
    fn.body = body
    new_tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    adapter_source = ast.unparse(new_tree)+'\n'
    copied_namespace = dict(namespace)
    exec(compile(new_tree, '<reviewed-separate-aux-adapter>', 'exec'), copied_namespace)
    return copied_namespace[fn.name], adapter_source


def run(output_dir, seconds):
    root = Path(__file__).resolve().parent
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='Selected Attention baseline runtime identity and activity; not optimization',
        status='starting', passed=False, worker_sha256=sha(__file__), started_utc=now(),
        sources=SOURCES, cases={}, checks=[], phases=[], activity_sessions=0,
        selected_candidate_changed=False, official_judge_binary_identity='UNKNOWN: Modal image only',
        hardware_counter_sessions=0, benchmark_claim=False,
        protocol=dict(original_oracle=True, selected_whole_kernel=True,
            before_samples=3, after_samples=3, warmups=2, eviction_bytes=256*1024*1024,
            auxiliary_replays_untimed=True, max_activity_sessions=1,
            profiler_overhead_subtracted=False, selected_pack_split_scheduler_unchanged=True))

    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        raw = json.dumps(report, separators=(',', ':'))+'\n'
        if len(raw.encode()) > MAX_SUMMARY:
            raise ValueError('Summary exceeds 1MiB; no evidence truncation')
        temporary = dest/'summary.tmp'
        temporary.write_text(raw)
        temporary.replace(dest/'summary.json')

    def gate():
        save()
        if time.monotonic()-started >= seconds:
            raise TimeoutError('Diagnostic worker deadline')

    def alarm(*_):
        raise TimeoutError('Diagnostic worker alarm')

    def phase(name, before):
        report['phases'].append({'name': name, 'wall_seconds': time.monotonic()-before})
        gate()

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(seconds))
    save()
    try:
        for name, digest in SOURCES.items():
            path = root/name
            if not path.is_file() or path.is_symlink() or sha(path) != digest:
                raise ValueError('Source binding mismatch: '+name)
        extra_files = ['cpu_identity.py', 'counter_gate.py'] + [name for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md')
            if (root/name).is_file()]
        report['verified_hashes'] = {name: sha(root/name) for name in [*SOURCES, *extra_files]}
        if os.environ.get('TRITON_INTERPRET'):
            raise ValueError('Interpreter mode unsupported')
        report['tool_inventory'] = tool_inventory()
        identity = load(root/'cpu_identity.py', 'baseline_cpu_identity')
        report['cpu_file_identity'] = identity.run(root/'selected.py', root/'reference.py', seconds=25)
        phase('cpu_tools_and_binary_identity', started)
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name():
            raise ValueError('Expected H200')
        if t.profiler.ProfilerActivity.CUDA not in t.profiler.supported_activities():
            raise ValueError('CUDA profiler activity unavailable; no fallback or installation')
        properties = t.cuda.get_device_properties(0)
        report['environment'] = dict(python=platform.python_version(), torch=t.__version__,
            triton=triton.__version__, cuda=t.version.cuda, gpu=properties.name,
            sm_count=properties.multi_processor_count, capability=t.cuda.get_device_capability(),
            supported_activities=sorted(str(a) for a in t.profiler.supported_activities()))
        sys.path.insert(0, str(root))
        helpers = load(root/'modal_worker.py', 'baseline_identity_helpers')
        reference = load(root/'reference.py', 'baseline_identity_reference')
        selected = load(root/'selected.py', 'baseline_identity_selected')
        import sgl_kernel.flash_attn as wrapper
        wrapper_path = Path(wrapper.__file__)
        wrapper_raw = wrapper_path.read_bytes()
        if len(wrapper_raw) > 64*1024:
            raise ValueError('Wrapper source exceeds 64KiB')
        wrapper_ast = identity.wrapper_ast(wrapper_raw)
        report['loaded_fa3'] = dict(wrapper_path=str(wrapper_path), wrapper_sha256=sha(wrapper_path),
            wrapper_source_utf8=wrapper_raw.decode(), wrapper_ast=wrapper_ast,
            native_schema=str(t.ops.sgl_kernel.fwd.default._schema),
            binding_function=selected.flash_attn_with_kvcache.__module__+'.'+selected.flash_attn_with_kvcache.__name__)
        if selected.flash_attn_with_kvcache is not wrapper.flash_attn_with_kvcache:
            raise ValueError('Selected public wrapper binding differs')
        report['loaded_flash_libraries'] = sorted(str(p) for p in t.ops.loaded_libraries
            if 'flash' in str(p).lower() or 'sgl' in str(p).lower())
        report['mapped_flash_binaries'] = mapped_flash_binaries()
        runners = {}
        before = time.monotonic()
        for name, case in CASES.items():
            gate()
            fresh = reference.generate_input(**case, device='cuda')
            if fresh['scheduler_metadata'] is not None:
                raise ValueError('Verify must retain normal per-call scheduler preparation')
            work = helpers.clone_inputs(fresh)
            graph, out = helpers.capture(selected.kernel_fn, work, fresh)
            metadata = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                stride=list(value.stride())) for key, value in fresh.items() if isinstance(value, t.Tensor)}
            page_raw = fresh['page_table'].contiguous().view(t.uint8).cpu().numpy().tobytes()
            report['cases'][name] = dict(case=case, config=fresh['config'],
                actual_lengths=fresh['seq_lens'].cpu().tolist(), tensor_metadata=metadata,
                page_table_sha256=hashlib.sha256(page_raw).hexdigest(),
                scheduler_metadata=None, logical_packed_rows_per_kv_head=8*8,
                logical_packed_rows_qualification='Query count times GQA group, not resolved kernel tile or PackGQA flag',
                before_us=[], after_us=[], warmup_us=[], auxiliary={})
            runners[name] = dict(fresh=fresh, work=work, graph=graph, out=out,
                addresses={key: value.data_ptr() for key, value in work.items() if isinstance(value, t.Tensor)},
                out_address=out.data_ptr())
        phase('generate_and_capture', before)
        eviction = t.ones(256*1024*1024//4, device='cuda')

        def checked(name, label, oracle=False):
            runner = runners[name]
            raw = reference.collect(runner['work'], runner['out'])
            intact, why = helpers.unchanged(runner['work'], runner['fresh'])
            stable = runner['addresses'] == {key: value.data_ptr() for key, value in runner['work'].items()
                if isinstance(value, t.Tensor)} and runner['out'].data_ptr() == runner['out_address']
            exact = t.equal(raw.contiguous().reshape(-1).view(t.uint8),
                runner['expected'].contiguous().reshape(-1).view(t.uint8)) if 'expected' in runner else True
            okay, message = reference.check_implementation(runner['work'], raw) if oracle else (True, 'Not repeated: byte invariant against original-oracle-checked result')
            if oracle and 'expected' not in runner:
                runner['expected'] = raw.clone()
            item = dict(case=name, phase=label, original_oracle_called=oracle,
                original_oracle_passed=bool(okay), message=message, stable_addresses=stable,
                immutable_inputs_and_nonnew_cache_unchanged=intact, immutable_message=why,
                exact_output_and_newkv_bytes=bool(exact), passed=bool(okay and stable and intact and exact))
            report['checks'].append(item)
            gate()
            if not item['passed']:
                raise AssertionError('Baseline invariant failure: '+name+'/'+label)

        before = time.monotonic()
        for name, runner in runners.items():
            helpers.restore(runner['work'], runner['fresh'])
            runner['graph'].replay()
            t.cuda.synchronize()
            checked(name, 'initial_oracle', oracle=True)
            for i in range(5):
                helpers.restore(runner['work'], runner['fresh'])
                value = helpers.timed(runner['graph'], eviction)
                report['cases'][name]['warmup_us' if i < 2 else 'before_us'].append(value)
                checked(name, 'warmup_'+str(i) if i < 2 else 'before_'+str(i-2))
        phase('original_oracles_and_unprofiled_before', before)
        before = time.monotonic()
        trace_path = dest/'activity-trace.json'
        report['activity_sessions'] += 1
        with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                t.profiler.ProfilerActivity.CUDA], record_shapes=False,
                profile_memory=False, with_stack=False) as profile:
            for name, runner in runners.items():
                with t.profiler.record_function('RESTORE_EVICT_'+name.upper()):
                    helpers.restore(runner['work'], runner['fresh'])
                    eviction.sum()
                    t.cuda.synchronize()
                with t.profiler.record_function(LABELS[name]):
                    runner['graph'].replay()
                    t.cuda.synchronize()
        profile.export_chrome_trace(str(trace_path))
        if trace_path.stat().st_size > MAX_TRACE:
            raise ValueError('Raw activity trace exceeds 32MiB; no truncation')
        trace_raw = trace_path.read_bytes()
        compressed = gzip.compress(trace_raw, mtime=0)
        if len(compressed) > MAX_TRACE_GZ:
            raise ValueError('Compressed trace exceeds 6MiB; no truncation')
        (dest/'activity-trace.json.gz').write_bytes(compressed)
        report['trace_artifacts'] = {'activity-trace.json': dict(bytes=len(trace_raw),
            sha256=hashlib.sha256(trace_raw).hexdigest()),
            'activity-trace.json.gz': dict(bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest())}
        report['activity'] = compact_trace(json.loads(trace_raw))
        phase('one_activity_session', before)
        before = time.monotonic()
        for name, runner in runners.items():
            checked(name, 'activity_result_oracle', oracle=True)
            for i in range(3):
                helpers.restore(runner['work'], runner['fresh'])
                report['cases'][name]['after_us'].append(helpers.timed(runner['graph'], eviction))
                checked(name, 'after_'+str(i))
            item = report['cases'][name]
            item['before_median_us'] = statistics.median(item['before_us'])
            item['after_median_us'] = statistics.median(item['after_us'])
            item['after_over_before_median_ratio'] = item['after_median_us']/item['before_median_us']
            item['timing_qualification'] = 'Whole selected CUDA graph, restore/evict outside event interval; 3 samples each, no performance-variant or significance claim'
        phase('post_activity_oracles_and_unprofiled_after', before)
        before = time.monotonic()
        aux_safe = (wrapper_ast.get('native_call_count') == 1
            and 'return_softmax_lse' in inspect.signature(wrapper.flash_attn_with_kvcache).parameters
            and wrapper_ast.get('return_flag_in_native_arguments') is False)
        if aux_safe:
            adapter, adapter_source = make_aux_adapter((root/'selected.py').read_text(), selected.__dict__)
            report['auxiliary_adapter'] = dict(source=adapter_source,
                sha256=hashlib.sha256(adapter_source.encode()).hexdigest(),
                native_argument_equivalence_basis='Fresh wrapper AST: return flag absent from native fwd args; selected helper AST differs only in return flag/containers')
            for name, runner in runners.items():
                gate()
                auxwork = helpers.clone_inputs(runner['fresh'])
                auxgraph, result = helpers.capture(adapter, auxwork, runner['fresh'])
                helpers.restore(auxwork, runner['fresh'])
                auxgraph.replay()
                t.cuda.synchronize()
                auxout, returned = result
                raw = reference.collect(auxwork, auxout)
                exact = t.equal(raw.contiguous().reshape(-1).view(t.uint8),
                    runner['expected'].contiguous().reshape(-1).view(t.uint8))
                cache_equal = all(t.equal(auxwork[key].view(t.uint8), runner['work'][key].view(t.uint8))
                    for key in ('k_cache', 'v_cache'))
                intact, why = helpers.unchanged(auxwork, runner['fresh'])
                if not (exact and cache_equal and intact):
                    raise AssertionError('Untimed auxiliary adapter byte invariance failed: '+name)
                report['cases'][name]['auxiliary'] = dict(status='validated_untimed_replay',
                    final_output_and_newkv_exact=True, full_caches_exact=True,
                    immutable_inputs_and_nonnew_cache_unchanged=intact, immutable_message=why,
                    returned=[dict(index=i, shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()))
                        if isinstance(value, t.Tensor) else dict(index=i, python_type=type(value).__name__)
                        for i, value in enumerate(returned)],
                    opaque_rest_meaning='UNKNOWN until exact schema/source field correspondence is proven')
                del auxgraph, auxwork, result
        else:
            report['auxiliary_adapter'] = {'status': 'not_attempted',
                'reason': 'Fresh wrapper did not prove return flag absent from native arguments'}
        phase('separate_untimed_auxiliary_replays', before)
        report['core_identity_activity_passed'] = True
        gate()
        remaining = seconds-(time.monotonic()-started)
        if remaining >= 50:
            try:
                counter = load(root/'counter_gate.py', 'baseline_identity_counter_gate')
                report['counter_capability_gate'] = counter.run_gate(root, dest, remaining)
            except TimeoutError:
                raise
            except Exception as exc:
                report['counter_capability_gate'] = dict(state='unexpected_gate_error',
                    error_type=type(exc).__name__, error=str(exc)[:1000],
                    hardware_counter_sessions=None, permission_denial_observed=None,
                    reason='No retry or inference. Core identity/activity remains independently recorded; outer app cleanup is still mandatory.')
            report['hardware_counter_sessions'] = report['counter_capability_gate'].get('hardware_counter_sessions')
        else:
            report['counter_capability_gate'] = dict(status='not_attempted_insufficient_remaining_time',
                remaining_seconds=remaining, required_seconds=50, sessions_attempted=0,
                reason='Core identity/activity takes priority; no worker/deadline extension')
        gate()
        report.update(status='complete', passed=True, completed_utc=now())
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.deadline <= 180:
        parser.error('Deadline must be 1..180 seconds')
    raise SystemExit(run(args.output_dir, args.deadline))
