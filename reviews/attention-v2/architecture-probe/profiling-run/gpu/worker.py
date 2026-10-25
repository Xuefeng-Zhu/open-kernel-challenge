"""One activity-only H200 session for two unchanged B32/L8 paths.

No Modal client, credentials, network, installation, timing benchmark or
algorithm edit. Optional installed-NCU collection is bounded and separate from
activity validity. Candidate/support bytes are verified before import.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

CASE = dict(batch=32, q_len=8, ctx=80000, heads='tp8', layer='full', page=1,
            kv_scale=1., spread='agentic', dist='coherent', seed=135701)
ARMS = ['fa3', 'bf16_custom']
RANGES = {'fa3': 'PROFILE_ONE_FA3', 'bf16_custom': 'PROFILE_ONE_BF16_CUSTOM'}
SOURCES = {
    'candidate.py': 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',
    'candidatecustomhelper.py': '59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82',
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
MAX_SUMMARY = 256*1024
MAX_TRACE = 8*1024*1024-MAX_SUMMARY
MAX_COMPACT_EVENTS = 512
MAX_COUNTER_SUMMARY = 64*1024
COUNTER_SUPPORT = ['counters.py', 'counter_target.py', 'inventory.json']
COUNTER_ARTIFACTS = ['fa3.ncu-rep', 'bf16_custom.ncu-rep',
    'fa3-counter-check.json', 'bf16_custom-counter-check.json']
GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
REQUIRED_FA3_STAGES = {'qk_norm', 'fp8_conversion', 'kv_cache_store',
    'length_schedule_prepare', 'fa3_mainloop', 'fa3_combine', 'output_norm_gate'}


class SummaryLimitError(RuntimeError):
    pass


def bounded_summary(report):
    raw = json.dumps(report, separators=(',', ':'))+'\n'
    overflow = len(raw.encode()) > MAX_SUMMARY
    if overflow and report.get('activity_passed') and 'activity' in report:
        original = report.get('counter_branch', {})
        encoded = json.dumps(original, sort_keys=True).encode()
        reduced = {k: original[k] for k in ('status', 'sessions_attempted', 'sessions_upper_bound',
            'error_type', 'error') if k in original}
        reduced.update(summary_reduced=True, original_summary_bytes=len(encoded),
            original_summary_sha256=hashlib.sha256(encoded).hexdigest())
        for key in ('error', 'query_error'):
            if key in original:
                text = str(original[key]);reduced[key] = text[:512]+text[-512:]
        reduced['records'] = [{k: row[k] for k in ('arm', 'status', 'exit_code',
            'report_sha256', 'report_preserved') if k in row} for row in original.get('records', [])[:2]]
        report['counter_branch'] = reduced
        report['counter_summary_reduced_for_size'] = True
        raw = json.dumps(report, separators=(',', ':'))+'\n'
        if len(raw.encode()) > MAX_SUMMARY:
            for arm in report['activity'].get('arms', {}).values():
                arm.pop('events', None)
                arm['event_table_omitted_for_size_raw_trace_retained'] = True
            raw = json.dumps(report, separators=(',', ':'))+'\n'
        overflow = len(raw.encode()) > MAX_SUMMARY
    if overflow:
        observed = len(raw.encode())
        report.pop('activity', None)
        report.update(status='incomplete_or_failed', passed=False,
            compact_activity_omitted=True, summary_overflow_observed_bytes=observed,
            error_type='SummaryLimitError',
            error='Compact summary exceeded256KiB; raw trace retained, compact activity omitted')
        raw = json.dumps(report, separators=(',', ':'))+'\n'
        if len(raw.encode()) > MAX_SUMMARY:
            raise SummaryLimitError('Non-activity summary exceeded256KiB')
    return raw, overflow


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def interval_union(intervals):
    """Length of interval union, not the sum of overlapping activity times."""
    values = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    if not values:
        return 0.
    total, low, high = 0., values[0][0], values[0][1]
    for a, b in values[1:]:
        if a > high:
            total += high-low
            low, high = a, b
        else:
            high = max(high, b)
    return total+high-low


def event_stage(arm, name, category):
    if arm == 'bf16_custom':
        if '_fused' in name:
            return 'integrated_producer_and_merge'
        if category == 'gpu_memset' or 'memset' in name.lower():
            return 'zero_or_memset_not_independently_attributed'
    else:
        for token, stage in [('_qk_rms_norm', 'qk_norm'),
            ('ConvertToFloat8E4M3', 'fp8_conversion'),
            ('store_kvcache', 'kv_cache_store'),
            ('prepare_varlen_num_blocks', 'length_schedule_prepare'),
            ('FlashAttnFwdSm90', 'fa3_mainloop'),
            ('FlashAttnFwdCombine', 'fa3_combine'),
            ('norm_gate', 'output_norm_gate')]:
            if token in name:
                return stage
    return 'other'


def compact_trace(trace):
    """Map GPU activity to fenced CPU ranges; preserve raw trace separately.

    A synchronize immediately before each range excludes preceding restores and
    eviction. Synchronize inside the range fences that arm's graph completion.
    CPU annotation span includes synchronization/profiler overhead, not latency.
    """
    events = trace.get('traceEvents')
    if not isinstance(events, list):
        raise ValueError('Chrome traceEvents absent')
    complete = [e for e in events if e.get('ph') == 'X' and isinstance(e.get('ts'), (int, float))
                and isinstance(e.get('dur'), (int, float)) and e['dur'] >= 0]
    result = dict(trace_events=len(events), arms={}, exact_cta_trace=False,
        hardware_counters_collected=False,
        qualification='Instrumented activity timestamps only; CPU spans and GPU intervals are not benchmark timings. No DRAM/tensor/stall/residency counter inference.')
    for arm, label in RANGES.items():
        markers = [e for e in complete if e.get('name') == label
                   and e.get('cat') in {'user_annotation', 'cpu_op'}]
        if len(markers) != 1:
            raise ValueError('Require exactly one CPU range for '+arm)
        marker = markers[0]
        low, high = marker['ts'], marker['ts']+marker['dur']
        within = [e for e in complete if e is not marker and e['ts'] >= low
                  and e['ts']+e['dur'] <= high and e.get('cat') in
                  (GPU_CATEGORIES | {'cuda_runtime', 'cuda_driver'})]
        if len(within) > MAX_COMPACT_EVENTS:
            raise ValueError('Compact event-count bound exceeded')
        gpu = [e for e in within if e.get('cat') in GPU_CATEGORIES]
        kernels = [e for e in gpu if e.get('cat') == 'kernel']
        if not kernels:
            raise ValueError('No GPU kernel attributed to '+arm)
        required = 'FlashAttnFwdSm90' if arm == 'fa3' else '_fused'
        matched = [e for e in kernels if required in str(e.get('name', ''))]
        if len(matched) != 1:
            raise ValueError('Require exactly one mainloop/integrated kernel for '+arm)
        gpu_intervals = [(e['ts'], e['ts']+e['dur']) for e in gpu]
        kernel_intervals = [(e['ts'], e['ts']+e['dur']) for e in kernels]
        compact = []
        stages = {}
        for e in within:
            cat, name = e.get('cat'), str(e.get('name', ''))
            stage = event_stage(arm, name, cat)
            compact.append(dict(name=name, category=cat, stage=stage, ts_us=e['ts'],
                duration_us=e['dur'], pid=e.get('pid'), tid=e.get('tid'), args=e.get('args', {})))
            if cat in GPU_CATEGORIES:
                stages.setdefault(stage, []).append((e['ts'], e['ts']+e['dur']))
        if arm == 'fa3' and not REQUIRED_FA3_STAGES <= set(stages):
            raise ValueError('Missing expected FA3 stage coverage: '+','.join(sorted(REQUIRED_FA3_STAGES-set(stages))))
        result['arms'][arm] = dict(range_name=label, cpu_range_start_us=low,
            cpu_range_end_us=high, cpu_annotated_span_us=high-low,
            cpu_span_qualification='Includes graph launch, wait and profiler instrumentation; not unprofiled host launch latency.',
            gpu_activity_span_us=max(b for _, b in gpu_intervals)-min(a for a, _ in gpu_intervals),
            gpu_activity_union_us=interval_union(gpu_intervals),
            gpu_kernel_union_us=interval_union(kernel_intervals),
            gpu_activity_duration_sum_us=sum(e['dur'] for e in gpu),
            sum_qualification='Diagnostic sum may double-count overlaps; use union/span for activity layout, never as paired benchmark latency.',
            stage_union_us={s: interval_union(v) for s, v in stages.items()},
            event_count=len(compact), kernel_count=len(kernels), mainloop_matches=1,
            events=compact)
    return result


def optional_counters(root, dest, deadline_monotonic):
    remaining = deadline_monotonic-time.monotonic()
    if remaining < 25:
        return dict(status='not_attempted_insufficient_time', sessions_attempted=0,
            remaining_seconds=remaining, reason='Need at least25s; no job extension')
    try:
        module = load(Path(root)/'counters.py', 'profile_counters')
        result = module.run_counters(Path(root), Path(dest), deadline_monotonic-3)
        if not isinstance(result, dict):
            raise TypeError('Counter result must be a dictionary')
        sessions = result.get('sessions_attempted')
        if not isinstance(sessions, int) or not 0 <= sessions <= 2:
            raise ValueError('Counter result has invalid session count')
        if len(json.dumps(result).encode()) > MAX_COUNTER_SUMMARY:
            raise ValueError('Counter summary exceeds64KiB; activity remains valid')
        return result
    except Exception as exc:
        return dict(status='unavailable_or_partial', sessions_attempted=None,
            sessions_upper_bound=2, error_type=type(exc).__name__,
            error=str(exc)[:1000], qualification='Optional counter failure does not invalidate completed activity; no bottleneck inferred')


def run(output_dir, deadline):
    root = Path(__file__).resolve().parent
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='Single CPU/CUDA activity session with optional bounded counters; not benchmark',
        status='starting', passed=False, started_utc=now(), case=CASE, arms=ARMS,
        source_sha256=SOURCES, worker_sha256=sha(__file__), checks=[], phases=[],
        activity_sessions=0, profiled_graph_replays={}, hardware_counter_sessions=0,
        activity_passed=False,
        counter_branch=dict(status='not_attempted', availability='requires installed-tool/permission/time gate'),
        benchmark_claim=False, exact_cta_trace=False,
        protocol=dict(seed=135701, graph_captures=2, independent_graph_pools=True,
            profiler_activities=['CPU', 'CUDA'], record_shapes=False, profile_memory=False,
            with_stack=False, max_activity_sessions=1, profiled_replays_per_arm=1,
            cache_eviction_bytes=256*1024*1024, oracle_checks_before_and_after=True,
            immutable_inputs_and_untouched_rows_checked=True, concurrency=False,
            algorithms_unchanged=True, no_official_check=True))

    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        raw, overflow = bounded_summary(report)
        tmp = dest/'summary.tmp'
        tmp.write_text(raw)
        tmp.replace(dest/'summary.json')
        if overflow:
            raise SummaryLimitError('Compact summary exceeded256KiB; raw trace retained')

    def gate():
        save()
        if time.monotonic()-started >= deadline:
            raise TimeoutError('Activity worker deadline')

    def alarm(*_):
        raise TimeoutError('Activity worker alarm')

    def phase(name, before):
        report['phases'].append(dict(name=name, wall_seconds=time.monotonic()-before))
        gate()

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(deadline))
    save()
    try:
        assert not os.environ.get('TRITON_INTERPRET')
        for name, expected in SOURCES.items():
            assert (root/name).is_file() and not (root/name).is_symlink()
            assert sha(root/name) == expected, 'Frozen source changed: '+name
        for name in COUNTER_SUPPORT:
            assert (root/name).is_file() and not (root/name).is_symlink(), 'Missing reviewed counter support: '+name
        report['profiling_support_sha256'] = {name: sha(root/name) for name in COUNTER_SUPPORT}
        import torch as t
        import triton
        assert t.cuda.is_available() and 'H200' in t.cuda.get_device_name()
        assert t.profiler.ProfilerActivity.CUDA in t.profiler.supported_activities()
        props = t.cuda.get_device_properties(0)
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=4)
        report['environment'] = dict(python=platform.python_version(), torch=t.__version__,
            triton=triton.__version__, cuda=t.version.cuda, gpu=props.name,
            sm_count=props.multi_processor_count, capability=t.cuda.get_device_capability(),
            driver=driver.stdout.strip() if driver.returncode == 0 else None,
            supported_activities=sorted(str(a) for a in t.profiler.supported_activities()))
        sys.path.insert(0, str(root/'official'))
        sys.path.insert(0, str(root))
        h = load(root/'modal_worker.py', 'activity_helpers')
        ref = load(root/'official/reference.py', 'activity_reference')
        candidate = load(root/'candidate.py', 'candidate')
        custom = load(root/'candidatecustomhelper.py', 'candidatecustomhelper')
        functions = dict(fa3=candidate.kernel_fn, bf16_custom=custom.bf16_custom_fn)
        phase('imports_and_hashes', started)
        before = time.monotonic()
        fresh = ref.generate_input(**CASE, device='cuda')
        report['input_metadata'] = dict(actual_lengths=fresh['seq_lens'].cpu().tolist(),
            config=fresh['config'], page_table_shape=list(fresh['page_table'].shape),
            tensor_metadata={k: dict(shape=list(v.shape), dtype=str(v.dtype), device=str(v.device))
                for k, v in fresh.items() if isinstance(v, t.Tensor)})
        runners = {}
        for arm in ARMS:
            work = h.clone_inputs(fresh)
            graph, out = h.capture(functions[arm], work, fresh)
            runners[arm] = dict(work=work, graph=graph, out=out,
                addresses={k: v.data_ptr() for k, v in work.items() if isinstance(v, t.Tensor)},
                out_address=out.data_ptr())
        phase('generate_and_capture', before)

        def checked(arm, label):
            r = runners[arm]
            raw = ref.collect(r['work'], r['out'])
            stable = r['addresses'] == {k: v.data_ptr() for k, v in r['work'].items() if isinstance(v, t.Tensor)}
            stable = stable and r['out'].data_ptr() == r['out_address']
            intact, why = h.unchanged(r['work'], fresh)
            okay, message = ref.check_implementation(r['work'], raw)
            exact = bool(t.equal(raw, r['expected'])) if 'expected' in r else None
            if label == 'pre_profile':
                r['expected'] = raw.clone()
            result = dict(arm=arm, label=label, passed=bool(okay and stable and intact and exact is not False),
                oracle_called=True, oracle_message=message, stable_addresses=stable,
                immutable_inputs_and_untouched_rows=intact, immutable_message=why,
                exact_output_newkv_equal_pre_profile=exact)
            report['checks'].append(result)
            gate()
            if not result['passed']:
                raise AssertionError('Official correctness/immutability gate failed: '+arm+'/'+label)

        before = time.monotonic()
        for arm in ARMS:
            gate()
            r = runners[arm]
            h.restore(r['work'], fresh)
            r['graph'].replay()
            t.cuda.synchronize()
            checked(arm, 'pre_profile')
        phase('pre_profile_oracles', before)
        before = time.monotonic()
        eviction = t.ones(256*1024*1024//4, device='cuda')
        trace_path = dest/'activity-trace.json'
        gate()
        report['activity_sessions'] += 1
        with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                t.profiler.ProfilerActivity.CUDA], record_shapes=False,
                profile_memory=False, with_stack=False) as profile:
            for arm in ARMS:
                r = runners[arm]
                with t.profiler.record_function('RESTORE_EVICT_'+arm.upper()):
                    h.restore(r['work'], fresh)
                    eviction.sum()
                    t.cuda.synchronize()
                label = RANGES[arm]
                with t.profiler.record_function(label):
                    t.cuda.nvtx.range_push(label)
                    try:
                        r['graph'].replay()
                        t.cuda.synchronize()
                    finally:
                        t.cuda.nvtx.range_pop()
                report['profiled_graph_replays'][arm] = report['profiled_graph_replays'].get(arm, 0)+1
        profile.export_chrome_trace(str(trace_path))
        assert trace_path.is_file() and not trace_path.is_symlink()
        if trace_path.stat().st_size > MAX_TRACE:
            trace_path.unlink()
            raise RuntimeError('Raw activity trace exceeds7936KiB transport allowance; oversized file removed')
        report['artifacts'] = {'activity-trace.json': dict(bytes=trace_path.stat().st_size,
            sha256=sha(trace_path), qualification='Unmodified complete raw Chrome trace')}
        phase('single_activity_session', before)
        before = time.monotonic()
        for arm in ARMS:
            gate()
            checked(arm, 'post_profile')
        phase('post_profile_oracles', before)
        gate()
        trace = json.loads(trace_path.read_text())
        report['activity'] = compact_trace(trace)
        assert report['activity_sessions'] == 1 and report['profiled_graph_replays'] == {a: 1 for a in ARMS}
        assert len(report['checks']) == 4 and all(x['passed'] for x in report['checks'])
        report.update(status='activity_complete', passed=True, activity_passed=True)
        save()
        counter_before = time.monotonic()
        report['counter_branch'] = optional_counters(root, dest, started+deadline)
        report['hardware_counter_sessions'] = report['counter_branch'].get('sessions_attempted')
        report['hardware_counter_sessions_upper_bound'] = 2
        report['counter_wall_seconds'] = time.monotonic()-counter_before
        for name in COUNTER_ARTIFACTS:
            path = dest/name
            if path.is_file() and not path.is_symlink():
                report['artifacts'][name] = dict(bytes=path.stat().st_size, sha256=sha(path),
                    qualification='Raw optional counter artifact; interpret only under counter status/validation')
        report.update(status='complete', passed=True)
        save()
        return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed', error_type=type(exc).__name__,
            error=str(exc)[:1200], traceback=traceback.format_exc()[-4000:])
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/profiling-run')
    parser.add_argument('--deadline', type=float, default=165)
    args = parser.parse_args()
    assert 1 <= args.deadline <= 165
    raise SystemExit(run(args.output_dir, args.deadline))
