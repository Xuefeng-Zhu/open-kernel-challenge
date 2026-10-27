"""Offline salvage of a partial diagnostic; never edits or upgrades raw status.

Attribution follows the host cudaGraphLaunch correlation and separately checks
the profiler's GPU annotation. CPU and GPU timestamp containment is not assumed.
No source, service, kernel or credential operation is performed.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

NAMES = ('full_verify_b16', 'full_verify_b32')
GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
MAX_TRACE = 32*1024*1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def interval_union(intervals):
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


def phase(name):
    for token, label in (('_fa3_prepare_fused', 'fused_preparation'),
        ('prepare_varlen_num_blocks', 'scheduler_preparation'),
        ('FlashAttnFwdSm90', 'fa3_mainloop'),
        ('FlashAttnFwdCombine', 'fa3_combine'), ('norm_gate', 'output_norm_gate')):
        if token in name:
            return label
    return 'other'


def numeric_complete(event):
    return event.get('ph') == 'X' and isinstance(event.get('ts'), (int, float)) \
        and isinstance(event.get('dur'), (int, float)) and event['dur'] >= 0


def overlaps(events):
    """Report apparent same-stream overlaps without inferring real concurrency."""
    ordered = sorted(events, key=lambda event: event['ts'])
    result = []
    for i, first in enumerate(ordered):
        for second in ordered[i+1:]:
            if first.get('args', {}).get('stream') != second.get('args', {}).get('stream'):
                continue
            overlap = min(first['ts']+first['dur'], second['ts']+second['dur']) - max(first['ts'], second['ts'])
            if overlap > .001:
                result.append(dict(first=first['name'], second=second['name'],
                    apparent_overlap_us=overlap, stream=first.get('args', {}).get('stream')))
    return result


def attribute(trace, name):
    label = 'PROFILE_SELECTED_'+name.upper()
    events = [event for event in trace['traceEvents'] if numeric_complete(event)]
    host = [event for event in events if event.get('cat') == 'user_annotation'
        and event.get('name') == label]
    if len(host) != 1:
        raise ValueError('Need one host annotation for '+name)
    host = host[0]
    launch = [event for event in events if event.get('cat') == 'cuda_runtime'
        and str(event.get('name', '')).startswith('cudaGraphLaunch')
        and event.get('pid') == host.get('pid') and event.get('tid') == host.get('tid')
        and host['ts'] <= event['ts']
        and event['ts']+event['dur'] <= host['ts']+host['dur']]
    if len(launch) != 1:
        raise ValueError('Need one host graph launch for '+name)
    correlation = launch[0].get('args', {}).get('correlation')
    if correlation is None:
        raise ValueError('Graph launch correlation absent')
    gpu = [event for event in events if event.get('cat') in GPU_CATEGORIES
        and event.get('args', {}).get('correlation') == correlation]
    if not gpu:
        raise ValueError('No GPU events have graph launch correlation')
    annotations = [event for event in events if event.get('cat') == 'gpu_user_annotation'
        and event.get('name') == label and event.get('args', {}).get('External id')
        == host.get('args', {}).get('External id')]
    if len(annotations) != 1:
        raise ValueError('Need matching GPU annotation/external id for '+name)
    annotation = annotations[0]
    if not all(annotation['ts']-.002 <= event['ts']
            and event['ts']+event['dur'] <= annotation['ts']+annotation['dur']+.002 for event in gpu):
        raise ValueError('Graph-correlated events escape matching GPU annotation')
    kernels = [event for event in gpu if event.get('cat') == 'kernel']
    mainloops = [event for event in kernels if phase(event['name']) == 'fa3_mainloop']
    rows = [dict(name=event['name'], category=event['cat'], phase=phase(event['name']),
        ts_us=event['ts'], duration_us=event['dur'], args=event.get('args', {})) for event in gpu]
    host_containment = [event for event in events if event.get('cat') in GPU_CATEGORIES
        and event['ts'] >= host['ts'] and event['ts']+event['dur'] <= host['ts']+host['dur']]
    intervals = [(event['ts'], event['ts']+event['dur']) for event in gpu]
    by_phase = {}
    for event in gpu:
        by_phase.setdefault(phase(event['name']), []).append((event['ts'], event['ts']+event['dur']))
    # This is a literal in the symbol, not a guess about the meaning of booleans.
    shape_pattern = (r'cute::tuple<cute::C<64>, cute::C<128>, cute::C<64> >')
    return dict(attribution='Host graph-launch correlation plus matching GPU annotation/external id',
        host_annotation=host, graph_launch=launch[0], gpu_annotation=annotation,
        graph_ids=sorted({event.get('args', {}).get('graph id') for event in gpu}),
        graph_node_ids=[event.get('args', {}).get('graph node id') for event in gpu],
        correlation=correlation, external_id=host.get('args', {}).get('External id'),
        kernel_count=len(kernels), phase_kernel_counts=dict(Counter(phase(event['name']) for event in kernels)),
        full_mainloop_symbols=[event['name'] for event in mainloops],
        mainloop_shape_tuple_literal=[64, 128, 64] if mainloops
            and all(shape_pattern in event['name'] for event in mainloops) else None,
        shape_qualification='Literal in executed Modal symbol; source typedef/boolean/ISA semantics are not inferred.',
        mainloop_resources=[event.get('args', {}) for event in mainloops],
        gpu_activity_span_us=max(b for _, b in intervals)-min(a for a, _ in intervals),
        gpu_activity_union_us=interval_union(intervals),
        instrumented_phase_union_us={label: interval_union(v) for label, v in by_phase.items()},
        apparent_same_stream_overlaps=overlaps(kernels),
        original_cpu_containment_mainloop_count=sum(phase(event['name']) == 'fa3_mainloop' for event in host_containment),
        original_cpu_containment_foreign_correlations=sorted({event.get('args', {}).get('correlation')
            for event in host_containment if event.get('args', {}).get('correlation') != correlation}),
        cpu_gpu_timestamp_containment_valid=all(event in host_containment for event in gpu),
        events=rows,
        timing_qualification='Instrumented activity durations only. Cross-clock offsets/apparent same-stream overlaps preclude concurrency or critical-path inference.')


def analyze(result, trace):
    report = dict(kind='Offline interpretation of preserved partial baseline diagnostic',
        raw_status=result['status'], raw_passed=result['passed'], raw_error=result.get('error'),
        raw_worker_sha256=result['worker_sha256'], raw_status_changed=False,
        selected_source_sha256=result['verified_hashes']['selected.py'],
        environment=result['environment'], tool_inventory=result['tool_inventory'],
        source_bindings=result['verified_hashes'],
        cpu_identity_status=result['cpu_file_identity']['status'],
        vendor_binary_files=result['cpu_file_identity']['binary_files'],
        mapped_flash_binaries=result['mapped_flash_binaries'],
        loaded_wrapper={key: result['loaded_fa3'][key] for key in
            ('wrapper_path', 'wrapper_sha256', 'wrapper_ast', 'native_schema', 'binding_function')},
        cases={}, observations=[], unknowns=[
            'Official judge binary identity: Modal evidence cannot establish it.',
            'Resolved PackGQA and split counts: wrapper receives None/0, no auxiliary replay was reached.',
            'Exact internal QK/PV instruction/dtype/accumulation sequence: no binary disassembly or matched internal source.',
            'Post-activity correctness and after timings: not reached.',
            'Auxiliary return replay and NCU query/capability/target sessions: not reached.',
            'DRAM throughput, tensor utilization, warp stalls and actual CTA residency: not measured.'],
        completion=dict(initial_oracle_checks=0, checked_preprofile_replays=0,
            post_activity_oracle_checks=0, after_timing_samples=0, auxiliary_replays=0,
            hardware_counter_sessions=result['hardware_counter_sessions'], activity_sessions=result['activity_sessions']))
    for name in NAMES:
        metadata = result['cases'][name]
        checks = [check for check in result['checks'] if check['case'] == name]
        oracles = [check for check in checks if check['original_oracle_called']]
        if not checks or not all(check['passed'] for check in checks):
            raise ValueError('Recorded preprofile correctness/invariance did not all pass')
        before = metadata['before_us']
        observed = attribute(trace, name)
        report['cases'][name] = dict(config=metadata['config'], case=metadata['case'],
            actual_lengths=metadata['actual_lengths'], lengths_sum=sum(metadata['actual_lengths']),
            logical_query_rows_per_kv_head=metadata['logical_packed_rows_per_kv_head'],
            logical_rows_qualification=metadata['logical_packed_rows_qualification'],
            tensor_metadata=metadata['tensor_metadata'], page_table_sha256=metadata['page_table_sha256'],
            scheduler_metadata_input=metadata['scheduler_metadata'],
            before_us=before, before_median_us=statistics.median(before),
            before_range_us=[min(before), max(before)], warmup_us=metadata['warmup_us'],
            recorded_original_oracles=oracles, recorded_invariance_checks=len(checks),
            before_timing_qualification='Three unprofiled whole selected graph samples after two warmups. Same fixture; no after confirmation, performance variant or official-score claim.',
            activity=observed)
        report['completion']['initial_oracle_checks'] += len(oracles)
        report['completion']['checked_preprofile_replays'] += len(checks)-len(oracles)
        report['completion']['after_timing_samples'] += len(metadata['after_us'])
    symbols = [report['cases'][name]['activity']['full_mainloop_symbols'] for name in NAMES]
    report['observations'] += [
        'Both Modal full-verify cases have the same complete mainloop symbol list: '+str(symbols[0] == symbols[1]),
        'Each trace contains fused preparation, scheduler preparation, mainloop, combine and norm/gate.',
        'The original parser used cross-domain CPU/GPU timestamp containment. B32 omitted its mainloop; B16 admitted unrelated restore copies.',
        'The activity trace was exported before the parser exception. Post-activity correctness was not executed, so salvaged activity is observational identity evidence only.',
        'Profiler occupancy estimate zero conflicts with a successfully executed kernel; it is not achieved hardware occupancy.',
        'NCU executable and RmProfilingAdminOnly:0 are observed. No new counter session or permission test was reached.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.directory
    result_raw = (root/'result.json').read_bytes()
    compressed = (root/'activity-trace.json.gz').read_bytes()
    raw = gzip.decompress(compressed)
    if len(raw) > MAX_TRACE:
        raise ValueError('Raw trace exceeds bound')
    result = json.loads(result_raw)
    for name, value in (('activity-trace.json', raw), ('activity-trace.json.gz', compressed)):
        expected = result['trace_artifacts'][name]
        if len(value) != expected['bytes'] or digest(value) != expected['sha256']:
            raise ValueError('Raw evidence digest mismatch: '+name)
    report = analyze(result, json.loads(raw))
    report['evidence'] = dict(result_sha256=digest(result_raw), raw_trace_sha256=digest(raw),
        gzip_trace_sha256=digest(compressed), analyzer_sha256=digest(Path(__file__).read_bytes()))
    path = root/'partial-analysis.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'path': str(path), 'sha256': digest(path.read_bytes()),
        'raw_status': report['raw_status'], 'raw_passed': report['raw_passed'],
        'before_medians_us': {name: report['cases'][name]['before_median_us'] for name in NAMES}}))


if __name__ == '__main__':
    main()
