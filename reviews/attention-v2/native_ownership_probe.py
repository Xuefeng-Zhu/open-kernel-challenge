"""Explicitly invoked H200 correctness probe; no timing or score measurement.

--list-cases imports only the standard library. Native execution needs an exact
candidate SHA and existing GPU authorization. Two different graph captures use
default private pools and independent inputs/cache/output. This never overlaps
replays of the SAME graph. Enqueue-before-wait does not prove physical overlap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
CASES = [
    dict(name='sliding_decode1', batch=1, ctx=80000, heads='tp8', layer='sliding',
         q_len=1, page=1, kv_scale=1., spread='agentic', dist='soft'),
    dict(name='sliding_decode16', batch=16, ctx=80000, heads='tp8', layer='sliding',
         q_len=1, page=1, kv_scale=1., spread='agentic', dist='soft'),
    dict(name='sliding_decode32', batch=32, ctx=80000, heads='tp8', layer='sliding',
         q_len=1, page=1, kv_scale=1., spread='agentic', dist='soft'),
    dict(name='full_decode32', batch=32, ctx=80000, heads='tp8', layer='full',
         q_len=1, page=1, kv_scale=1., spread='agentic', dist='coherent'),
    dict(name='nearby_full_page64_nonunit', batch=2, ctx=9000, heads='tp8', layer='full',
         q_len=1, page=64, kv_scale=1/64, spread='mixed', dist='soft'),
]


def selected_cases(bounded_modal=False):
    if not bounded_modal:
        return CASES
    # This explicitly reduced-context plan targets dispatch/ownership, not scored
    # timing. Full B32 still has two 8192-token chunks from the shape bound.
    reduced = [{**case, 'ctx': 8192 if case['name']=='full_decode32' else 4096}
               for case in CASES[:4]]
    ticket = dict(name='tiny_full_verify2_local_ticket', batch=2, ctx=1024,
                  heads='tp8', layer='full', q_len=8, page=64, kv_scale=1/64,
                  spread='mixed', dist='soft')
    fa3 = dict(name='tiny_full_verify1_retained_fa3', batch=1, ctx=1024,
               heads='tp8', layer='full', q_len=8, page=64, kv_scale=1.,
               spread='mixed', dist='soft')
    nearby = {**CASES[4], 'ctx': 2048}
    return [ticket, reduced[0], fa3, *reduced[1:], nearby]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(rounds, bounded_modal=False):
    cases = selected_cases(bounded_modal)
    return dict(cases=cases, rounds=rounds, graphs_per_case=2,
                labels_per_round=['A1', 'B', 'A2'], checked_replays=len(cases)*2*3*rounds,
                warmup_calls_per_graph=3, warmup_calls_checked=False,
                qualification='Scored dispatch families with varying lengths; not exact scored fixtures. '
                'Decode1 uses agentic lengths instead of the scored fixed length. No performance claim.'
                + (' Bounded plan reduces contexts and adds two L8 paths; partial coverage is possible.'
                   if bounded_modal else ''),
                same_graph_overlap=False, physical_kernel_overlap_measured=False,
                profiler_pairs_requested=len(cases) if bounded_modal else 0,
                bounded_modal=bounded_modal)


def summarize_trace(trace):
    """Compact CUPTI kernel intervals only; no counters or CPU-event timing.

    CUDA graphs may use internal streams. Distinct streams alone cannot identify
    distinct graph invocations; only distinct nonzero graph IDs establish that.
    """
    kernels, seen = [], set()
    for event in trace.get('traceEvents', []):
        if event.get('ph') != 'X' or event.get('cat') != 'kernel':
            continue
        args = event.get('args', {})
        start, duration = event.get('ts'), event.get('dur')
        stream = args.get('original_stream', args.get('stream'))
        if (not isinstance(start, (int, float)) or not isinstance(duration, (int, float))
                or not math.isfinite(start) or not math.isfinite(duration) or duration <= 0):
            continue
        graph = args.get('graph id')
        if graph in (None, 0, '0', '0x0', -1, '-1', ''):
            graph = None
        device = args.get('device', event.get('pid'))
        identity = (start, duration, str(stream), str(device), str(graph), event.get('name'))
        if identity in seen:
            continue
        seen.add(identity)
        kernels.append(dict(start=float(start), end=float(start+duration), stream=stream,
                            device=device, graph=graph, name=str(event.get('name', ''))[:72]))
    origin = min((k['start'] for k in kernels), default=0.)

    def overlap(field):
        total, maximum, intervals = 0., 0, []
        for device in {str(k['device']) for k in kernels}:
            points = {}
            for kernel in kernels:
                value = kernel[field]
                if str(kernel['device']) != device or value is None:
                    continue
                for at, delta in ((kernel['start'], 1), (kernel['end'], -1)):
                    points.setdefault(at, []).append((str(value), delta))
            active, last = {}, None
            for at in sorted(points):
                count = sum(v > 0 for v in active.values())
                if last is not None and at > last:
                    maximum = max(maximum, count)
                    if count > 1:
                        total += at-last
                        if len(intervals) < 2:
                            intervals.append([device, round(last-origin, 3), round(at-origin, 3), count])
                for value, delta in points[at]:
                    active[value] = active.get(value, 0)+delta
                last = at
        return dict(overlap_us=round(total, 3), max_concurrent=maximum,
                    first_intervals_device_start_end_count=intervals)

    streams, graphs = overlap('stream'), overlap('graph')
    return dict(kernel_count=len(kernels),
                kernels_with_graph_id=sum(k['graph'] is not None for k in kernels),
                stream_overlap=streams, graph_overlap=graphs,
                physical_kernel_overlap_observed=streams['overlap_us'] > 0,
                cross_graph_overlap_observed=graphs['overlap_us'] > 0,
                evidence=('observed_cross_graph_kernel_overlap' if graphs['overlap_us'] > 0
                          else 'unverified_cross_graph_overlap'),
                qualification='Profiler perturbs execution; GPU activity intervals are not benchmark '
                              'timings. Internal graph streams can differ within one graph. '
                              'No overlap is not proof of safety or serialization without profiling.',
                first_kernel_intervals=[dict(name=k['name'], device=k['device'], stream=k['stream'],
                                             graph=k['graph'], start_us=round(k['start']-origin,3),
                                             duration_us=round(k['end']-k['start'],3))
                                        for k in kernels[:2]],
                kernel_intervals_truncated=len(kernels)>2)


def run_case(case, runner, rounds, deadline_check, emit, profile_emit=None):
    """Orchestration separated for CPU tests; runner owns all native operations."""
    deadline_check()
    states = [runner.prepare(case, arm) for arm in range(2)]
    runner.assert_isolation(states)
    records = []
    for cycle in range(rounds):
        for label in ['A1', 'B', 'A2']:
            deadline_check()
            # Both streams receive restore+replay before ANY wait or oracle check.
            if profile_emit is not None and cycle == 0 and label == 'A1':
                profile_emit(dict(case=case['name'], **runner.profile_pair(states, label)))
            else:
                for state in states:
                    runner.restore_and_launch(state, label)
                runner.wait_all(states)
            for state in states:
                record = runner.check(state, label, cycle)
                records.append(record)
                emit(record)
                if not record['passed']:
                    raise AssertionError('Native correctness/ownership check failed')
    return records


class NativeRunner:
    def __init__(self, torch, helpers, reference, candidate, deadline_check):
        self.torch, self.h, self.ref, self.module = torch, helpers, reference, candidate
        self.deadline_check = deadline_check

    def addresses(self, data):
        return {k: v.data_ptr() for k, v in data.items() if isinstance(v, self.torch.Tensor)}

    def prepare(self, case, arm):
        self.deadline_check()
        cfg = {k: v for k, v in case.items() if k != 'name'}
        seeds = (7301 + 100*arm, 7302 + 100*arm)
        fresh = [self.ref.generate_input(**cfg, seed=s, device='cuda') for s in seeds]
        lengths = [x['seq_lens'].cpu().tolist() for x in fresh]
        if lengths[0] == lengths[1]:
            raise AssertionError('A/B actual lengths did not change; coverage not established')
        work = self.h.clone_inputs(fresh[0])
        # Existing helper warms the candidate three times before capture. Both
        # preparations finish serially before overlap, warming immutable configs.
        graph, out = self.h.capture(self.module.kernel_fn, work, fresh[0])
        self.torch.cuda.synchronize()
        return dict(arm=arm, case=case['name'], fresh=fresh, work=work, graph=graph,
                    output=out, stream=self.torch.cuda.Stream(), event=self.torch.cuda.Event(),
                    addresses=self.addresses(work), output_address=out.data_ptr(),
                    lengths=lengths, seeds=seeds, first=None)

    def assert_isolation(self, states):
        a, b = states
        if a['graph'] is b['graph']:
            raise AssertionError('Must use two independent graph objects')
        addresses_a = set(a['addresses'].values()) | {a['output_address']}
        addresses_b = set(b['addresses'].values()) | {b['output_address']}
        if addresses_a & addresses_b:
            raise AssertionError('Independent inputs/cache/output unexpectedly alias')
        # Default graph() pools are not explicitly shared by capture().
        if a['graph'].pool() == b['graph'].pool():
            raise AssertionError('Expected distinct default private graph pools')

    def restore_and_launch(self, state, label):
        version = 1 if label == 'B' else 0
        state['version'] = version
        with self.torch.cuda.stream(state['stream']):
            self.h.restore(state['work'], state['fresh'][version])
            state['graph'].replay()
            state['event'].record()

    def wait_all(self, states):
        for state in states:
            state['event'].synchronize()

    def profile_pair(self, states, label):
        """Profile exactly one checked pair after restores finish, before oracles."""
        t = self.torch
        # Keep restoration kernels/memcopies out of the activity observation.
        for state in states:
            state['version'] = 1 if label == 'B' else 0
            with t.cuda.stream(state['stream']):
                self.h.restore(state['work'], state['fresh'][state['version']])
                state['event'].record()
        self.wait_all(states)
        self.deadline_check()

        def launch_pair():
            for state in states:
                with t.cuda.stream(state['stream']):
                    state['graph'].replay()
                    state['event'].record()
            self.wait_all(states)

        if t.profiler.ProfilerActivity.CUDA not in t.profiler.supported_activities():
            launch_pair()
            return dict(evidence='unverified_profiler_cuda_activity_unavailable',
                        physical_kernel_overlap_observed=False, cross_graph_overlap_observed=False)
        # No experimental profiler options, hardware counters, stack capture or
        # tensor recording. Only the replay pair lives inside this region.
        with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                                           t.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as prof:
            launch_pair()
        self.deadline_check()
        try:
            with tempfile.TemporaryDirectory(prefix='ownership-trace-') as directory:
                trace_path = Path(directory)/'trace.json'
                prof.export_chrome_trace(str(trace_path))
                return summarize_trace(json.loads(trace_path.read_text()))
        except Exception as error:
            # The pair already completed; do not replay it to recover evidence.
            return dict(evidence='unverified_trace_export_or_parse_failed',
                        error_type=type(error).__name__, error=str(error)[:240],
                        physical_kernel_overlap_observed=False, cross_graph_overlap_observed=False)

    def check(self, state, label, cycle):
        t = self.torch
        work, out = state['work'], state['output']
        stable = self.addresses(work) == state['addresses'] and out.data_ptr() == state['output_address']
        # Guard against a nonfinite reference before tolerance comparisons.
        oracle = self.ref.ref_kernel(work)
        finite_ref = all(bool(t.isfinite(x.float()).all()) for x in oracle)
        result = self.h.check(self.ref, work, out, state['fresh'][state['version']])
        raw = self.ref.collect(work, out)
        returned = None
        if label == 'A1':
            state['first'] = raw.clone()
        elif label == 'A2':
            returned = bool(t.equal(raw, state['first']))
        passed = result['passed'] and stable and finite_ref and returned is not False
        # Complete oracle reads and A snapshots before another stream restores
        # the input/cache for the next replay. This is a correctness probe only.
        t.cuda.current_stream().synchronize()
        return dict(case=state['case'], arm=state['arm'], cycle=cycle, label=label,
                    passed=bool(passed), stable_addresses=stable, finite_reference=finite_ref,
                    A_return_output_and_newKV_equal=returned,
                    actual_seq_lens=state['lengths'][state['version']],
                    seed=state['seeds'][state['version']], **{k:v for k,v in result.items() if k!='passed'})


def save(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    compact = report.get('plan', {}).get('bounded_modal', False)
    temporary.write_text(json.dumps(report, indent=None if compact else 2,
                                    separators=(',', ':') if compact else None) + '\n')
    temporary.replace(path)


def append_record(report, record):
    """Retain every value with a single explicit schema in the compact report."""
    if report.get('plan', {}).get('bounded_modal', False):
        columns = report.setdefault('record_columns', list(record))
        if set(columns) != set(record):
            raise ValueError('Correctness record schema changed')
        report['records'].append([record[key] for key in columns])
    else:
        report['records'].append(record)


def worker(args):
    started = time.monotonic()
    report = dict(status='starting', plan=plan(args.rounds, args.bounded_modal), records=[],
                  overlap_profiles=[], timing_claim=False)
    save(args.output, report)
    def deadline_check():
        if time.monotonic() - started > args.deadline:
            raise TimeoutError('Internal wall deadline exceeded')
    try:
        candidate_path = Path(args.candidate).resolve()
        if sha256(candidate_path) != args.expected_sha:
            raise ValueError('Candidate source SHA256 mismatch')
        import torch
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Native probe requires an available H200; no fallback device')
        torch.cuda.set_device(0)
        import triton
        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(HERE/'official'))
        import modal_worker as helpers
        reference = helpers.load(HERE/'official/reference.py', 'ownership_reference')
        candidate = helpers.load(candidate_path, 'ownership_candidate')
        if sha256(candidate_path) != args.expected_sha:
            raise AssertionError('Candidate source changed during import')
        report.update(status='running', candidate_name=candidate_path.name,
                      candidate_sha256=args.expected_sha,
                      official_sha256={name:sha256(HERE/'official'/name)
                          for name in ['TASK.md','AGENTS.md','reference.py','checks.py','baseline.py']},
                      harness_sha256=sha256(__file__), helpers_sha256=sha256(HERE/'modal_worker.py'),
                      environment=dict(python=platform.python_version(), torch=torch.__version__,
                                       cuda_runtime=torch.version.cuda, triton=triton.__version__,
                                       gpu=torch.cuda.get_device_name(0),
                                       device_capability=list(torch.cuda.get_device_capability(0))))
        save(args.output, report)
        runner = NativeRunner(torch, helpers, reference, candidate, deadline_check)
        def emit(record):
            append_record(report, record)
            save(args.output, report)
        def emit_profile(profile):
            report['overlap_profiles'].append(profile)
            save(args.output, report)
        for case in selected_cases(args.bounded_modal):
            deadline_check()
            run_case(case, runner, args.rounds, deadline_check, emit,
                     emit_profile if args.bounded_modal else None)
            if sha256(candidate_path) != args.expected_sha:
                raise AssertionError('Candidate source changed during probe')
        torch.cuda.synchronize()
        report['status'] = 'passed'
        report['correctness_complete'] = True
        report['cross_graph_overlap_coverage_complete'] = bool(args.bounded_modal and all(
            p.get('cross_graph_overlap_observed', False) for p in report['overlap_profiles']))
        if args.bounded_modal and not report['cross_graph_overlap_coverage_complete']:
            report['status'] = 'correctness_passed_overlap_unverified'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        report['worker_wall_seconds'] = time.monotonic() - started
        save(args.output, report)
    return 0 if report['status'] in ('passed', 'correctness_passed_overlap_unverified') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--expected-sha')
    parser.add_argument('--output')
    parser.add_argument('--rounds', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--deadline', type=int)
    parser.add_argument('--bounded-modal', action='store_true',
                        help='Seven reduced-context cases and a compact GPU kernel overlap trace summary')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.deadline is None:
        args.deadline = 75 if args.bounded_modal else 180
    if args.list_cases:
        print(json.dumps(plan(args.rounds, args.bounded_modal), indent=2))
        return 0
    if not args.candidate or not args.output or not args.expected_sha:
        parser.error('Native run requires --candidate, --expected-sha and --output')
    if len(args.expected_sha)!=64 or any(c not in '0123456789abcdef' for c in args.expected_sha):
        parser.error('--expected-sha must be a lowercase SHA256')
    if not 30 <= args.deadline <= 600:
        parser.error('--deadline must be 30..600 seconds')
    if args.bounded_modal and (args.deadline > 80 or args.rounds != 1):
        parser.error('--bounded-modal requires one round and an internal deadline of at most 80 seconds')
    if args.worker:
        return worker(args)
    if Path(args.output).exists():
        parser.error('--output already exists; use a new report path to preserve evidence')
    # Hard outer watchdog also bounds blocking native synchronization/compilation.
    # Native runs only in this child; process exit releases its CUDA resources.
    command = [sys.executable, __file__, '--worker', '--candidate', args.candidate,
               '--expected-sha', args.expected_sha, '--output', args.output,
               '--rounds', str(args.rounds), '--deadline', str(args.deadline)]
    if args.bounded_modal:
        command.append('--bounded-modal')
    try:
        result = subprocess.run(command, timeout=args.deadline+10, check=False)
        return result.returncode
    except subprocess.TimeoutExpired:
        report = json.loads(Path(args.output).read_text()) if Path(args.output).exists() else {}
        report.update(status='timeout', error='Outer watchdog terminated native worker; partial coverage only')
        save(args.output, report)
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
