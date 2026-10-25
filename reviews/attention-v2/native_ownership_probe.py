"""Explicitly invoked H200 correctness probe; no timing or score measurement.

--list-cases imports only the standard library. Native execution needs an exact
candidate SHA and existing GPU authorization. Two different graph captures use
default private pools and independent inputs/cache/output. This never overlaps
replays of the SAME graph. Enqueue-before-wait does not prove physical overlap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(rounds):
    return dict(cases=CASES, rounds=rounds, graphs_per_case=2,
                labels_per_round=['A1', 'B', 'A2'], checked_replays=5*2*3*rounds,
                warmup_calls_per_graph=3, warmup_calls_checked=False,
                qualification='Scored dispatch families with varying lengths; not exact scored fixtures. '
                'Decode1 uses agentic lengths instead of the scored fixed length. No performance claim.',
                same_graph_overlap=False, physical_kernel_overlap_measured=False)


def run_case(case, runner, rounds, deadline_check, emit):
    """Orchestration separated for CPU tests; runner owns all native operations."""
    deadline_check()
    states = [runner.prepare(case, arm) for arm in range(2)]
    runner.assert_isolation(states)
    records = []
    for cycle in range(rounds):
        for label in ['A1', 'B', 'A2']:
            deadline_check()
            # Both streams receive restore+replay before ANY wait or oracle check.
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
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def worker(args):
    started = time.monotonic()
    report = dict(status='starting', plan=plan(args.rounds), records=[], timing_claim=False)
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
            report['records'].append(record)
            save(args.output, report)
        for case in CASES:
            deadline_check()
            run_case(case, runner, args.rounds, deadline_check, emit)
            if sha256(candidate_path) != args.expected_sha:
                raise AssertionError('Candidate source changed during probe')
        torch.cuda.synchronize()
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        report['worker_wall_seconds'] = time.monotonic() - started
        save(args.output, report)
    return 0 if report['status']=='passed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--expected-sha')
    parser.add_argument('--output')
    parser.add_argument('--rounds', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--deadline', type=int, default=180)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.list_cases:
        print(json.dumps(plan(args.rounds), indent=2))
        return 0
    if not args.candidate or not args.output or not args.expected_sha:
        parser.error('Native run requires --candidate, --expected-sha and --output')
    if len(args.expected_sha)!=64 or any(c not in '0123456789abcdef' for c in args.expected_sha):
        parser.error('--expected-sha must be a lowercase SHA256')
    if not 30 <= args.deadline <= 600:
        parser.error('--deadline must be 30..600 seconds')
    if args.worker:
        return worker(args)
    if Path(args.output).exists():
        parser.error('--output already exists; use a new report path to preserve evidence')
    # Hard outer watchdog also bounds blocking native synchronization/compilation.
    # Native runs only in this child; process exit releases its CUDA resources.
    command = [sys.executable, __file__, '--worker', '--candidate', args.candidate,
               '--expected-sha', args.expected_sha, '--output', args.output,
               '--rounds', str(args.rounds), '--deadline', str(args.deadline)]
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
