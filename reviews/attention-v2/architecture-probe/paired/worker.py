"""One bounded BF16-custom versus unchanged-FA3 correctness/replay diagnostic.

No Modal, credentials, network or official submission. Whole-call graph timing
starts only after all three cases pass unchanged official checks in A→B→A.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import signal
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
SUPPORT = {'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055'}
NAMES = ['fa3', 'bf16_custom']
TARGETS = [dict(name=f'full_verify{b}_scored_family', batch=b, q_len=8, ctx=80000,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent') for b in [16, 32]]
ADVERSARIAL = dict(name='full_verify16_page64_boundary', batch=16, q_len=8,
    ctx=8193, heads='tp8', layer='full', page=64, kv_scale=1/64, spread='fixed', dist='soft')
WARMUP_PAIRS = 4  # Four two-arm samples = two balanced reverse-order pairs.
MEASURED_PAIRS = 12  # Twelve two-arm samples = six balanced reverse-order pairs.


def boundary_lengths(version):
    values = [0, 1, 63, 64, 65, 8191, 8192, 8193]
    return (values if version == 0 else list(reversed(values)))*2


def order_and_version(index):
    return (NAMES if index % 2 == 0 else list(reversed(NAMES))), (index//2) % 2


def execute_aba(runners, gate, emit):
    for label in ['A1', 'B', 'A2']:
        for name in NAMES:
            gate()
            result = runners[name].replay_and_check(label)
            emit(dict(runner=name, label=label, **result))
            if not result['passed']:
                raise AssertionError('Official correctness/replay gate failed; timing forbidden')


def summarize_timing(samples):
    if len(samples) != MEASURED_PAIRS:
        raise ValueError('Incomplete timing cannot be summarized as pass')
    if not all(math.isfinite(x['raw_us'][n]) and x['raw_us'][n] > 0 for x in samples for n in NAMES):
        raise ValueError('Nonfinite or nonpositive timing')
    ratios = [x['raw_us']['fa3']/x['raw_us']['bf16_custom'] for x in samples]
    balanced = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, len(ratios), 2)]
    return dict(raw_median_us={n: statistics.median(x['raw_us'][n] for x in samples) for n in NAMES},
        raw_min_max_us={n: [min(x['raw_us'][n] for x in samples), max(x['raw_us'][n] for x in samples)] for n in NAMES},
        fa3_over_bf16_custom_ratios=ratios, reverse_order_pair_geomeans=balanced,
        median_reverse_order_pair_ratio=statistics.median(balanced),
        qualification='Six paired ratios per scored family, two fresh fixtures, no confidence interval or official score')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bounded_kernel_names(names):
    kept, total_bytes, omitted = [], 0, 0
    for name in sorted(set(names)):
        size = len(name.encode('utf-8'))
        if len(kept) < 64 and total_bytes+size <= 16384:
            kept.append(name)
            total_bytes += size
        else:
            omitted += 1
    return dict(kernel_names=kept, kernel_names_utf8_bytes=total_bytes,
        omitted_kernel_name_count=omitted, kernel_names_truncated=False)


def main(args):
    start = time.monotonic()
    report = dict(kind='M64 BF16 custom versus unchanged FA3 whole-call diagnostic; not official score',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        correctness_complete=False, timing_complete=False, cases=[], phases=[], timings=[],
        source_sha256=SOURCE_SHA256, support_sha256=SUPPORT,
        protocol=dict(required_cases=[ADVERSARIAL, *TARGETS], graph_captures_per_case=2,
            replay_labels=['A1', 'B', 'A2'], official_oracle_calls_per_runner_per_case=3,
            warmup_two_arm_samples=WARMUP_PAIRS, measured_two_arm_samples=MEASURED_PAIRS,
            reverse_order_measured_pairs=MEASURED_PAIRS//2, input_versions=2,
            timed_region='Whole kernel_fn CUDA graph replay including per-call prep, cache stores, partials, ticket-zeroing and integrated merge',
            timed_region_excludes='Input/cache restores, eviction, validation and calibration',
            timing_replay_checks='Exact output/newKV against same-source same-input official-oracle-checked bytes, plus immutable inputs and untouched cache rows',
            eviction_bytes=256*1024*1024, concurrency_tested=False, all_official_shapes_tested=False,
            tolerances='Unmodified official/reference.py and official/checks.py', formal_dispatch_modified=False))

    def save():
        report['worker_wall_seconds'] = time.monotonic()-start
        raw = json.dumps(report, separators=(',', ':'))+'\n'
        if len(raw.encode()) > 200000:
            raise RuntimeError('Bounded report exceeded 200KB')
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(raw)
        tmp.replace(args.output)

    def gate():
        save()
        if time.monotonic()-start >= args.deadline:
            raise TimeoutError('Paired worker deadline')

    def alarm(*_):
        raise TimeoutError('Paired worker alarm')

    def phase(name, before):
        report['phases'].append(dict(name=name, wall_seconds=time.monotonic()-before))
        gate()

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(args.deadline))
    save()
    try:
        assert not __import__('os').environ.get('TRITON_INTERPRET')
        for name, sha in [('candidate.py', SOURCE_SHA256), *SUPPORT.items()]:
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != sha:
                raise ValueError('Frozen support/source mismatch: '+name)
        report['payload_sha256'] = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ['worker.py', 'candidatecustomhelper.py', 'candidate.py', *SUPPORT]}
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name():
            raise RuntimeError('Native H200 required')
        report['environment'] = dict(gpu=t.cuda.get_device_name(), capability=t.cuda.get_device_capability(),
            torch=t.__version__, cuda=t.version.cuda, triton=triton.__version__, python=platform.python_version())
        sys.path.insert(0, str(HERE/'official'))
        sys.path.insert(0, str(HERE))
        h = load(HERE/'modal_worker.py', 'bf16_pair_helpers')
        ref = load(HERE/'official/reference.py', 'bf16_pair_reference')
        candidate = load(HERE/'candidate.py', 'candidate')
        custom = load(HERE/'candidatecustomhelper.py', 'candidatecustomhelper')
        functions = dict(fa3=candidate.kernel_fn, bf16_custom=custom.bf16_custom_fn)
        phase('imports_and_hashes', start)

        def fresh(case, version):
            data = ref.generate_input(**{k: v for k, v in case.items() if k != 'name'},
                seed=135701+version, device='cuda')
            if case['name'] == ADVERSARIAL['name']:
                lengths = t.tensor(boundary_lengths(version), device='cuda', dtype=t.int32)
                flat = data['page_table'].reshape(-1)
                target_b = boundary_lengths(version).index(0)
                destination = target_b*data['page_table'].shape[1]
                original_zero = int((flat == 0).nonzero()[0])
                old = flat[destination].clone()
                flat[destination] = 0
                flat[original_zero] = old
                data['seq_lens'].copy_(lengths)
                pos = lengths.to(t.int64)[:, None]+t.arange(8, device='cuda')
                data['positions'].copy_(pos.reshape(-1))
                data['cache_seqlens'].copy_(lengths+8)
                data['cache_loc'].copy_(ref.new_slots(data).reshape(-1))
                for b, seq in enumerate(boundary_lengths(version)):
                    valid = (seq+8+63)//64
                    data['page_table'][b, valid:] = -1
                assert data['scheduler_metadata'] is None
                assert int(data['cache_loc'][target_b*8]) == 0
                assert t.equal(data['cache_loc'], ref.new_slots(data).reshape(-1))
            return data

        class Runner:
            def __init__(self, name, inputs):
                self.name, self.inputs = name, inputs
                self.work = h.clone_inputs(inputs[0])
                self.graph, self.out = h.capture(functions[name], self.work, inputs[0])
                self.addresses = {k: v.data_ptr() for k, v in self.work.items() if isinstance(v, t.Tensor)}
                self.out_address = self.out.data_ptr()
                self.expected = {}

            def checked(self, version, oracle):
                raw = ref.collect(self.work, self.out)
                stable = self.addresses == {k: v.data_ptr() for k, v in self.work.items() if isinstance(v, t.Tensor)}
                stable = stable and self.out.data_ptr() == self.out_address
                intact, why = h.unchanged(self.work, self.inputs[version])
                exact = bool(t.equal(raw, self.expected[version])) if version in self.expected else None
                if oracle:
                    okay, message = ref.check_implementation(self.work, raw)
                    self.expected[version] = raw.clone()
                    okay = okay and exact is not False
                else:
                    okay, message = exact is True, 'Exact bytes match same-input official-oracle-checked result'
                return dict(passed=bool(okay and stable and intact), stable_addresses=stable,
                    immutable_inputs_and_untouched_rows=intact, immutable_message=why,
                    official_oracle_called=oracle, oracle_message=message, exact_output_newkv_return=exact)

            def replay_and_check(self, label):
                version = 1 if label == 'B' else 0
                h.restore(self.work, self.inputs[version])
                self.graph.replay()
                t.cuda.synchronize()
                result = self.checked(version, True)
                result['actual_lengths'] = self.inputs[version]['seq_lens'].cpu().tolist()
                return result

            def timed(self, version, eviction):
                h.restore(self.work, self.inputs[version])
                value = h.timed(self.graph, eviction)
                return value, self.checked(version, False)

        retained = {}
        for case in [ADVERSARIAL, *TARGETS]:
            before = time.monotonic()
            gate()
            inputs = [fresh(case, v) for v in range(2)]
            assert not t.equal(inputs[0]['seq_lens'], inputs[1]['seq_lens']), 'A/B lengths must change'
            item = dict(case=case, seeds=[135701, 135702], records=[],
                input_lengths=[x['seq_lens'].cpu().tolist() for x in inputs],
                page_table_shape=list(inputs[0]['page_table'].shape))
            report['cases'].append(item)
            runners = {n: Runner(n, inputs) for n in NAMES}
            execute_aba(runners, gate, lambda row: (item['records'].append(row), save()))
            item['passed'] = True
            phase(case['name']+'_capture_aba', before)
            if case is not ADVERSARIAL:
                retained[case['name']] = runners
            else:
                del runners, inputs
        report['correctness_complete'] = True
        gate()
        eviction = t.ones(256*1024*1024//4, device='cuda')
        empty = t.cuda.CUDAGraph()
        with t.cuda.graph(empty):
            pass
        for _ in range(2):
            empty.replay()
        for case in TARGETS:
            before = time.monotonic()
            gate()
            samples, warmups = [], []
            timing = dict(case=case, samples=samples, warmups=warmups, calibration_us=[])
            report['timings'].append(timing)
            timing['calibration_us'] = [h.timed(empty, eviction) for _ in range(3)]
            for i in range(WARMUP_PAIRS+MEASURED_PAIRS):
                gate()
                order, version = order_and_version(i)
                row = dict(index=i, order=order, input_version=version, raw_us={}, checks={})
                (warmups if i < WARMUP_PAIRS else samples).append(row)
                for name in order:
                    gate()
                    value, check = retained[case['name']][name].timed(version, eviction)
                    row['raw_us'][name], row['checks'][name] = value, check
                    save()
                    if not check['passed']:
                        raise AssertionError('Timed replay failed exact bytes or immutable-input check')
            timing['calibration_us'] += [h.timed(empty, eviction) for _ in range(3)]
            offset = statistics.median(timing['calibration_us'])
            timing['empty_graph_median_us'] = offset
            for row in warmups+samples:
                row['calibrated_us'] = {n: value-offset for n, value in row['raw_us'].items()}
            timing['calibration_resolves_both'] = all(v > 0 for row in samples for v in row['calibrated_us'].values())
            timing['summary'] = summarize_timing(samples)
            timing['calibration_qualification'] = 'Separate empty-graph subtraction without clipping; raw timings primary; not official calibration'
            phase(case['name']+'_paired_timing', before)
        report['timing_complete'] = True
        report['fa3_profiler'] = dict(attempted=False, actual_scheduler_verified=False)
        if args.deadline-(time.monotonic()-start) > 15:
            try:
                before = time.monotonic()
                runner = retained[TARGETS[1]['name']]['fa3']
                h.restore(runner.work, runner.inputs[0])
                with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CUDA]) as profile:
                    runner.graph.replay()
                    t.cuda.synchronize()
                names = sorted({event.name for event in profile.events() if 'cuda' in str(event.device_type).lower()})
                report['fa3_profiler'] = dict(attempted=True, **bounded_kernel_names(names),
                    actual_scheduler_verified=False, wall_seconds=time.monotonic()-before,
                    qualification='One unchanged FA3 B32/L8/page1 graph outside timing; names require template/source matching before scheduler claims')
            except Exception as exc:
                report['fa3_profiler'] = dict(attempted=True, error_type=type(exc).__name__, actual_scheduler_verified=False)
        report.update(status='complete', passed=True)
        save()
        return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed', error_type=type(exc).__name__, error=str(exc)[:800])
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=110)
    args = parser.parse_args()
    assert 1 <= args.deadline <= 110
    raise SystemExit(main(args))
