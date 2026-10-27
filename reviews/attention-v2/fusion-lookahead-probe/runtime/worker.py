"""Source-bound fusion plus A2-15 composition diagnostic; local preparation until separately launched.

No Modal, network, credentials, profiler, official check, loaded-module mutation,
or FA3 argument override. All complete-call work stays inside each CUDA graph.
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
PROBE = HERE.parent
V2 = HERE.parents[1]
SOURCE_HASHES = {
    'baseline.py': '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0',
    'candidate.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
}
SUPPORT_HASHES = {
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
NAMES = ['baseline', 'candidate']
SEEDS = [135711, 135712]
TARGET_PAIRS = [(32, 1)]
TARGET = dict(name='full_B32_L1_scored', batch=32, q_len=1, ctx=80000,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic',
    dist='coherent', boundary=None)
TARGET_BOUNDARY = dict(name='full_B32_L1_boundary', batch=32, q_len=1, ctx=8193,
    heads='tp8', layer='full', page=64, kv_scale=1/64, spread='fixed',
    dist='soft', boundary='target')
FALLBACK_BOUNDARY = dict(name='full_B4_L1_fallback_unchanged', batch=4, q_len=1, ctx=8193,
    heads='tp8', layer='full', page=64, kv_scale=1/64, spread='fixed',
    dist='soft', boundary='fallback')
DP_SMOKE = dict(name='full_B32_L1_dp_smoke', batch=32, q_len=1, ctx=65,
    heads='dp', layer='full', page=64, kv_scale=1/64, spread='fixed',
    dist='soft', boundary='dp')
CONTROL = dict(name='full_B16_L8_fusion_unchanged', batch=16, q_len=8, ctx=80000,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic',
    dist='coherent', boundary=None)
CASES = [TARGET_BOUNDARY, FALLBACK_BOUNDARY, DP_SMOKE, TARGET, CONTROL]
TIMED_CASES = [TARGET, CONTROL]
WARMUP_SAMPLES = 4
MEASURED_SAMPLES = 12


def boundary_lengths(batch, version, family='target'):
    if batch == 32 and family == 'dp':
        values = [0, 63, 64, 65] * 8
    elif batch == 32 and family == 'target':
        values = [0, 1, 63, 64, 65, 8191, 8192, 8193] * 4
    elif batch == 4 and family == 'fallback':
        values = [0, 63, 64, 8193]
    else:
        raise ValueError('Only the three declared boundary families are supported')
    return values if version == 0 else list(reversed(values))


def order_and_version(index):
    return (NAMES if index % 2 == 0 else list(reversed(NAMES))), (index // 2) % 2


def source_gate(probe=PROBE, support=V2):
    observed = {}
    for root, hashes in [(probe, SOURCE_HASHES), (support, SUPPORT_HASHES)]:
        for name, expected in hashes.items():
            got = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if got != expected:
                raise ValueError('Frozen source/support mismatch: ' + name)
            observed[name] = got
    return observed


def timing_summary(samples):
    if len(samples) != MEASURED_SAMPLES:
        raise ValueError('Incomplete measured sample set')
    values = [row['raw_us'][name] for row in samples for name in NAMES]
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('Nonfinite or nonpositive whole-call timing')
    ratios = [row['raw_us']['baseline'] / row['raw_us']['candidate'] for row in samples]
    balanced = [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios), 2)]
    return dict(raw_median_us={name: statistics.median(row['raw_us'][name] for row in samples)
                              for name in NAMES},
        raw_range_us={name: [min(row['raw_us'][name] for row in samples),
                            max(row['raw_us'][name] for row in samples)] for name in NAMES},
        baseline_over_candidate_ratios=ratios,
        reverse_order_pair_geomeans=balanced,
        median_reverse_order_pair_ratio=statistics.median(balanced),
        positive_reverse_order_pairs=sum(value > 1 for value in balanced),
        median_paired_latency_reduction_percent=100*(1-1/statistics.median(balanced)),
        qualification='Six reverse-order pairs, two input versions, one session; no confidence interval or official score')


def benefit_gate(timings):
    expected = {case['name'] for case in TIMED_CASES}
    if len(timings) != 2 or {row['case']['name'] for row in timings} != expected:
        raise ValueError('Complete target and unchanged-control timing families required')
    summaries = {row['case']['name']: timing_summary(row['samples']) for row in timings}
    target = summaries[TARGET['name']]
    reduction = target['median_paired_latency_reduction_percent']
    target_ok = reduction >= 4. and target['positive_reverse_order_pairs'] == 6
    control = summaries[CONTROL['name']]
    slowdown = 100*(1/control['median_reverse_order_pair_ratio']-1)
    control_ok = slowdown <= 5.
    return dict(admitted=target_ok and control_ok,
        target=dict(case=TARGET['name'], median_paired_latency_reduction_percent=reduction,
            positive_reverse_order_pairs=target['positive_reverse_order_pairs'], passed=target_ok),
        unchanged_control=dict(case=CONTROL['name'], paired_candidate_slowdown_percent=slowdown,
            passed=control_ok),
        new_question='Replication and integration of a previously measured component with validated fusion',
        predeclared_gate='Target median balanced latency reduction >=4%, all six balanced ratios positive; unchanged control slowdown <=5%',
        historical_A2_15_goal='The original 10% goal was not met; this new replication gate does not retcon that result',
        qualification='One session only; passing permits parent consideration of one complete official check, not automatic submission or promotion')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(args):
    start = time.monotonic()
    report = dict(kind='Validated preparation fusion plus A2-15 pipeline composition, not official score',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        source_sha256=SOURCE_HASHES, support_sha256=SUPPORT_HASHES,
        cases=[], concurrent_submissions=[], timings=[], phases=[], benefit=None,
        correctness_complete=False, timing_complete=False, official_check_admitted=False,
        protocol=dict(target_pairs=TARGET_PAIRS, seeds=SEEDS,
            replay_labels=['A1', 'B', 'A2'], cases=CASES,
            cache_gate='Entire post-store K/V bytes exact across arms after each ABA label; each attention output uses original oracle and its own exact return-A check',
            warmup_two_arm_samples=WARMUP_SAMPLES, measured_two_arm_samples=MEASURED_SAMPLES,
            eviction_bytes=256 * 1024 * 1024,
            timed_region='Entire kernel_fn CUDA graph device work: preparation, cache stores, FA3 scheduling/compute/combine and output gate; graph memory reused on replay, host allocation excluded',
            excludes='Fixture/scheduler-metadata creation, cache/input restore, L2 eviction, validation, empty-graph diagnostic',
            tolerances='Unmodified official/reference.py and official/checks.py',
            all_official_shapes_tested=False, profiler_used=False,
            correctness_only=args.correctness_only,
            concurrency='Two independent candidate target-boundary graph captures with private default allocation pools and disjoint writable caches, submitted on separate streams; no claim of actual hardware overlap',
            performance_gate='NEW target composition-replication cut >=4% and 6/6 positive balanced pairs; unchanged fusion control slowdown <=5%; old A2-15 10% goal remains unmet'))

    def save():
        report['worker_wall_seconds'] = time.monotonic() - start
        data = json.dumps(report, separators=(',', ':')) + '\n'
        if len(data.encode()) > 400000:
            raise RuntimeError('Result exceeds bounded 400KB transport')
        temp = args.output.with_suffix('.tmp')
        temp.write_text(data)
        temp.replace(args.output)

    def gate():
        save()
        if time.monotonic() - start >= args.deadline:
            raise TimeoutError('Composition worker deadline')

    def alarm(*_):
        raise TimeoutError('Composition worker alarm')

    def phase(name, before):
        report['phases'].append(dict(name=name, wall_seconds=time.monotonic() - before))
        gate()

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(args.deadline))
    save()
    try:
        if __import__('os').environ.get('TRITON_INTERPRET'):
            raise RuntimeError('Native GPU execution required')
        report['verified_hashes'] = source_gate()
        report['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name():
            raise RuntimeError('Native H200 required')
        report['environment'] = dict(gpu=t.cuda.get_device_name(),
            capability=t.cuda.get_device_capability(), torch=t.__version__,
            cuda=t.version.cuda, triton=triton.__version__, python=platform.python_version())
        sys.path.insert(0, str(V2 / 'official'))
        helpers = load(V2 / 'modal_worker.py', 'composition_runtime_helpers')
        ref = load(V2 / 'official/reference.py', 'composition_runtime_reference')
        baseline = load(PROBE / 'baseline.py', 'composition_runtime_baseline')
        candidate = load(PROBE / 'candidate.py', 'composition_runtime_candidate')
        functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)
        phase('imports_and_hashes', start)

        def fresh(case, version):
            values = {key: val for key, val in case.items() if key not in {'name', 'boundary'}}
            data = ref.generate_input(**values, seed=SEEDS[version], device='cuda')
            if case['boundary']:
                batch, length = case['batch'], case['q_len']
                values = boundary_lengths(batch, version, case['boundary'])
                lengths = t.tensor(values, device='cuda', dtype=t.int32)
                target_b = values.index(0)
                table = data['page_table']
                flat = table.reshape(-1)
                destination = target_b * table.shape[1]
                original_zero = int((flat == 0).nonzero()[0])
                previous = flat[destination].clone()
                flat[destination] = 0
                flat[original_zero] = previous
                data['seq_lens'].copy_(lengths)
                positions = lengths.long()[:, None] + t.arange(length, device='cuda')
                data['positions'].copy_(positions.reshape(-1))
                data['cache_seqlens'].copy_(lengths + length)
                data['cache_loc'].copy_(ref.new_slots(data).reshape(-1))
                for b, seq in enumerate(values):
                    table[b, (seq + length + 63) // 64:] = -1
                # Rebuild the serving-supplied decode metadata after changing live
                # lengths. This occurs in fixture creation, outside timed calls.
                data['scheduler_metadata'] = ref._fa3_scheduler_metadata(
                    data['cache_seqlens'], data['cu_seqlens_q'],
                    data['config']['num_q_heads'], data['config']['num_kv_heads'],
                    case['page'], length, case['layer'])
                assert int(data['cache_loc'][target_b * length]) == 0
                assert t.equal(data['cache_loc'], ref.new_slots(data).reshape(-1))
                assert bool((table == -1).any())
            if case['q_len'] == 1 and data['scheduler_metadata'] is None:
                raise AssertionError('Full decode must exercise actual supplied FA3 scheduler metadata')
            if case['q_len'] == 8 and data['scheduler_metadata'] is not None:
                raise AssertionError('Verify scheduler metadata must remain None')
            return data

        def byte_row(name, left, right):
            a = left.contiguous().reshape(-1).view(t.uint8)
            b = right.contiguous().reshape(-1).view(t.uint8)
            return dict(name=name, bytes=a.numel(), exact=bool(t.equal(a, b)),
                        unequal_byte_count=int((a != b).sum()))

        class Runner:
            def __init__(self, name, inputs):
                self.name, self.inputs = name, inputs
                self.work = helpers.clone_inputs(inputs[0])
                self.graph, self.out = helpers.capture(functions[name], self.work, inputs[0])
                self.addresses = {key: val.data_ptr() for key, val in self.work.items() if isinstance(val, t.Tensor)}
                self.out_address = self.out.data_ptr()
                self.expected = {}

            def check(self, version, oracle):
                raw = ref.collect(self.work, self.out)
                addresses = {key: val.data_ptr() for key, val in self.work.items() if isinstance(val, t.Tensor)}
                stable = addresses == self.addresses and self.out.data_ptr() == self.out_address
                intact, why = helpers.unchanged(self.work, self.inputs[version])
                exact = bool(t.equal(raw, self.expected[version])) if version in self.expected else None
                if oracle:
                    okay, message = ref.check_implementation(self.work, raw)
                    self.expected[version] = raw.clone()
                    okay = okay and exact is not False
                else:
                    okay, message = exact is True, 'Exact bytes match same-input official-oracle-checked bytes'
                return dict(passed=bool(okay and stable and intact), stable_addresses=stable,
                    immutable_inputs_and_untouched_rows=intact, immutable_message=why,
                    official_oracle_called=oracle, oracle_message=message,
                    exact_output_newkv_return=exact)

            def replay(self, version, oracle=True):
                helpers.restore(self.work, self.inputs[version])
                self.graph.replay()
                t.cuda.synchronize()
                return self.check(version, oracle)

            def timed(self, version, eviction):
                helpers.restore(self.work, self.inputs[version])
                value = helpers.timed(self.graph, eviction)
                return value, self.check(version, False)

        retained = {}
        # Complete all original oracle/replay/cache gates before any timing.
        for case in CASES:
            before = time.monotonic()
            gate()
            inputs = [fresh(case, version) for version in [0, 1]]
            assert not t.equal(inputs[0]['seq_lens'], inputs[1]['seq_lens'])
            assert not t.equal(inputs[0]['qkv'], inputs[1]['qkv'])
            assert not t.equal(inputs[0]['page_table'], inputs[1]['page_table'])
            item = dict(case=case, input_lengths=[x['seq_lens'].cpu().tolist() for x in inputs],
                metadata_shapes=[None if x['scheduler_metadata'] is None else list(x['scheduler_metadata'].shape) for x in inputs],
                records=[], cache_cross_arm=[])
            report['cases'].append(item)
            runners = {name: Runner(name, inputs) for name in NAMES}
            for label, version in [('A1', 0), ('B', 1), ('A2', 0)]:
                for name in NAMES:
                    gate()
                    row = dict(runner=name, label=label, input_version=version,
                               **runners[name].replay(version))
                    item['records'].append(row)
                    gate()
                    if not row['passed']:
                        raise AssertionError('Official replay/cache gate failed; timing forbidden')
                cache_rows = [byte_row(key, runners['baseline'].work[key], runners['candidate'].work[key])
                              for key in ['k_cache', 'v_cache']]
                item['cache_cross_arm'].append(dict(label=label, input_version=version,
                    comparisons=cache_rows, passed=all(row['exact'] for row in cache_rows)))
                if not item['cache_cross_arm'][-1]['passed']:
                    raise AssertionError('Cross-arm K/V bytes differ; timing forbidden')
            if case['boundary'] == 'target':
                second = Runner('candidate', inputs)
                first = runners['candidate']
                for key in ['k_cache', 'v_cache']:
                    if first.work[key].data_ptr() == second.work[key].data_ptr():
                        raise AssertionError('Concurrent captures must own disjoint writable caches')
                helpers.restore(first.work, inputs[0])
                helpers.restore(second.work, inputs[1])
                streams = [t.cuda.Stream(), t.cuda.Stream()]
                ready = t.cuda.Event()
                ready.record()
                for stream, runner in zip(streams, [first, second]):
                    stream.wait_event(ready)
                    with t.cuda.stream(stream):
                        runner.graph.replay()
                for stream in streams:
                    t.cuda.current_stream().wait_stream(stream)
                t.cuda.synchronize()
                checks = [first.check(0, True), second.check(1, True)]
                concurrent = dict(case=case['name'], disjoint_writable_caches=True,
                    separate_graph_captures=True, separate_streams=True,
                    actual_hardware_overlap_measured=False, checks=checks,
                    passed=all(row['passed'] for row in checks))
                report['concurrent_submissions'].append(concurrent)
                if not concurrent['passed']:
                    raise AssertionError('Independent graph submission gate failed; timing forbidden')
                del second, first, streams, runners, inputs
            elif case in TIMED_CASES:
                retained[case['name']] = runners
            else:
                del runners, inputs
            item['passed'] = True
            phase(case['name'] + '_oracle_replay', before)
        report['correctness_complete'] = True
        gate()
        if args.correctness_only:
            report.update(status='correctness_complete', passed=True,
                timing_omitted='Explicit correctness-only mode; no performance claim')
            save()
            return 0
        eviction = t.ones(256 * 1024 * 1024 // 4, device='cuda')
        empty = t.cuda.CUDAGraph()
        with t.cuda.graph(empty):
            pass
        for _ in range(2):
            empty.replay()
        for case in TIMED_CASES:
            before = time.monotonic()
            gate()
            row = dict(case=case, warmups=[], samples=[], calibration_us=[])
            report['timings'].append(row)
            row['calibration_us'] = [helpers.timed(empty, eviction) for _ in range(3)]
            for index in range(WARMUP_SAMPLES + MEASURED_SAMPLES):
                order, version = order_and_version(index)
                sample = dict(index=index, order=order, input_version=version, raw_us={}, checks={})
                (row['warmups'] if index < WARMUP_SAMPLES else row['samples']).append(sample)
                for name in order:
                    gate()
                    value, check = retained[case['name']][name].timed(version, eviction)
                    sample['raw_us'][name], sample['checks'][name] = value, check
                    gate()
                    if not check['passed']:
                        raise AssertionError('Measured replay failed exact output/newKV or immutability gate')
            row['calibration_us'] += [helpers.timed(empty, eviction) for _ in range(3)]
            offset = statistics.median(row['calibration_us'])
            row['empty_graph_median_us'] = offset
            for sample in row['warmups'] + row['samples']:
                sample['calibrated_us'] = {name: value - offset for name, value in sample['raw_us'].items()}
            row['calibration_resolves_both'] = all(value > 0 for sample in row['samples'] for value in sample['calibrated_us'].values())
            row['summary'] = timing_summary(row['samples'])
            row['calibration_qualification'] = 'Separate empty graph subtraction without clipping; raw timings primary, not official judge calibration'
            phase(case['name'] + '_timing', before)
        report['benefit'] = benefit_gate(report['timings'])
        report.update(status='complete', passed=True, timing_complete=True,
            official_check_admitted=report['benefit']['admitted'])
        save()
        return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=160.)
    parser.add_argument('--correctness-only', action='store_true')
    raise SystemExit(main(parser.parse_args()))
