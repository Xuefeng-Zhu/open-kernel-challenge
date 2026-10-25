"""Focused A2-13/A2-15 H200 execution diagnostic, not an official score.

This file contains no Modal client or credentials. It is executed only by the
separately reviewed bounded controller. Local imports/tests need no torch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCES = {
    'a213': ('mixed_fa3_fp8_qk.py', 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'),
    'a215': ('mixed_fa3_page_lookahead.py', 'fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c'),
}
SUPPORT = {
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
TARGET = dict(name='full_decode32_scored_family', batch=32, q_len=1, ctx=80000,
              heads='tp8', layer='full', page=1, kv_scale=1.,
              spread='agentic', dist='coherent')
ADVERSARIAL = dict(name='full_decode32_page64_boundary', batch=32, q_len=1,
                   ctx=8193, heads='tp8', layer='full', page=64, kv_scale=1/64,
                   spread='fixed', dist='soft')
WARMUP_PAIRS = 4
MEASURED_PAIRS = 12


def boundary_lengths(version):
    values = [0, 1, 63, 64, 65, 8191, 8192, 8193]
    return (values if version == 0 else list(reversed(values))) * 4


def order_and_version(index):
    # Each adjacent reverse-order pair uses the same independently checked input.
    return (['a213', 'a215'] if index % 2 == 0 else ['a215', 'a213']), (index // 2) % 2


def execute_aba(runners, gate, emit):
    for label in ['A1', 'B', 'A2']:
        for name in ['a213', 'a215']:
            gate()
            result = runners[name].replay_and_check(label)
            emit(dict(runner=name, label=label, **result))
            if not result['passed']:
                raise AssertionError('Correctness/replay gate failed; timing is forbidden')


def summarize_timing(samples):
    if len(samples) != MEASURED_PAIRS:
        raise ValueError('Incomplete timing samples must not be summarized as a pass')
    if not all(math.isfinite(x['raw_us'][n]) and x['raw_us'][n] > 0
               for x in samples for n in SOURCES):
        raise ValueError('Nonfinite or nonpositive raw timing')
    ratios = [x['raw_us']['a213'] / x['raw_us']['a215'] for x in samples]
    balanced = [math.sqrt(ratios[i] * ratios[i+1]) for i in range(0, len(ratios), 2)]
    return dict(raw_median_us={n: statistics.median(x['raw_us'][n] for x in samples)
                               for n in SOURCES},
                raw_min_max_us={n: [min(x['raw_us'][n] for x in samples),
                                    max(x['raw_us'][n] for x in samples)] for n in SOURCES},
                parent_over_candidate_ratios=ratios,
                reverse_order_pair_geomeans=balanced,
                median_reverse_order_pair_ratio=statistics.median(balanced),
                qualification='Six reverse-order pair ratios from two fixtures only; no official calibration, confidence interval or overall score')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(args):
    start = time.monotonic()
    report = dict(kind='Focused H200 correctness/replay and paired timing; not official score',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        correctness_complete=False, timing_complete=False, cases=[], phases=[],
        source_sha256={n: sha for n, (_, sha) in SOURCES.items()}, support_sha256=SUPPORT,
        protocol=dict(required_cases=[ADVERSARIAL, TARGET], graph_captures_per_case=2,
            replay_labels=['A1', 'B', 'A2'], oracle_calls_per_runner_per_case=2,
            warmup_pairs=WARMUP_PAIRS, measured_pairs=MEASURED_PAIRS,
            input_versions=2, timing_replay_check='Exact output/newKV bytes against same-source, same-input oracle-checked result; full immutable-input/untouched-row check',
            eviction_bytes=256*1024*1024, measured_scope='Whole kernel_fn CUDA graph replay',
            timed_region_excludes='Input/cache restores, eviction, validation and calibration',
            concurrency_tested=False, all_official_shapes_tested=False,
            unchanged_control_tested=False, precision='Unmodified exact candidate sources'))

    def save():
        report['worker_wall_seconds'] = time.monotonic() - start
        data = json.dumps(report, separators=(',', ':')) + '\n'
        if len(data.encode()) > 100000:
            raise RuntimeError('Small-result guard exceeded')
        temp = args.output.with_suffix('.tmp')
        temp.write_text(data)
        temp.replace(args.output)

    def gate():
        save()
        if time.monotonic() - start >= args.deadline:
            raise TimeoutError('Bounded paired worker deadline; preserve partial evidence')

    def phase(name, before):
        report['phases'].append(dict(name=name, wall_seconds=time.monotonic()-before))
        gate()

    save()
    try:
        for filename, sha in [*SOURCES.values(), *SUPPORT.items()]:
            if hashlib.sha256((HERE/filename).read_bytes()).hexdigest() != sha:
                raise ValueError('Frozen payload hash mismatch: ' + filename)
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name():
            raise RuntimeError('Native H200 required; no fallback')
        report['environment'] = dict(gpu=t.cuda.get_device_name(),
            capability=t.cuda.get_device_capability(), torch=t.__version__,
            cuda=t.version.cuda, triton=triton.__version__, python=platform.python_version())
        sys.path.insert(0, str(HERE/'official'))
        h = load(HERE/'modal_worker.py', 'a215_pair_helpers')
        ref = load(HERE/'official/reference.py', 'a215_pair_reference')
        modules = {n: load(HERE/f, 'pair_'+n) for n, (f, _) in SOURCES.items()}
        phase('imports_and_hashes', start)

        def fresh(case, version):
            seed = 135701 + version
            data = ref.generate_input(**{k:v for k,v in case.items() if k != 'name'},
                                      seed=seed, device='cuda')
            if case['name'] == ADVERSARIAL['name']:
                lengths = t.tensor(boundary_lengths(version), device='cuda', dtype=t.int32)
                # The original fixed-length fixture fills every page-table slot.
                # Swapping physical page zero into a zero-length request preserves
                # its scattered, bijective mapping and makes new cache row0 valid.
                flat = data['page_table'].reshape(-1)
                target_b = boundary_lengths(version).index(0)
                destination = target_b * data['page_table'].shape[1]
                original_zero = int((flat == 0).nonzero()[0])
                tmp = flat[destination].clone()
                flat[destination] = 0
                flat[original_zero] = tmp
                data['seq_lens'].copy_(lengths)
                data['positions'].copy_(lengths.to(t.int64))
                data['cache_seqlens'].copy_(lengths + 1)
                data['cache_loc'].copy_(ref.new_slots(data).reshape(-1))
                for b, seq in enumerate(boundary_lengths(version)):
                    valid = (seq + 1 + 63) // 64
                    data['page_table'][b, valid:] = -1
                data['scheduler_metadata'] = ref._fa3_scheduler_metadata(
                    data['cache_seqlens'], data['cu_seqlens_q'], 16, 2, 64, 1, 'full')
                assert int(data['cache_loc'][target_b]) == 0
                assert t.equal(data['cache_loc'], ref.new_slots(data).reshape(-1))
            return data

        class Runner:
            def __init__(self, name, inputs):
                self.name, self.inputs = name, inputs
                self.work = h.clone_inputs(inputs[0])
                self.graph, self.out = h.capture(modules[name].kernel_fn, self.work, inputs[0])
                self.addresses = {k: v.data_ptr() for k,v in self.work.items() if isinstance(v,t.Tensor)}
                self.out_address = self.out.data_ptr()
                self.expected = {}

            def raw_checked(self, version, oracle):
                raw = ref.collect(self.work, self.out)
                stable = self.addresses == {k:v.data_ptr() for k,v in self.work.items() if isinstance(v,t.Tensor)}
                stable = stable and self.out.data_ptr() == self.out_address
                intact, why = h.unchanged(self.work, self.inputs[version])
                if oracle:
                    okay, message = ref.check_implementation(self.work, raw)
                    self.expected[version] = raw.clone()
                    exact = None
                else:
                    exact = bool(t.equal(raw, self.expected[version]))
                    okay, message = exact, 'Exact bytes equal prior same-input oracle-checked output/newKV'
                return dict(passed=bool(okay and stable and intact), stable_addresses=stable,
                            immutable_inputs_and_untouched_rows=intact, immutable_message=why,
                            official_oracle_called=oracle, oracle_message=message,
                            exact_output_newkv_return=exact)

            def replay_and_check(self, label):
                version = 1 if label == 'B' else 0
                h.restore(self.work, self.inputs[version])
                self.graph.replay()
                t.cuda.synchronize()
                result = self.raw_checked(version, label != 'A2')
                result['actual_lengths'] = self.inputs[version]['seq_lens'].cpu().tolist()
                return result

            def timed(self, version, eviction):
                h.restore(self.work, self.inputs[version])
                value = h.timed(self.graph, eviction)
                result = self.raw_checked(version, False)
                return value, result

        target_runners = None
        # Shorter adversarial case first; never measure before both cases pass.
        for case in [ADVERSARIAL, TARGET]:
            before = time.monotonic()
            gate()
            inputs = [fresh(case, version) for version in range(2)]
            if t.equal(inputs[0]['seq_lens'], inputs[1]['seq_lens']):
                raise AssertionError('A/B actual lengths did not change')
            item = dict(case=case, seeds=[135701,135702], records=[])
            report['cases'].append(item)
            runners = {n: Runner(n, inputs) for n in SOURCES}
            execute_aba(runners, gate, lambda row: (item['records'].append(row), save()))
            item['passed'] = True
            phase(case['name']+'_prepare_capture_and_aba', before)
            if case is TARGET:
                target_runners = runners
            else:
                del runners, inputs
        report['correctness_complete'] = True
        gate()
        before = time.monotonic()
        eviction = t.ones(256*1024*1024//4, device='cuda')
        samples, warmups = [], []
        report['timing'] = dict(samples=samples, warmups=warmups, calibration_us=[])
        empty = t.cuda.CUDAGraph()
        with t.cuda.graph(empty):
            pass
        for _ in range(2):
            empty.replay()
        report['timing']['calibration_us'] = [h.timed(empty, eviction) for _ in range(3)]
        for i in range(WARMUP_PAIRS + MEASURED_PAIRS):
            gate()
            order, version = order_and_version(i)
            row = dict(index=i, order=order, input_version=version, raw_us={}, checks={})
            # Persist each completed arm so a timeout retains its evidence.
            (warmups if i < WARMUP_PAIRS else samples).append(row)
            for name in order:
                gate()
                value, check = target_runners[name].timed(version, eviction)
                row['raw_us'][name] = value
                row['checks'][name] = check
                save()
                if not check['passed']:
                    raise AssertionError('Timing replay changed oracle-validated bytes or immutable inputs')
        report['timing']['calibration_us'] += [h.timed(empty, eviction) for _ in range(3)]
        offset = statistics.median(report['timing']['calibration_us'])
        report['timing']['empty_graph_median_us'] = offset
        for row in warmups + samples:
            row['calibrated_us'] = {n: raw-offset for n,raw in row['raw_us'].items()}
        report['timing']['calibration_resolves_both'] = all(
            value > 0 for row in samples for value in row['calibrated_us'].values())
        report['timing']['summary'] = summarize_timing(samples)
        report['timing']['calibration_qualification'] = 'Separate empty-graph subtraction, no clipping; not official calibration'
        report['timing_complete'] = True
        phase('target_paired_timing', before)
        report.update(status='complete', passed=True)
        save()
        return 0
    except Exception as error:
        report.update(status='incomplete_or_failed', error_type=type(error).__name__, error=str(error)[:600])
        save()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=52)
    raise SystemExit(main(parser.parse_args()))
