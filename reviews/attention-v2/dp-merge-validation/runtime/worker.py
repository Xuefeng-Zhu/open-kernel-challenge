"""Bounded DP multi-chunk correctness validation of unchanged composition.

No timing, profiler, Modal, credentials, network, source patch or official job.
The complete entrypoints execute in independent default-pool CUDA graphs.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
SOURCE_ROOT = V2 / 'fusion-lookahead-probe'
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
SEEDS = [135721, 135722]
CASES = [
    dict(name='DP_page64_ctx8193_soft_scaled', batch=32, q_len=1, ctx=8193,
         heads='dp', layer='full', page=64, kv_scale=1/64, spread='fixed', dist='soft'),
    dict(name='DP_page1_ctx16385_coherent', batch=32, q_len=1, ctx=16385,
         heads='dp', layer='full', page=1, kv_scale=1., spread='fixed', dist='coherent'),
]


def boundary_lengths(ctx, version):
    if version not in (0, 1):
        raise ValueError('Only two declared fixture versions')
    if ctx == 8193:
        values = [0, 1, 63, 64, 65, 8191, 8192, 8193] * 4
    elif ctx == 16385:
        values = [0, 1, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193,
                  16320, 16321, 16383, 16384, 16385] * 2
    else:
        raise ValueError('Only the two declared DP capacities')
    return values if version == 0 else list(reversed(values))


def valid_pages(seq, length, page):
    if seq < 0 or length <= 0 or page not in (1, 64):
        raise ValueError('Invalid fixed-protocol cache extent')
    return (seq + length + page - 1) // page


def isolation_schedule():
    # The primary graph has own oracle-checked versions 0/1 from ABA. The
    # independent graph has own version0 from its setup oracle and version1
    # from its concurrent oracle check. Both active replays mutate inputs.
    # Never use nonoracle exact comparison against an uninitialized version.
    return [(0, 1, 1, 1), (1, 0, 0, 1)]


def geometry(case):
    """Shape-only native launch contract, verified against exact source below."""
    width = valid_pages(case['ctx'], case['q_len'], case['page'])
    span = width * case['page']
    splits = (span + 8191) // 8192
    merge_bs = 1 << (splits - 1).bit_length()
    n = 32 * 16 * splits * 16
    cache_bytes = 32 * span * 16 * 64  # E4M3, per K or V cache tensor
    return dict(page_table_shape=[32, width], shape_bound_tokens=span,
        chunk_size=8192, splits=splits, merge_bs=merge_bs,
        producer_grid=[32, 16, splits], merge_grid=[32, 128], block_m=16,
        partial_shapes=dict(o=[n, 64], m=[n], l=[n]),
        source_derived_partial_bytes=n * (64 + 2) * 4,
        cache_bytes_per_tensor=cache_bytes,
        qualification='Exact-source shape-derived allocation/launch contract; internal scratch pointers are not exposed')


def source_gate(source=SOURCE_ROOT, support=V2):
    observed = {}
    for root, hashes in [(source, SOURCE_HASHES), (support, SUPPORT_HASHES)]:
        for name, expected in hashes.items():
            got = hashlib.sha256((root/name).read_bytes()).hexdigest()
            if got != expected:
                raise ValueError('Frozen source/support mismatch: ' + name)
            observed[name] = got
    candidate = ast.parse((source/'candidate.py').read_text())
    function = next(node for node in candidate.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_e2_kernel_fn')
    text = ast.get_source_segment((source/'candidate.py').read_text(), function)
    required = ['chunk_size = 8192 if window == 0 and batch in (16, 32) else 0',
        'page_lookahead = batch == 32 and length == 1 and window == 0 and not rope',
        'part_o = torch.empty((n, d), dtype=torch.float32, device=dev)',
        'part_m = torch.empty((n,), dtype=torch.float32, device=dev)',
        'part_l = torch.empty((n,), dtype=torch.float32, device=dev)',
        'if separate_merge and splits > 1:', 'BS=triton.next_power_of_2(splits)']
    if not all(item in text for item in required):
        raise AssertionError('Exact-source allocation/merge contract changed')
    return observed


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(args):
    start = time.monotonic()
    report = dict(kind='Unchanged composition DP multichunk correctness robustness',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        correctness_complete=False, official_check_admitted=False, timings=[],
        source_sha256=SOURCE_HASHES, support_sha256=SUPPORT_HASHES,
        cases=[], independent_setup_checks=[], concurrent_submissions=[], phases=[], counts={},
        protocol=dict(correctness_only=True, seeds=SEEDS, cases=CASES,
            replay_labels=['A1', 'B', 'A2'], original_tolerances=True,
            pool='Each graph uses its default private allocation pool; no explicit pool sharing',
            scratch='Source allocates fresh FP32 partial tensors per captured invocation; direct scratch pointers unobserved',
            concurrency='Independent captured candidate graphs, disjoint caches/outputs and separate streams; no actual overlap measurement',
            no_performance_claim=True))

    def save():
        report['worker_wall_seconds'] = time.monotonic() - start
        encoded = json.dumps(report, separators=(',', ':')) + '\n'
        if len(encoded.encode()) > 400000:
            raise RuntimeError('Result exceeds bounded 400KB transport')
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(encoded)
        temporary.replace(args.output)

    def gate():
        save()
        if time.monotonic() - start >= args.deadline:
            raise TimeoutError('DP correctness worker deadline')

    def alarm(*_):
        raise TimeoutError('DP correctness worker alarm')

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
        report['environment'] = dict(gpu=t.cuda.get_device_name(), capability=t.cuda.get_device_capability(),
            torch=t.__version__, cuda=t.version.cuda, triton=triton.__version__, python=platform.python_version())
        sys.path.insert(0, str(V2/'official'))
        helpers = load(V2/'modal_worker.py', 'dp_multichunk_helpers')
        ref = load(V2/'official/reference.py', 'dp_multichunk_reference')
        baseline = load(SOURCE_ROOT/'baseline.py', 'dp_multichunk_baseline')
        candidate = load(SOURCE_ROOT/'candidate.py', 'dp_multichunk_candidate')
        functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)

        def fresh(case, version):
            data = ref.generate_input(**{key: value for key, value in case.items() if key != 'name'},
                                      seed=SEEDS[version], device='cuda')
            values = boundary_lengths(case['ctx'], version)
            lengths = t.tensor(values, device='cuda', dtype=t.int32)
            table = data['page_table']
            page, length = case['page'], case['q_len']
            target_b = values.index(0)
            flat = table.reshape(-1)
            zero_at = int((flat == 0).nonzero()[0])
            destination = target_b * table.shape[1]
            previous = flat[destination].clone()
            flat[destination] = 0
            flat[zero_at] = previous
            data['seq_lens'].copy_(lengths)
            data['positions'].copy_((lengths.long()[:, None] + t.arange(length, device='cuda')).reshape(-1))
            data['cache_seqlens'].copy_(lengths + length)
            data['cache_loc'].copy_(ref.new_slots(data).reshape(-1))
            for b, seq in enumerate(values):
                table[b, valid_pages(seq, length, page):] = -1
            data['scheduler_metadata'] = ref._fa3_scheduler_metadata(
                data['cache_seqlens'], data['cu_seqlens_q'],
                data['config']['num_q_heads'], data['config']['num_kv_heads'], page, length, case['layer'])
            assert data['scheduler_metadata'] is not None
            assert int(data['cache_loc'][target_b * length]) == 0
            assert t.equal(data['cache_loc'], ref.new_slots(data).reshape(-1))
            assert bool((table == -1).any())
            assert bool((data['cache_loc'] >= 0).all())
            assert len(t.unique(data['cache_loc'])) == 32
            contract = geometry(case)
            cfg = data['config']
            assert (cfg['batch'], cfg['q_len'], cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'],
                    cfg['window'], cfg['rope_theta']) == (32, 1, 128, 16, 64, 0, 0.)
            assert list(table.shape) == contract['page_table_shape']
            assert contract['splits'] == (2 if page == 64 else 3)
            for b, seq in enumerate(values):
                count = valid_pages(seq, length, page)
                assert bool((table[b, :count] >= 0).all())
                assert bool((table[b, count:] == -1).all())
            return data

        def byte_row(key, left, right):
            a = left.contiguous().reshape(-1).view(t.uint8)
            b = right.contiguous().reshape(-1).view(t.uint8)
            return dict(name=key, bytes=a.numel(), exact=bool(t.equal(a, b)),
                        unequal_byte_count=int((a != b).sum()))

        class Runner:
            def __init__(self, name, inputs):
                self.name, self.inputs = name, inputs
                self.work = helpers.clone_inputs(inputs[0])
                self.graph, self.out = helpers.capture(functions[name], self.work, inputs[0])
                self.addresses = {key: value.data_ptr() for key, value in self.work.items() if isinstance(value, t.Tensor)}
                self.out_address = self.out.data_ptr()
                pool_method = getattr(self.graph, 'pool', None)
                self.pool = tuple(pool_method()) if callable(pool_method) else None
                self.expected = {}

            def evidence(self):
                return dict(output_address=self.out_address,
                    graph_private_pool=None if self.pool is None else list(self.pool),
                    graph_pool_handle_observed=self.pool is not None,
                    graph_pool_qualification='Distinct handles checked when API exists; otherwise default-pool helper source contract only',
                    k_cache_address=self.work['k_cache'].data_ptr(), v_cache_address=self.work['v_cache'].data_ptr(),
                    output_shape=list(self.out.shape), direct_scratch_pointers_observed=False)

            def check(self, version, oracle):
                raw = ref.collect(self.work, self.out)
                addresses = {key: value.data_ptr() for key, value in self.work.items() if isinstance(value, t.Tensor)}
                stable = addresses == self.addresses and self.out.data_ptr() == self.out_address
                intact, why = helpers.unchanged(self.work, self.inputs[version])
                exact = bool(t.equal(raw, self.expected[version])) if version in self.expected else None
                finite = bool(t.isfinite(self.out).all())
                if oracle:
                    okay, message = ref.check_implementation(self.work, raw)
                    self.expected[version] = raw.clone()
                    okay = okay and exact is not False
                else:
                    okay, message = exact is True, 'Exact own oracle-checked output/new-KV bytes'
                return dict(passed=bool(okay and stable and intact and finite), stable_addresses=stable,
                    immutable_inputs_and_untouched_rows=intact, immutable_message=why,
                    official_oracle_called=oracle, oracle_message=message,
                    exact_output_newkv_return=exact, finite_output=finite)

            def replay(self, version, oracle=True):
                helpers.restore(self.work, self.inputs[version])
                self.graph.replay()
                t.cuda.synchronize()
                return self.check(version, oracle)

        for case_index, case in enumerate(CASES):
            phase_start = time.monotonic()
            gate()
            inputs = [fresh(case, version) for version in (0, 1)]
            for key in ['seq_lens', 'qkv', 'page_table']:
                assert not t.equal(inputs[0][key], inputs[1][key])
            item = dict(case=case, geometry=geometry(case),
                input_lengths=[value['seq_lens'].cpu().tolist() for value in inputs],
                scheduler_metadata_shapes=[list(value['scheduler_metadata'].shape) for value in inputs],
                records=[], cache_cross_arm=[])
            report['cases'].append(item)
            runners = {name: Runner(name, inputs) for name in NAMES}
            item['capture_evidence'] = {name: runner.evidence() for name, runner in runners.items()}
            assert runners['baseline'].out_address != runners['candidate'].out_address
            if runners['baseline'].pool is not None and runners['candidate'].pool is not None:
                assert runners['baseline'].pool != runners['candidate'].pool
            for label, version in [('A1', 0), ('B', 1), ('A2', 0)]:
                for name in NAMES:
                    gate()
                    row = dict(runner=name, label=label, input_version=version,
                               **runners[name].replay(version))
                    item['records'].append(row)
                    if not row['passed']:
                        raise AssertionError('DP original oracle/replay gate failed')
                comparisons = [byte_row(key, runners['baseline'].work[key], runners['candidate'].work[key])
                               for key in ['k_cache', 'v_cache']]
                item['cache_cross_arm'].append(dict(label=label, input_version=version,
                    comparisons=comparisons, passed=all(row['exact'] for row in comparisons)))
                if not item['cache_cross_arm'][-1]['passed']:
                    raise AssertionError('DP cross-arm K/V cache bytes differ')
            if case_index == 0:
                first = runners['candidate']
                second = Runner('candidate', inputs)
                pool_observed = first.pool is not None and second.pool is not None
                if pool_observed:
                    assert first.pool != second.pool
                assert first.out_address != second.out_address
                for key in ['k_cache', 'v_cache']:
                    assert first.work[key].data_ptr() != second.work[key].data_ptr()
                setup_check = dict(case=case['name'], runner='candidate_independent', input_version=0,
                    **second.replay(0, True))
                report['independent_setup_checks'].append(setup_check)
                if not setup_check['passed']:
                    raise AssertionError('Independent DP graph own-version0 setup oracle failed')
                gate()
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
                    disjoint_outputs=True,
                    distinct_default_graph_pools=True if pool_observed else None,
                    capture_evidence=[first.evidence(), second.evidence()],
                    separate_graph_captures=True, separate_streams=True,
                    actual_hardware_overlap_measured=False, direct_scratch_pointers_observed=False,
                    checks=checks, isolation_checks=[], passed=all(row['passed'] for row in checks))
                report['concurrent_submissions'].append(concurrent)
                if not concurrent['passed']:
                    raise AssertionError('Independent DP graph submission gate failed')
                # Reuse each graph's output and captured allocations, then check
                # the other graph's retained output/cache without replaying it.
                independent = [first, second]
                for active_index, version, inactive_index, retained_version in isolation_schedule():
                    active, inactive = independent[active_index], independent[inactive_index]
                    gate()
                    concurrent['isolation_checks'].append(dict(action='replay_active', input_version=version,
                        **active.replay(version, False)))
                    concurrent['isolation_checks'].append(dict(action='check_inactive_without_replay', input_version=retained_version,
                        **inactive.check(retained_version, False)))
                concurrent['passed'] = all(row['passed'] for row in concurrent['isolation_checks'])
                if not concurrent['passed']:
                    raise AssertionError('Independent graph output/allocation reuse isolation failed')
                del first, second, streams, independent, active, inactive, runner
            item['passed'] = True
            report['phases'].append(dict(name=case['name'], wall_seconds=time.monotonic()-phase_start))
            del runners, inputs
            gate()
        records = [row for case in report['cases'] for row in case['records']]
        extra = [row for group in report['concurrent_submissions'] for row in group['checks']]
        setup = report['independent_setup_checks']
        isolation = [row for group in report['concurrent_submissions'] for row in group['isolation_checks']]
        report['counts'] = dict(original_oracle_calls=sum(row['official_oracle_called'] for row in records+extra+setup),
            independent_setup_oracle_calls=sum(row['official_oracle_called'] for row in setup),
            exact_return_A_checks=sum(row['label'] == 'A2' and row['exact_output_newkv_return'] is True for row in records),
            cross_arm_cache_byte_comparisons=sum(len(label['comparisons']) for case in report['cases'] for label in case['cache_cross_arm']),
            concurrent_original_oracle_calls=sum(row['official_oracle_called'] for row in extra),
            post_concurrent_isolation_checks=len(isolation),
            extra_candidate_replays=sum(row['action'] == 'replay_active' for row in isolation))
        assert report['counts'] == dict(original_oracle_calls=15, independent_setup_oracle_calls=1, exact_return_A_checks=4,
            cross_arm_cache_byte_comparisons=12, concurrent_original_oracle_calls=2,
            post_concurrent_isolation_checks=4, extra_candidate_replays=2)
        report.update(status='passed', passed=True, correctness_complete=True,
            timing_omitted='Correctness-only validation; no new official or performance claim')
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
    raise SystemExit(main(parser.parse_args()))
