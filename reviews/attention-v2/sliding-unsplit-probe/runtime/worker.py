"""Source-bound sliding unsplit diagnostic; local preparation until separately launched.

No Modal, network, credentials, official check or loaded-module mutation.
The candidate alone requests num_splits=1; one activity session is separate from timing. All complete-call work stays inside each CUDA graph.
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
    'candidate.py': '586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf',
}
SUPPORT_HASHES = {
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
NAMES = ['baseline', 'candidate']
TARGET_PAIRS = [(16, 8), (32, 8)]
IDENTITY_SHA256 = '46fd1727b4744ce74f4eb4514db704710451dffba527d28abefe2c8800e9e4a0'
TARGETS = [dict(name=f'sliding_B{b}_L8_scored', batch=b, q_len=8,
    ctx=80000, heads='tp8', layer='sliding', page=1, kv_scale=1.,
    spread='agentic', dist='soft', boundary=False, edge=False) for b, _ in TARGET_PAIRS]
BOUNDARIES = [dict(name=f'sliding_B{b}_L8_boundary', batch=b, q_len=8,
    ctx=2049, heads='tp8', layer='sliding', page=64, kv_scale=1/64,
    spread='fixed', dist='soft', boundary=True, edge=False) for b, _ in TARGET_PAIRS]
EDGES = [dict(name=f'sliding_B{b}_L8_edges', batch=b, q_len=8,
    ctx=2048, heads='tp8', layer='sliding', page=1, kv_scale=1.,
    spread='fixed', dist='edges', boundary=False, edge=True) for b, _ in TARGET_PAIRS]
CONTROL = dict(name='full_B16_L8_unchanged_control', batch=16, q_len=8,
    ctx=80000, heads='tp8', layer='full', page=1, kv_scale=1.,
    spread='agentic', dist='coherent', boundary=False, edge=False)
WARMUP_SAMPLES = 4
MEASURED_SAMPLES = 12
ACTIVITY_EVENT_BYTES = 128 * 1024


def boundary_lengths(batch, version):
    base = [0, 1, 63, 64, 65, 2047, 2048, 2049]
    values = base * (batch // len(base))
    return values if version == 0 else list(reversed(values))


def order_and_version(index):
    return (NAMES if index % 2 == 0 else list(reversed(NAMES))), (index // 2) % 2


def source_gate(probe=PROBE, support=V2):
    observed = {'identity.py': IDENTITY_SHA256}
    if hashlib.sha256((HERE/'identity.py').read_bytes()).hexdigest() != IDENTITY_SHA256:
        raise ValueError('Frozen identity helper mismatch')
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
    saved = [statistics.mean(samples[j]['raw_us']['baseline']-samples[j]['raw_us']['candidate']
                             for j in [i, i+1]) for i in range(0, len(samples), 2)]
    return dict(raw_median_us={name: statistics.median(row['raw_us'][name] for row in samples)
                              for name in NAMES},
        raw_range_us={name: [min(row['raw_us'][name] for row in samples),
                            max(row['raw_us'][name] for row in samples)] for name in NAMES},
        baseline_over_candidate_ratios=ratios,
        reverse_order_pair_geomeans=balanced,
        median_reverse_order_pair_ratio=statistics.median(balanced),
        reverse_order_pair_mean_savings_us=saved,
        median_reverse_order_pair_mean_saving_us=statistics.median(saved),
        positive_reverse_order_pairs=sum(ratio > 1 for ratio in balanced),
        qualification='Six reverse-order pairs, two input versions, one session; no confidence interval or official score')


def classify_kernel(name):
    lower = name.lower()
    if 'flashattnfwdcombine' in lower or 'flash_fwd_splitkv_combine' in lower:
        return 'combine'
    if 'prepare_varlen_num_blocks' in lower:
        return 'schedule_prepare'
    if 'flashattnfwd' in lower or 'flash_fwd_kernel' in lower:
        return 'mainloop'
    return 'other'


def activity_gate(events, labels):
    """Attribute exact CUDA activity to synchronized CPU marker intervals.

    Complete names and every observed CUDA interval remain in the bounded
    report. Unassigned CUDA activity or a missing recognized mainloop/combine
    contract fails closed before timing; other kernel names remain visible.
    Ordinary activity cannot establish CTA occupancy, grid or hardware stalls.
    """
    encoded = json.dumps(events, separators=(',', ':')).encode('utf-8')
    if len(encoded) > ACTIVITY_EVENT_BYTES:
        raise ValueError('Activity event list exceeds exact 128KiB transport')
    markers = {}
    for row in events:
        if row['name'] in labels and row['device_type'] != 'CUDA':
            if row['name'] in markers:
                raise ValueError('Duplicate activity marker')
            markers[row['name']] = row
    if set(markers) != set(labels):
        raise ValueError('Incomplete activity marker set')
    groups = {label: dict(events=[], counts={kind: 0 for kind in
              ['mainloop', 'combine', 'schedule_prepare', 'other']}) for label in labels}
    unassigned = []
    for row in events:
        if not all(math.isfinite(row[key]) for key in ['start_us', 'end_us']):
            raise ValueError('Nonfinite profiler interval')
        if row['end_us'] < row['start_us']:
            raise ValueError('Negative profiler interval')
        if row['device_type'] != 'CUDA':
            continue
        matched = [name for name, marker in markers.items()
                   if row['start_us'] >= marker['start_us'] and row['end_us'] <= marker['end_us']]
        if len(matched) != 1:
            unassigned.append(row)
            continue
        name = matched[0]
        kind = classify_kernel(row['name'])
        groups[name]['events'].append({**row, 'kind': kind})
        groups[name]['counts'][kind] += 1
    mechanism = []
    for case in TARGETS:
        base, cand = [groups[activity_label(case, name)]['counts'] for name in NAMES]
        mechanism.append(dict(case=case['name'], baseline_counts=base, candidate_counts=cand,
            passed=base['combine'] >= 1 and cand['combine'] == 0
                and base['mainloop'] >= 1 and cand['mainloop'] >= 1,
            scheduler_prepare_removed=cand['schedule_prepare'] == 0 and base['schedule_prepare'] > 0,
            scheduler_prepare_qualification='Actual counts only; preparation may persist with num_splits=1'))
    return dict(passed=not unassigned and all(row['passed'] for row in mechanism),
        sessions=1, marker_intervals=markers, groups=groups, unassigned_cuda_events=unassigned,
        mechanism=mechanism, raw_event_list=events, event_list_utf8_bytes=len(encoded),
        event_names_truncated=False, hardware_counters=False,
        qualification='Instrumented one-replay CUDA activity for mechanism only; never benchmark latency or exact CTA/occupancy evidence')


def activity_label(case, name):
    return 'UNSPLIT_MECHANISM_' + case['name'] + '_' + name


def activity_diagnostic(t, retained, cases, gate):
    labels = [activity_label(case, name) for case in cases for name in NAMES]
    checks = []
    for case in cases:
        for name in NAMES:
            runner = retained[case['name']][name]
            check = runner.replay(0, False)
            if not check['passed']:
                raise AssertionError('Pre-activity replay/cache check failed')
    gate()
    with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                                       t.profiler.ProfilerActivity.CUDA]) as profile:
        for case in cases:
            for name in NAMES:
                with t.profiler.record_function(activity_label(case, name)):
                    retained[case['name']][name].graph.replay()
                    t.cuda.synchronize()
    rows = []
    for event in profile.events():
        device = str(event.device_type).split('.')[-1]
        if device == 'CUDA' or event.name in labels:
            rows.append(dict(name=event.name, device_type=device,
                start_us=float(event.time_range.start), end_us=float(event.time_range.end)))
    output = activity_gate(rows, labels)
    for case in cases:
        for name in NAMES:
            checks.append(dict(case=case['name'], runner=name,
                               **retained[case['name']][name].check(0, False)))
    output['replay_checks'] = checks
    output['passed'] = output['passed'] and all(row['passed'] for row in checks)
    return output


def benefit_gate(timings):
    expected = {case['name'] for case in TARGETS + [CONTROL]}
    if len(timings) != 3 or {row['case']['name'] for row in timings} != expected:
        raise ValueError('Complete two-target and unchanged-control timings required')
    summaries = {row['case']['name']: timing_summary(row['samples']) for row in timings}
    targets = []
    for case in TARGETS:
        summary = summaries[case['name']]
        saving = summary['median_reverse_order_pair_mean_saving_us']
        positive = summary['positive_reverse_order_pairs']
        targets.append(dict(case=case['name'], median_paired_mean_saving_us=saving,
            positive_reverse_order_pairs=positive, passed=saving >= 2 and positive == 6))
    control = summaries[CONTROL['name']]
    control_slowdown = 1/control['median_reverse_order_pair_ratio']-1
    control_ok = control_slowdown <= .05
    return dict(admitted=all(row['passed'] for row in targets) and control_ok,
        targets=targets, unchanged_control=dict(case=CONTROL['name'],
            paired_candidate_slowdown_percent=100*control_slowdown, passed=control_ok),
        predeclared_gate='Each target median paired mean saving >=2us and 6/6 positive reverse-order ratios; unchanged control slowdown <=5%',
        qualification='One session only; useful evidence permits parent consideration of one official check, not automatic submission or promotion')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(args):
    start = time.monotonic()
    report = dict(kind='Sliding B16/B32 num_splits=1 diagnostic, not official score',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        source_sha256=SOURCE_HASHES, support_sha256=SUPPORT_HASHES,
        cases=[], prep_comparisons=[], concurrent_submissions=[], timings=[], phases=[],
        fa3_identity=None, activity=None, mechanism_complete=False, benefit=None,
        preparation_complete=False, correctness_complete=False, timing_complete=False,
        official_check_admitted=False,
        protocol=dict(target_pairs=TARGET_PAIRS, seeds=[135701, 135702],
            replay_labels=['A1', 'B', 'A2'], cases=BOUNDARIES + EDGES + TARGETS + [CONTROL],
            prep_gate='Exact native FP8 Q and entire post-store K/V bytes for both input versions of all six sliding cases before any complete-call timing',
            warmup_two_arm_samples=WARMUP_SAMPLES, measured_two_arm_samples=MEASURED_SAMPLES,
            eviction_bytes=256 * 1024 * 1024,
            timed_region='Entire kernel_fn CUDA graph device work: preparation, cache stores, FA3 scheduling/compute/combine and output gate; graph memory reused on replay, host allocation excluded',
            excludes='Fixture/scheduler-metadata creation, cache/input restore, L2 eviction, validation, empty-graph diagnostic',
            tolerances='Unmodified official/reference.py and official/checks.py',
            all_official_shapes_tested=False, activity_sessions=1, hardware_counters=False,
            performance_gate='Each sliding target median reverse-order pair mean saving >=2us, all six ratio pairs positive; unchanged full B16 verify <=5% median ratio regression',
            correctness_only=args.correctness_only,
            concurrency='Two independent graph captures with private default allocation pools, concurrently submitted on separate streams for each boundary target; no claim of actual hardware overlap'))

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
            raise TimeoutError('Sliding unsplit worker deadline')

    def alarm(*_):
        raise TimeoutError('Sliding unsplit worker alarm')

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
        # Import installed wrappers, inspect the actual callable before any
        # candidate invocation, CUDA availability query, fixture or activity.
        sys.path.insert(0, str(V2 / 'official'))
        identity = load(HERE / 'identity.py', 'unsplit_identity')
        baseline = load(PROBE / 'baseline.py', 'unsplit_runtime_baseline')
        candidate = load(PROBE / 'candidate.py', 'unsplit_runtime_candidate')
        report['fa3_identity'] = identity.inspect_fa3(baseline.flash_attn_with_kvcache)
        report['fa3_identity']['same_callable_in_both_modules'] = (
            baseline.flash_attn_with_kvcache is candidate.flash_attn_with_kvcache)
        gate()
        if not report['fa3_identity']['admitted']:
            raise RuntimeError('Installed explicit num_splits keyword API not verified; no candidate call')
        if not report['fa3_identity']['exact_observed_extensions_verified']:
            raise RuntimeError('Installed observed FA3 extension identity not fully verified: '
                + report['fa3_identity']['extension_identity_reason'] + '; no candidate call')
        if not report['fa3_identity']['same_callable_in_both_modules']:
            raise RuntimeError('Baseline/candidate refer to different installed FA3 callables')
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name():
            raise RuntimeError('Native H200 required')
        report['environment'] = dict(gpu=t.cuda.get_device_name(),
            capability=t.cuda.get_device_capability(), torch=t.__version__,
            cuda=t.version.cuda, triton=triton.__version__, python=platform.python_version())
        helpers = load(V2 / 'modal_worker.py', 'unsplit_runtime_helpers')
        ref = load(V2 / 'official/reference.py', 'unsplit_runtime_reference')
        functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)
        phase('imports_and_hashes', start)

        def fresh(case, version):
            values = {key: val for key, val in case.items() if key not in {'name', 'boundary', 'edge'}}
            data = ref.generate_input(**values, seed=135701 + version, device='cuda')
            if case['boundary']:
                batch, length = case['batch'], case['q_len']
                lengths = t.tensor(boundary_lengths(batch, version), device='cuda', dtype=t.int32)
                target_b = boundary_lengths(batch, version).index(0)
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
                for b, seq in enumerate(boundary_lengths(batch, version)):
                    table[b, (seq + length + 63) // 64:] = -1
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

        def prep_compare(inputs):
            # The same established fused preparation remains in both modules.
            # Compare native Q and entire post-store cache bytes without FA3.
            works = {name: helpers.clone_inputs(inputs) for name in NAMES}
            queries = {}
            cfg = inputs['config']
            batch, length, hq, hk, d = [cfg[key] for key in
                ['batch', 'q_len', 'num_q_heads', 'num_kv_heads', 'head_dim']]
            tokens = batch * length
            for name, module in [('baseline', baseline), ('candidate', candidate)]:
                work = works[name]
                qkv = work['qkv'].view(tokens, -1)
                query = t.empty((tokens, hq, d), device=qkv.device, dtype=t.float8_e4m3fn)
                cosine_sine = module._fa3_cos_sin(cfg['rope_theta'], d, qkv.device)
                module._fa3_prepare_fused[(triton.cdiv(tokens * (hq + 2 * hk), 4),)](
                    qkv, work['positions'], cosine_sine, work['cache_loc'],
                    work['k_cache'], work['v_cache'], work['k_scale'], work['v_scale'],
                    query, qkv.stride(0), work['k_cache'].numel() // (hk * d), module.RMS_EPS,
                    T=tokens, HQ=hq, HKV=hk, D=d, BD=triton.next_power_of_2(d), BM=4,
                    ROPE=True, SCALED=cfg['kv_scale'] != 1., num_warps=4)
                queries[name] = query
            t.cuda.synchronize()
            comparisons = [byte_row('Q_FP8', queries['baseline'], queries['candidate'])]
            comparisons += [byte_row(key, works['baseline'][key], works['candidate'][key])
                            for key in ['k_cache', 'v_cache']]
            intact = [helpers.unchanged(work, inputs) for work in works.values()]
            finite = all(bool(t.isfinite(value.float()).all()) for value in queries.values())
            return dict(passed=all(row['exact'] for row in comparisons) and finite and all(x[0] for x in intact),
                comparisons=comparisons, finite_queries=finite,
                immutable_and_untouched=[dict(passed=x[0], message=x[1]) for x in intact],
                qualification='Same native unchanged preparation only; attention reduction is checked against original oracle, not required bit-identical across split policies')

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
        # Complete all prep comparisons and native replay gates before timing.
        for case in BOUNDARIES + EDGES + TARGETS + [CONTROL]:
            before = time.monotonic()
            gate()
            inputs = [fresh(case, version) for version in [0, 1]]
            if not case['edge']:
                assert not t.equal(inputs[0]['seq_lens'], inputs[1]['seq_lens'])
            assert not t.equal(inputs[0]['qkv'], inputs[1]['qkv'])
            assert not t.equal(inputs[0]['page_table'], inputs[1]['page_table'])
            for version, values in enumerate(inputs) if case['layer'] == 'sliding' else []:
                row = dict(case=case['name'], input_version=version, **prep_compare(values))
                report['prep_comparisons'].append(row)
                gate()
                if not row['passed']:
                    raise AssertionError('Native preparation bytes differ; complete-call timing forbidden')
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
                    raise AssertionError('New/cache bytes differ across arms; timing forbidden')
            if case['boundary']:
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
            elif case in TARGETS or case is CONTROL:
                retained[case['name']] = runners
            else:
                del runners, inputs
            item['passed'] = True
            phase(case['name'] + '_prep_replay', before)
        report['preparation_complete'] = report['correctness_complete'] = True
        gate()
        if args.correctness_only:
            report.update(status='correctness_complete', passed=True,
                timing_omitted='Explicit correctness-only mode; no performance claim')
            save()
            return 0
        report['activity'] = activity_diagnostic(t, retained, TARGETS, gate)
        gate()
        if not report['activity']['passed']:
            raise AssertionError('Actual split/combine mechanism not established; timing forbidden')
        report['mechanism_complete'] = True
        eviction = t.ones(256 * 1024 * 1024 // 4, device='cuda')
        empty = t.cuda.CUDAGraph()
        with t.cuda.graph(empty):
            pass
        for _ in range(2):
            empty.replay()
        for case in TARGETS + [CONTROL]:
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
