"""Common-PV alpha-one validation, isolated frequency probe, paired whole-call timing.

All candidate work remains in complete CUDA graph entrypoints. The separately
frozen diagnostic module is never timed. No source/module mutation, profiler,
Modal, credentials, network, tolerance change or official request is present.
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
    'baseline.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'candidate.py': '14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5',
}
DIAGNOSTIC_HASH = 'ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60'
SUPPORT_HASHES = {
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
NAMES = ['baseline', 'candidate']
SEEDS = [135731, 135732]
TARGET = dict(name='full_B32_L1_scored', batch=32, q_len=1, ctx=80000,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent',
    boundary=False, force=None)
FIRST = dict(name='full_B32_L1_first_max', batch=32, q_len=1, ctx=8193,
    heads='tp8', layer='full', page=64, kv_scale=1/64, spread='fixed', dist='soft',
    boundary=True, force='first')
MIDDLE = dict(FIRST, name='full_B32_L1_middle_max', force='middle')
LAST = dict(FIRST, name='full_B32_L1_last_max', force='last')
DP = dict(FIRST, name='full_B32_L1_DP_boundary', heads='dp', force=None)
CONTROL = dict(name='full_B16_L8_fusion_unchanged', batch=16, q_len=8, ctx=80000,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent',
    boundary=False, force=None)
CASES = [TARGET, FIRST, MIDDLE, LAST, DP, CONTROL]
DIAGNOSTIC_CASES = CASES[:5]
TIMED_CASES = [TARGET, CONTROL]
WARMUP_SAMPLES = 4
MEASURED_SAMPLES = 12


def boundary_lengths(version):
    if version not in (0, 1):
        raise ValueError('Only the two declared fixture versions')
    values = [0, 1, 63, 64, 65, 8191, 8192, 8193] * 4
    return values if version == 0 else list(reversed(values))


def leader_token(seq, placement):
    if seq < 0 or placement not in ('first', 'middle', 'last'):
        raise ValueError('Invalid forced history leader')
    if not seq:
        return None
    tiles = (seq+63)//64
    tile = {'first':0, 'middle':(tiles-1)//2, 'last':tiles-1}[placement]
    return tile*64


def expected_tile_rows(lengths, hkv, splits, force=None):
    if hkv <= 0 or splits <= 0 or force not in (None, 'first', 'middle', 'last'):
        raise ValueError('Invalid CTA layout')
    rows = []
    for seq in lengths:
        if type(seq) is not int or seq < 0 or seq > splits*8192:
            raise ValueError('History length outside fixed split capacity')
        leader = leader_token(seq, force) if force else None
        for _ in range(hkv):
            for split in range(splits):
                count = max(0, min(seq, (split+1)*8192)-split*8192)
                total = (count+63)//64
                slow = None
                if force:
                    slow = int(total > 0)
                    if leader is not None and split*8192 < leader < min(seq, (split+1)*8192):
                        slow += 1
                rows.append(dict(total=total, slow=slow, fast=None if slow is None else total-slow))
    return rows


def stats_check(actual, lengths, hkv, splits, force=None):
    expected = expected_tile_rows(lengths, hkv, splits, force)
    if len(actual) != len(expected):
        raise ValueError('Diagnostic CTA count mismatch')
    for observed, want in zip(actual, expected):
        if len(observed) != 3 or any(type(value) is not int or value < 0 for value in observed):
            raise ValueError('Diagnostic counts must be nonnegative integer triples')
        total, fast, slow = observed
        if fast+slow != total or total != want['total']:
            raise ValueError('Diagnostic visited/fast/slow or CPU history-tile mismatch')
        if total > 0 and slow == 0:
            raise ValueError('First historical tile of each nonempty CTA must take the slow path')
        if force and (slow != want['slow'] or fast != want['fast']):
            raise ValueError('Forced history leader fast/slow count mismatch')
    return dict(passed=True, total=sum(row[0] for row in actual),
        fast=sum(row[1] for row in actual), slow=sum(row[2] for row in actual),
        expected_total_by_cta=[row['total'] for row in expected],
        expected_slow_by_cta=[row['slow'] for row in expected])


def without_stats(data):
    return {key:value for key,value in data.items() if key != 'alpha_stats'}


def order_and_version(index):
    return (NAMES if index%2 == 0 else list(reversed(NAMES))), (index//2)%2


def timing_summary(samples):
    if len(samples) != MEASURED_SAMPLES:
        raise ValueError('Twelve measured two-arm samples required')
    for offset, row in enumerate(samples):
        order, version = order_and_version(offset+WARMUP_SAMPLES)
        if row['index'] != offset+WARMUP_SAMPLES or row['order'] != order or row['input_version'] != version:
            raise ValueError('Measured order/fixture plan changed')
        if not all(math.isfinite(row['raw_us'][name]) and row['raw_us'][name] > 0 for name in NAMES):
            raise ValueError('Nonfinite/nonpositive complete-call time')
    ratios = [row['raw_us']['baseline']/row['raw_us']['candidate'] for row in samples]
    balanced = [math.sqrt(ratios[index]*ratios[index+1]) for index in range(0, len(ratios), 2)]
    versions = [samples[index]['input_version'] for index in range(0, len(samples), 2)]
    per_fixture = {}
    for version in (0, 1):
        values = [ratio for ratio, fixture in zip(balanced, versions) if fixture == version]
        if len(values) != 3:
            raise ValueError('Three reverse-order pairs per fixture required')
        median = statistics.median(values)
        per_fixture[str(version)] = dict(pair_ratios=values, median_ratio=median,
            latency_reduction_percent=100*(1-1/median), slowdown_percent=100*(1/median-1),
            positive_pairs=sum(value > 1 for value in values))
    return dict(raw_median_us={name:statistics.median(row['raw_us'][name] for row in samples) for name in NAMES},
        reverse_order_pair_geomeans=balanced, pair_input_versions=versions,
        median_reverse_order_pair_ratio=statistics.median(balanced),
        positive_reverse_order_pairs=sum(value > 1 for value in balanced), per_fixture=per_fixture,
        qualification='Six complete-call reverse-order pairs; three per fresh fixture; no judge replay-cost adjustment or confidence interval')


def benefit_gate(timings):
    if len(timings) != 2 or {row['case']['name'] for row in timings} != {row['name'] for row in TIMED_CASES}:
        raise ValueError('Target and unchanged control required')
    summaries = {row['case']['name']:timing_summary(row['samples']) for row in timings}
    target, control = summaries[TARGET['name']], summaries[CONTROL['name']]
    target_ok = all(row['latency_reduction_percent'] >= 2. for row in target['per_fixture'].values())
    target_ok = target_ok and target['positive_reverse_order_pairs'] == 6
    control_ok = all(row['slowdown_percent'] <= 2. for row in control['per_fixture'].values())
    return dict(admitted=target_ok and control_ok, target=target, unchanged_control=control,
        target_passed=target_ok, unchanged_control_passed=control_ok,
        predeclared_gate='Target >=2% in EACH fixture and all six positive pairs; unchanged-control slowdown <=2% in EACH fixture',
        qualification='One session only; fresh-input confirmation still required before promotion or a complete official check')


def source_gate(probe=PROBE, support=V2, diagnostic=HERE/'diagnostic.py'):
    observed = {}
    for root, hashes in [(probe, SOURCE_HASHES), (support, SUPPORT_HASHES)]:
        for name, expected in hashes.items():
            if (root/name).is_symlink():
                raise ValueError('Symlink source refused: '+name)
            got = hashlib.sha256((root/name).read_bytes()).hexdigest()
            if got != expected:
                raise ValueError('Frozen source/support mismatch: '+name)
            observed[name] = got
    got = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    if diagnostic.is_symlink() or got != DIAGNOSTIC_HASH:
        raise ValueError('Frozen diagnostic source mismatch')
    observed['diagnostic.py'] = got
    return observed


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(args):
    started = time.monotonic()
    report = dict(kind='Common-PV alpha-one exact validation, separate frequency diagnostic, uninstrumented paired timing',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        source_sha256=SOURCE_HASHES, support_sha256=SUPPORT_HASHES, diagnostic_sha256=DIAGNOSTIC_HASH,
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[], independent_setup_checks=[], concurrent_submissions=[], diagnostics=[], timings=[], phases=[], counts={},
        correctness_complete=False, diagnostic_complete=False, timing_complete=False,
        official_check_admitted=False, benefit=None, frequency_gate=None,
        protocol=dict(seeds=SEEDS, cases=CASES, replay_labels=['A1','B','A2'],
            original_oracle_and_tolerances=True, exact_cross_arm_output_newkv=True,
            diagnostics='Separate frozen module, 3 int32 counters per CTA; designated alpha_stats only writable extra; never timed',
            frequency_gate='Scored-family fast count >0 in EACH fixture; no frequency threshold',
            timed_region='Complete uninstrumented kernel_fn graph device work, raw CUDA events, no judge replay-cost subtraction',
            outside_timing='Fixture/metadata creation, restores, cold-L2 eviction, all validation, all diagnostics',
            cold_l2_eviction_bytes=256*1024*1024, warmup_samples=4, measured_samples=12,
            concurrency='Independent graph captures/default pools/disjoint writable caches and outputs; actual overlap and direct scratch pointers unobserved',
            performance_gate='Target >=2% per fixture, all6 favorable balanced pairs; control slowdown <=2% per fixture',
            profiler_used=False, instrumented_timings_reported=False, all_official_shapes_tested=False))

    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        encoded = json.dumps(report, separators=(',', ':'))+'\n'
        if len(encoded.encode()) > 500000:
            raise RuntimeError('Result exceeds bounded 500KB transport')
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(encoded)
        temporary.replace(args.output)

    def gate():
        save()
        if time.monotonic()-started >= args.deadline:
            raise TimeoutError('Common-PV runtime deadline')

    def alarm(*_):
        raise TimeoutError('Common-PV runtime alarm')

    def phase(name, before):
        report['phases'].append(dict(name=name, wall_seconds=time.monotonic()-before))
        gate()

    if not 0 < args.deadline <= 160:
        raise ValueError('Reviewed worker deadline is at most160s')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(args.deadline))
    save()
    try:
        if __import__('os').environ.get('TRITON_INTERPRET'):
            raise RuntimeError('Native GPU execution required')
        report['verified_hashes'] = source_gate()
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name():
            raise RuntimeError('Native H200 required')
        report['environment'] = dict(gpu=t.cuda.get_device_name(), capability=t.cuda.get_device_capability(),
            torch=t.__version__, cuda=t.version.cuda, triton=triton.__version__, python=platform.python_version())
        sys.path.insert(0, str(V2/'official'))
        helpers = load(V2/'modal_worker.py', 'common_pv_runtime_helpers')
        ref = load(V2/'official/reference.py', 'common_pv_runtime_reference')
        baseline = load(PROBE/'baseline.py', 'common_pv_runtime_baseline')
        candidate = load(PROBE/'candidate.py', 'common_pv_runtime_candidate')
        diagnostic = load(HERE/'diagnostic.py', 'common_pv_runtime_diagnostic')
        functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)
        phase('imports_and_hashes', started)

        def byte_row(name, left, right):
            a = left.contiguous().reshape(-1).view(t.uint8)
            b = right.contiguous().reshape(-1).view(t.uint8)
            same_shape = a.shape == b.shape
            exact = same_shape and bool(t.equal(a, b))
            return dict(name=name, bytes=a.numel(), exact=exact,
                unequal_byte_count=int((a != b).sum()) if same_shape else None)

        def fresh(case, version):
            data = ref.generate_input(**{key:value for key,value in case.items() if key not in {'name','boundary','force'}},
                                      seed=SEEDS[version], device='cuda')
            evidence = None
            if case['boundary']:
                values = boundary_lengths(version)
                lengths = t.tensor(values, device='cuda', dtype=t.int32)
                table, page, length = data['page_table'], case['page'], case['q_len']
                target_b = values.index(0)
                flat = table.reshape(-1)
                zero_at = int((flat == 0).nonzero()[0])
                destination = target_b*table.shape[1]
                previous = flat[destination].clone()
                flat[destination] = 0
                flat[zero_at] = previous
                data['seq_lens'].copy_(lengths)
                data['positions'].copy_((lengths.long()[:,None]+t.arange(length, device='cuda')).reshape(-1))
                data['cache_seqlens'].copy_(lengths+length)
                data['cache_loc'].copy_(ref.new_slots(data).reshape(-1))
                for b, seq in enumerate(values):
                    table[b, (seq+length+page-1)//page:] = -1
                data['scheduler_metadata'] = ref._fa3_scheduler_metadata(
                    data['cache_seqlens'], data['cu_seqlens_q'], data['config']['num_q_heads'],
                    data['config']['num_kv_heads'], page, length, case['layer'])
                assert int(data['cache_loc'][target_b*length]) == 0
                assert t.equal(data['cache_loc'], ref.new_slots(data).reshape(-1))
                assert bool((table == -1).any())
            if case['force']:
                cfg = data['config']
                query, new_key, _, _ = ref.split_qkv(data)
                query.zero_()
                query[...,0] = 1.
                new_key.zero_()
                data['k_cache'].view(t.uint8).zero_()
                keys = data['k_cache'].view(-1, cfg['num_kv_heads'], cfg['head_dim'])
                value = t.tensor(32., device='cuda').to(t.float8_e4m3fn)
                positions = []
                for b, seq in enumerate(data['seq_lens'].cpu().tolist()):
                    token = leader_token(seq, case['force'])
                    positions.append(token)
                    if token is not None:
                        physical = int(data['page_table'][b,token//cfg['page_size']])*cfg['page_size']+token%cfg['page_size']
                        keys[physical,:,0].copy_(value.expand(cfg['num_kv_heads']))
                        assert bool((keys[physical,:,0].float() == 32.).all())
                encoded_q = ref._rms(query).to(t.bfloat16).to(t.float8_e4m3fn).float()
                reference_q = (ref._rms(query)*cfg['qk_scale_factor']).to(t.bfloat16).float()
                assert bool((encoded_q[...,0] > 0).all()) and not bool((encoded_q[...,1:] != 0).any())
                assert bool((reference_q[...,0] > 0).all()) and not bool((reference_q[...,1:] != 0).any())
                assert not bool((new_key != 0).any())
                expected_nonzero = sum(token is not None for token in positions)*cfg['num_kv_heads']
                observed_nonzero = int((data['k_cache'].view(t.uint8) != 0).sum())
                assert observed_nonzero == expected_nonzero
                evidence = dict(placement=case['force'], historical_leader_positions=positions,
                    encoded_query_dim0_positive=True, reference_query_dim0_positive=True,
                    query_other_components_zero=True, new_keys_zero=True,
                    cache_nonzero_components=observed_nonzero, expected_nonzero_components=expected_nonzero,
                    leader_raw_e4m3_value=32., nonleader_history_keys_zero=True,
                    fixture_change='Only Q, new K and historical K edited outside graph; V/newV/gate remain fresh')
            if case['q_len'] == 1 and data['scheduler_metadata'] is None:
                raise AssertionError('Actual serving-supplied decode metadata required')
            if case['q_len'] == 8 and data['scheduler_metadata'] is not None:
                raise AssertionError('Verify metadata must remain None')
            return data, evidence

        class Runner:
            def __init__(self, name, inputs):
                self.name, self.inputs = name, inputs
                self.work = helpers.clone_inputs(inputs[0])
                self.graph, self.out = helpers.capture(functions[name], self.work, inputs[0])
                self.addresses = {key:value.data_ptr() for key,value in self.work.items() if isinstance(value,t.Tensor)}
                self.out_address = self.out.data_ptr()
                pool_method = getattr(self.graph,'pool',None)
                self.pool = tuple(pool_method()) if callable(pool_method) else None
                self.expected = {}

            def evidence(self):
                return dict(output_address=self.out_address,
                    k_cache_address=self.work['k_cache'].data_ptr(), v_cache_address=self.work['v_cache'].data_ptr(),
                    graph_private_pool=None if self.pool is None else list(self.pool),
                    direct_scratch_pointers_observed=False)

            def check(self, version, oracle):
                raw = ref.collect(self.work,self.out)
                addresses = {key:value.data_ptr() for key,value in self.work.items() if isinstance(value,t.Tensor)}
                stable = addresses == self.addresses and self.out.data_ptr() == self.out_address
                intact, why = helpers.unchanged(self.work,self.inputs[version])
                exact = byte_row('own_output_newkv',raw,self.expected[version])['exact'] if version in self.expected else None
                finite = bool(t.isfinite(self.out).all())
                if oracle:
                    okay, message = ref.check_implementation(self.work,raw)
                    self.expected[version] = raw.clone()
                    okay = okay and exact is not False
                else:
                    okay, message = exact is True, 'Exact own previously original-oracle-checked bytes'
                return dict(passed=bool(okay and stable and intact and finite), stable_addresses=stable,
                    immutable_inputs_and_untouched_rows=intact, immutable_message=why,
                    official_oracle_called=oracle, oracle_message=message,
                    exact_output_newkv_return=exact, finite_output=finite)

            def replay(self,version,oracle=True):
                helpers.restore(self.work,self.inputs[version])
                self.graph.replay()
                t.cuda.synchronize()
                return self.check(version,oracle)

            def timed(self,version,eviction):
                helpers.restore(self.work,self.inputs[version])
                value = helpers.timed(self.graph,eviction)
                return value,self.check(version,False)

        retained, diagnostic_runners = {}, {}
        for case in CASES:
            before = time.monotonic()
            gate()
            pairs = [fresh(case,version) for version in (0,1)]
            inputs = [pair[0] for pair in pairs]
            for key in ['seq_lens','qkv','page_table']:
                assert not t.equal(inputs[0][key],inputs[1][key])
            item = dict(case=case,config=dict(inputs[0]['config']),
                page_table_shape=list(inputs[0]['page_table'].shape),
                input_lengths=[data['seq_lens'].cpu().tolist() for data in inputs],
                leader_evidence=[pair[1] for pair in pairs], records=[],cache_cross_arm=[],output_newkv_cross_arm=[])
            report['cases'].append(item)
            runners = {name:Runner(name,inputs) for name in NAMES}
            item['capture_evidence'] = {name:runner.evidence() for name,runner in runners.items()}
            assert runners['baseline'].out_address != runners['candidate'].out_address
            if runners['baseline'].pool is not None and runners['candidate'].pool is not None:
                assert runners['baseline'].pool != runners['candidate'].pool
            for key in ['k_cache','v_cache']:
                assert runners['baseline'].work[key].data_ptr() != runners['candidate'].work[key].data_ptr()
            for label,version in [('A1',0),('B',1),('A2',0)]:
                for name in NAMES:
                    gate()
                    row = dict(runner=name,label=label,input_version=version,**runners[name].replay(version))
                    item['records'].append(row)
                    if not row['passed']:
                        raise AssertionError('Original oracle/ABA gate failed; diagnostic/timing forbidden')
                comparisons = [byte_row(key,runners['baseline'].work[key],runners['candidate'].work[key]) for key in ['k_cache','v_cache']]
                item['cache_cross_arm'].append(dict(label=label,input_version=version,comparisons=comparisons,
                    passed=all(row['exact'] for row in comparisons)))
                out_comparison = byte_row('output_newkv',ref.collect(runners['baseline'].work,runners['baseline'].out),
                    ref.collect(runners['candidate'].work,runners['candidate'].out))
                item['output_newkv_cross_arm'].append(dict(label=label,input_version=version,**out_comparison))
                if not item['cache_cross_arm'][-1]['passed'] or not out_comparison['exact']:
                    raise AssertionError('Cross-arm output/newKV/cache byte gate failed')
            if case['name'] == MIDDLE['name']:
                first = runners['candidate']
                second = Runner('candidate',inputs)
                assert first.out_address != second.out_address
                if first.pool is not None and second.pool is not None:
                    assert first.pool != second.pool
                for key in ['k_cache','v_cache']:
                    assert first.work[key].data_ptr() != second.work[key].data_ptr()
                setup = dict(case=case['name'],input_version=0,**second.replay(0,True))
                report['independent_setup_checks'].append(setup)
                if not setup['passed']:
                    raise AssertionError('Independent graph own-A setup oracle failed')
                helpers.restore(first.work,inputs[0])
                helpers.restore(second.work,inputs[1])
                streams = [t.cuda.Stream(),t.cuda.Stream()]
                ready = t.cuda.Event()
                ready.record()
                for stream,runner in zip(streams,[first,second]):
                    stream.wait_event(ready)
                    with t.cuda.stream(stream):
                        runner.graph.replay()
                for stream in streams:
                    t.cuda.current_stream().wait_stream(stream)
                t.cuda.synchronize()
                concurrent = dict(case=case['name'],checks=[first.check(0,True),second.check(1,True)],
                    capture_evidence=[first.evidence(),second.evidence()],isolation_checks=[],
                    disjoint_writable_caches=True,disjoint_outputs=True,separate_graph_captures=True,
                    separate_streams=True,actual_hardware_overlap_measured=False,direct_scratch_pointers_observed=False)
                report['concurrent_submissions'].append(concurrent)
                for active,version,inactive,retained_version in [(first,1,second,1),(second,0,first,1)]:
                    gate()
                    concurrent['isolation_checks'].append(dict(action='replay_active',input_version=version,
                        **active.replay(version,False)))
                    concurrent['isolation_checks'].append(dict(action='check_inactive_without_replay',input_version=retained_version,
                        **inactive.check(retained_version,False)))
                concurrent['passed'] = all(row['passed'] for row in concurrent['checks']+concurrent['isolation_checks'])
                if not concurrent['passed']:
                    raise AssertionError('Independent graph submission/reuse isolation failed')
                del first,second,streams,active,inactive,runner
            if case in TIMED_CASES:
                retained[case['name']] = runners
            if case in DIAGNOSTIC_CASES:
                diagnostic_runners[case['name']] = runners['candidate']
            item['passed'] = True
            del runners,inputs,pairs
            phase(case['name']+'_oracle_replay',before)
        originals = [row for case in report['cases'] for row in case['records']]
        extra = [row for group in report['concurrent_submissions'] for row in group['checks']]
        isolation = [row for group in report['concurrent_submissions'] for row in group['isolation_checks']]
        report['counts'] = dict(original_oracle_calls=sum(row['official_oracle_called'] for row in originals+extra+report['independent_setup_checks']),
            independent_setup_oracle_calls=len(report['independent_setup_checks']),concurrent_original_oracle_calls=len(extra),
            exact_return_A_checks=sum(row['label']=='A2' and row['exact_output_newkv_return'] is True for row in originals),
            cross_arm_cache_byte_comparisons=sum(len(group['comparisons']) for case in report['cases'] for group in case['cache_cross_arm']),
            cross_arm_output_newkv_ABA_comparisons=sum(len(case['output_newkv_cross_arm']) for case in report['cases']),
            post_concurrent_isolation_checks=len(isolation),extra_candidate_replays=sum(row['action']=='replay_active' for row in isolation),
            diagnostic_calls=0,diagnostic_reference_replays=0,checked_warmup_measured_replays=0,
            cross_arm_output_newkv_timing_comparisons=0)
        assert report['counts']['original_oracle_calls'] == 39
        assert report['counts']['exact_return_A_checks'] == 12
        assert report['counts']['cross_arm_cache_byte_comparisons'] == 36
        report['correctness_complete'] = True
        gate()

        for case in DIAGNOSTIC_CASES:
            before = time.monotonic()
            control_runner = diagnostic_runners[case['name']]
            for version in (0,1):
                gate()
                own = control_runner.replay(version,False)
                if not own['passed']:
                    raise AssertionError('Uninstrumented diagnostic control did not reproduce own oracle bytes')
                report['counts']['diagnostic_reference_replays'] += 1
                inputs = control_runner.inputs[version]
                cfg = inputs['config']
                span = inputs['page_table'].shape[1]*cfg['page_size']
                splits = (span+8191)//8192
                count = cfg['batch']*cfg['num_kv_heads']*splits
                diag_inputs = helpers.clone_inputs(inputs)
                diag_inputs['alpha_stats'] = t.full((count,3),-1,dtype=t.int32,device='cuda')
                assert diag_inputs['alpha_stats'].is_contiguous() and diag_inputs['alpha_stats'].dtype == t.int32
                diag_work = helpers.clone_inputs(diag_inputs)
                graph,out = helpers.capture(diagnostic.kernel_fn,diag_work,diag_inputs)
                addresses = {key:value.data_ptr() for key,value in diag_work.items() if isinstance(value,t.Tensor)}
                out_address = out.data_ptr()
                helpers.restore(diag_work,diag_inputs)
                graph.replay()
                t.cuda.synchronize()
                plain_work = without_stats(diag_work)
                intact,why = helpers.unchanged(plain_work,inputs)
                stable = addresses == {key:value.data_ptr() for key,value in diag_work.items() if isinstance(value,t.Tensor)} and out.data_ptr()==out_address
                raw = ref.collect(plain_work,out)
                exact = byte_row('diagnostic_output_newkv',raw,control_runner.expected[version])
                caches = [byte_row(key,diag_work[key],control_runner.work[key]) for key in ['k_cache','v_cache']]
                actual = diag_work['alpha_stats'].cpu().tolist()
                row = dict(case=case['name'],input_version=version,stats_shape=[count,3],splits=splits,
                    actual_stats=actual,control_replay_check=own,output_newkv_comparison=exact,cache_comparisons=caches,
                    immutable_inputs_and_untouched_rows=intact,immutable_message=why,stable_addresses=stable,
                    finite_output=bool(t.isfinite(out).all()),designated_output='alpha_stats',
                    timed=False,original_oracle_called=False)
                report['diagnostics'].append(row)
                row['stats_validation'] = stats_check(actual,inputs['seq_lens'].cpu().tolist(),cfg['num_kv_heads'],splits,case['force'])
                row['passed'] = exact['exact'] and all(value['exact'] for value in caches) and intact and stable and row['finite_output']
                report['counts']['diagnostic_calls'] += 1
                if not row['passed']:
                    raise AssertionError('Diagnostic faithfulness/stats gate failed; timing forbidden')
                del diag_inputs,diag_work,plain_work,graph,out,inputs
                gate()
            phase(case['name']+'_untimed_diagnostic',before)
        assert report['counts']['diagnostic_calls'] == report['counts']['diagnostic_reference_replays'] == 10
        report['diagnostic_complete'] = True
        scored_fast = [next(row['stats_validation']['fast'] for row in report['diagnostics']
                           if row['case']==TARGET['name'] and row['input_version']==version) for version in (0,1)]
        report['frequency_gate'] = dict(opportunity=all(value>0 for value in scored_fast),scored_fast_counts=scored_fast,
            condition='Nonzero scored-family fast count in EACH fixture; no tuned threshold')
        if not report['frequency_gate']['opportunity']:
            report.update(status='complete',passed=True,outcome='no_scored_fast_opportunity',
                benefit=dict(admitted=False,reason='No fast-tile opportunity in at least one scored fixture'),
                timing_omitted='Predeclared scored opportunity gate failed; instrumented timing is never used')
            save()
            return 0
        # Release boundary/DP fixture graphs before uninstrumented timing.
        diagnostic_runners.clear()
        del control_runner
        gate()
        eviction = t.ones(256*1024*1024//4,device='cuda')
        for case in TIMED_CASES:
            before = time.monotonic()
            row = dict(case=case,warmups=[],samples=[])
            report['timings'].append(row)
            for index in range(WARMUP_SAMPLES+MEASURED_SAMPLES):
                order,version = order_and_version(index)
                sample = dict(index=index,order=order,input_version=version,raw_us={},checks={})
                (row['warmups'] if index<WARMUP_SAMPLES else row['samples']).append(sample)
                for name in order:
                    gate()
                    value,check = retained[case['name']][name].timed(version,eviction)
                    sample['raw_us'][name],sample['checks'][name] = value,check
                    report['counts']['checked_warmup_measured_replays'] += 1
                    if not check['passed']:
                        raise AssertionError('Uninstrumented timing replay did not reproduce exact own oracle bytes')
                left,right = [retained[case['name']][name] for name in NAMES]
                sample['cross_arm_output_newkv'] = byte_row('output_newkv',ref.collect(left.work,left.out),ref.collect(right.work,right.out))
                report['counts']['cross_arm_output_newkv_timing_comparisons'] += 1
                if not sample['cross_arm_output_newkv']['exact']:
                    raise AssertionError('Timed complete-call cross-arm output/newKV bytes differ')
            row['summary'] = timing_summary(row['samples'])
            phase(case['name']+'_uninstrumented_timing',before)
        assert report['counts']['checked_warmup_measured_replays'] == 64
        assert report['counts']['cross_arm_output_newkv_timing_comparisons'] == 32
        report['benefit'] = benefit_gate(report['timings'])
        report.update(status='complete',passed=True,timing_complete=True,outcome='timed')
        save()
        return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed',passed=False,error_type=type(exc).__name__,error=str(exc)[:1500])
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline',type=float,default=160.)
    raise SystemExit(main(parser.parse_args()))
