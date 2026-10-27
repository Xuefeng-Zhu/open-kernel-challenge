"""Bounded prospective V-stage whole-call validation; default is CPU preflight.

No service/credential/controller, profiler, install, production routing or official
entrypoint is changed. --execute requires independent prior native mechanism,
resource/lifetime review and the parent's serialized measurement admission.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
REPO = HERE.parents[3]
SOURCE_HASHES = {
    'kernel.py': '6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e',
    'config.py': '686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0',
    'fusion-lookahead-probe/candidate.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
}
SUPPORT_HASHES = {
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
LICENSE_HASHES = {
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
INSTALLED_HASHES = {
    '/opt/sglang/lib/python3.12/site-packages/triton/experimental/gluon/__init__.py': 'd7b607f028fed01715550c5e5ef708b0bd71881ccda220fb7390248ec29ca904',
    '/opt/sglang/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/hopper/__init__.py': 'c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc',
    '/opt/sglang/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/ampere/async_copy.py': '86e23dfd1231d2709e93a126c91349254440a930ef2196c6d5e3fcefc7e8f623',
}
NAMES = ['fa3', 'v_after_wait', 'v_before_wait']
SEEDS = [135751, 135752]
PT_STRIDE = 86008
BOUNDARY_UNIT = dict(name='B32_L8_short_tail_unit', batch=32, q_len=8, ctx=129,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='fixed', dist='soft', boundary=True)
BOUNDARY_SCALED = dict(BOUNDARY_UNIT, name='B32_L8_short_tail_scale_1_64', kv_scale=1/64)
TARGET = dict(name='B32_L8_ctx4096_coherent_unit', batch=32, q_len=8, ctx=4096,
    heads='tp8', layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent', boundary=False)
CASES = [BOUNDARY_UNIT, BOUNDARY_SCALED, TARGET]
WARMUP_SAMPLES, MEASURED_SAMPLES = 4, 12
MAX_REPORT = 500000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_gate(probe=PROBE, support=V2, repo=REPO):
    observed = {}
    paths = {name:probe/name for name in ('kernel.py','config.py')}
    paths['fusion-lookahead-probe/candidate.py'] = support/'fusion-lookahead-probe/candidate.py'
    paths.update({name:support/name for name in SUPPORT_HASHES})
    paths.update({name:repo/name for name in LICENSE_HASHES})
    expected = {**SOURCE_HASHES, **SUPPORT_HASHES, **LICENSE_HASHES}
    for name,path in paths.items():
        if path.is_symlink() or digest(path.read_bytes()) != expected[name]:
            raise ValueError('Frozen source/support/license mismatch: '+name)
        if path.suffix == '.py':
            compile(path.read_bytes(), name, 'exec')
        observed[name] = expected[name]
    return observed


def installed_gate():
    if importlib.metadata.version('triton') != '3.8.0':
        raise ValueError('Pinned installed Triton3.8.0 required')
    for name,expected in INSTALLED_HASHES.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024*1024 or digest(path.read_bytes()) != expected:
            raise ValueError('Pinned installed Gluon bytes changed: '+name)
    return dict(triton='3.8.0', source_sha256=INSTALLED_HASHES)


def boundary_lengths(version):
    if version not in (0,1):
        raise ValueError('Only declared A/B versions')
    values = [0,1,55,56,63,64,65,129]*4
    return values if version == 0 else list(reversed(values))


def padded_row(row, width=PT_STRIDE):
    """CPU structural model; GPU fixture uses the same valid prefix/-1 padding."""
    if len(row) > width or any(type(value) is not int or value < -1 for value in row):
        raise ValueError('Invalid live page1 prefix or padding width')
    return list(row)+[-1]*(width-len(row))


def order_and_version(index):
    if type(index) is not int or index < 0:
        raise ValueError('Nonnegative sample index required')
    pair = index//2
    shift = pair%len(NAMES)
    order = NAMES[shift:]+NAMES[:shift]
    return (order if index%2 == 0 else list(reversed(order))), pair%2


def timing_summary(samples):
    if len(samples) != MEASURED_SAMPLES:
        raise ValueError('Twelve measured three-arm rounds required')
    for offset,row in enumerate(samples):
        index = offset+WARMUP_SAMPLES
        order,version = order_and_version(index)
        if row['index'] != index or row['order'] != order or row['input_version'] != version:
            raise ValueError('Declared order/fixture plan changed')
        if set(row['raw_us']) != set(NAMES) or not all(math.isfinite(row['raw_us'][name]) and row['raw_us'][name] > 0 for name in NAMES):
            raise ValueError('Nonfinite/nonpositive/missing complete-call timings')
    comparisons = {}
    for control in ['v_after_wait','fa3']:
        ratios = [row['raw_us'][control]/row['raw_us']['v_before_wait'] for row in samples]
        balanced = [math.sqrt(ratios[index]*ratios[index+1]) for index in range(0,len(ratios),2)]
        versions = [samples[index]['input_version'] for index in range(0,len(samples),2)]
        per_fixture = {}
        for version in (0,1):
            values = [value for value,fixture in zip(balanced,versions) if fixture == version]
            if len(values) != 3:
                raise ValueError('Three reverse-order pairs per seed required')
            median = statistics.median(values)
            per_fixture[str(version)] = dict(pair_ratios=values, median_ratio=median,
                latency_reduction_percent=100*(1-1/median), positive_pairs=sum(value>1 for value in values))
        comparisons['v_before_wait_vs_'+control] = dict(reverse_order_pair_geomeans=balanced,
            pair_input_versions=versions, positive_reverse_order_pairs=sum(value>1 for value in balanced),
            per_fixture=per_fixture)
    return dict(raw_median_us={name:statistics.median(row['raw_us'][name] for row in samples) for name in NAMES},
        comparisons=comparisons,
        qualification='Raw whole-call events; six reverse-order pairs, three per fresh seed; nearby ctx4096 padded geometry only; no score extrapolation or confidence interval')


def benefit_gate(samples):
    summary = timing_summary(samples)
    def acceptable(name):
        row = summary['comparisons'][name]
        return row['positive_reverse_order_pairs'] == 6 and all(value['latency_reduction_percent'] >= 2. for value in row['per_fixture'].values())
    mechanism = acceptable('v_before_wait_vs_v_after_wait')
    actual_fa3 = acceptable('v_before_wait_vs_fa3')
    return dict(mechanism_passed=mechanism, actual_fa3_passed=actual_fa3,
        production_replacement_admitted=False, official_check_admitted=False,
        prospective_followup_supported=mechanism and actual_fa3,
        predeclared_gate='Each comparison requires >=2% median reduction in EACH fresh seed and all six favorable reverse-order pairs; actual FA3 win is independently required before any replacement proposal',
        summary=summary,
        qualification='Passing this small nearby diagnostic is insufficient for production promotion, score prediction or official admission')


def main(args):
    started = time.monotonic()
    report = dict(kind='V-stage prototype adapters versus unchanged actual working FA3 whole call',
        started_utc=datetime.now(timezone.utc).isoformat(), status='starting', passed=False,
        source_sha256=SOURCE_HASHES, support_sha256=SUPPORT_HASHES,
        worker_sha256=digest(Path(__file__).read_bytes()), correctness_complete=False, timing_complete=False,
        cases=[], independent_setup_checks=[], concurrent_submissions=[], timings=[], counts={}, phases=[],
        benefit=None, official_check_admitted=False, production_replacement_admitted=False,
        protocol=dict(seeds=SEEDS, cases=CASES, arms=NAMES, page_table_stride=PT_STRIDE,
            original_oracle_and_tolerances=True, no_production_dispatch_change=True,
            prerequisite='Independent current-source native contrast/resource/lifetime review PASS; external parent/controller admission, not inferred by this worker',
            prototype_call='Working fused FP8 Q/new-KV preparation, Q BF16 widening, unmodified raw Gluon core, same output RMS/gate; all inside graph/call',
            actual_baseline='Exact source3da kernel_fn, full B32L8 FA3 route and unchanged automatic scheduler/splits',
            precision='Both prototypes preserve BF16->E4M3->BF16 Q; E4M3 K/V widened BF16, strict BF16 WGMMA, FP32 online state, BF16 P/PV, no original tolerance change',
            graph_replay='A1/B/A2 fresh QKV/mapping/cache/length metadata at stable addresses; private captures, disjoint writable caches and outputs',
            byte_comparisons='Exact own replay and prototype pair output/newKV; all-three newKV and whole K/V exact; FA3 output only unchanged original oracle',
            concurrency='Two independent v_before_wait graphs/streams; actual hardware overlap and internal scratch pointers unobserved',
            timing='Raw cold-L2 whole-call graph events, 4 total warmup +12 total measured rounds across two already oracle-checked fixtures, 3 arms/round',
            cold_l2_eviction_bytes=256*1024*1024, warmup_samples=WARMUP_SAMPLES, measured_samples=MEASURED_SAMPLES,
            excluded_from_timing='Fixture generation/padding/restores/eviction/validation; all preparation/cast/cache append/core/gate remain timed',
            performance_gate='Before-vs-after and before-vs-actualFA3 separately require >=2% EACH seed and all6 favorable pairs',
            profiler_used=False, official_score_predicted=False, max_nominal_context=4096))
    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        raw = (json.dumps(report,separators=(',',':'))+'\n').encode()
        if len(raw) > MAX_REPORT:
            raise ValueError('Result exceeds bounded500KB transport')
        temp = args.output.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(args.output)
    def gate():
        save()
        if time.monotonic()-started >= args.deadline:
            raise TimeoutError('V-stage runtime deadline')
    def alarm(*_):
        raise TimeoutError('V-stage runtime alarm')
    def phase(name,before):
        report['phases'].append(dict(name=name,wall_seconds=time.monotonic()-before)); gate()
    if not 0 < args.deadline <= 160:
        raise ValueError('Deadline must be positive and <=160s')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(math.ceil(args.deadline)); save()
    try:
        report['verified_hashes'] = source_gate()
        report['installed_source_gate'] = installed_gate()
        if __import__('os').environ.get('TRITON_INTERPRET'):
            raise ValueError('Native H200 required')
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name() or triton.__version__ != '3.8.0':
            raise ValueError('Pinned native H200/Triton3.8.0 required')
        report['environment'] = dict(torch=t.__version__,triton=triton.__version__,cuda=t.version.cuda,
            gpu=t.cuda.get_device_name(),capability=t.cuda.get_device_capability(),python=platform.python_version())
        sys.path.insert(0,str(V2/'official'))
        helpers = load(V2/'modal_worker.py','vstage_runtime_helpers')
        ref = load(V2/'official/reference.py','vstage_runtime_reference')
        working = load(V2/'fusion-lookahead-probe/candidate.py','vstage_runtime_working3da')
        kernel = load(PROBE/'kernel.py','vstage_runtime_immutable_core')
        config = load(PROBE/'config.py','vstage_runtime_fixed_config').ProbeConfig()

        def prototype(arm):
            options = config.compile_kwargs(arm)
            def complete_call(data):
                cfg = data['config']
                batch,length,hq,hkv,d = [cfg[key] for key in ['batch','q_len','num_q_heads','num_kv_heads','head_dim']]
                qkv = data['qkv'].view(batch*length,-1)
                q = t.empty((batch*length,hq,d),device=qkv.device,dtype=t.float8_e4m3fn)
                kc,vc = data['k_cache'],data['v_cache']
                working._fa3_prepare_fused[(triton.cdiv(batch*length*(hq+2*hkv),4),)](
                    qkv,data['positions'],qkv,data['cache_loc'],kc,vc,data['k_scale'],data['v_scale'],q,
                    qkv.stride(0),kc.numel()//(hkv*d),working.RMS_EPS,
                    T=batch*length,HQ=hq,HKV=hkv,D=d,BD=64,BM=4,ROPE=False,SCALED=cfg['kv_scale']!=1.,num_warps=4)
                q_bf16 = q.to(t.bfloat16).view(batch,length,hq,d)
                raw_out = t.empty((batch,length,hq,d),device=qkv.device,dtype=t.bfloat16)
                kernel.attention_vstage_kernel[config.grid](q_bf16,kc.view(-1,hkv,d),vc.view(-1,hkv,d),
                    data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],raw_out,**options)
                gate_tensor = qkv[:,(hq+2*hkv)*d:]
                out = working.rms_norm_gate(raw_out.reshape(batch*length,hq*d),gate_tensor,None,
                    head_dim=d,eps=working.RMS_EPS,has_norm=True)
                return out.view(batch,length,hq*d)
            return complete_call
        functions = dict(fa3=working.kernel_fn,v_after_wait=prototype('v_after_wait'),v_before_wait=prototype('v_before_wait'))
        phase('imports_source_binding',started)

        def byte_row(name,left,right):
            a,b = [value.contiguous().reshape(-1).view(t.uint8) for value in [left,right]]
            same = a.shape == b.shape
            return dict(name=name,bytes=a.numel(),exact=same and bool(t.equal(a,b)),
                unequal_byte_count=int((a!=b).sum()) if same else None)

        def fresh(case,version):
            data = ref.generate_input(**{key:value for key,value in case.items() if key not in {'name','boundary'}},
                seed=SEEDS[version],device='cuda')
            original_width = data['page_table'].shape[1]
            if case['boundary']:
                values = boundary_lengths(version)
                lengths = t.tensor(values,dtype=t.int32,device='cuda')
                table = data['page_table']
                zero_request = values.index(0)
                flat = table.reshape(-1)
                zero_at = int((flat==0).nonzero()[0])
                destination = zero_request*table.shape[1]
                previous = flat[destination].clone(); flat[destination]=0; flat[zero_at]=previous
                data['seq_lens'].copy_(lengths)
                data['positions'].copy_((lengths.long()[:,None]+t.arange(8,device='cuda')).reshape(-1))
                data['cache_seqlens'].copy_(lengths+8)
                data['cache_loc'].copy_(ref.new_slots(data).reshape(-1))
                for b,seq in enumerate(values):
                    table[b,seq+8:] = -1
                assert int(data['cache_loc'][zero_request*8]) == 0
            table = data['page_table']
            if table.dtype != t.int32 or data['seq_lens'].dtype != t.int32 or original_width > PT_STRIDE:
                raise ValueError('Compile-matched int32 metadata and bounded page1 prefix required')
            padded = t.full((32,PT_STRIDE),-1,dtype=t.int32,device='cuda')
            padded[:,:original_width].copy_(table)
            data['page_table'] = padded
            assert bool((padded[:,original_width:] == -1).all())
            capacity = data['k_cache'].numel()//(2*64)
            for b,seq in enumerate(data['seq_lens'].cpu().tolist()):
                live = padded[b,:seq+8]
                assert live.numel() == seq+8 and bool(((live>=0)&(live<capacity)).all())
                assert bool((padded[b,seq+8:] == -1).all())
            assert t.equal(data['cache_loc'],ref.new_slots(data).reshape(-1))
            assert data['scheduler_metadata'] is None
            assert (data['config']['batch'],data['config']['q_len'],data['config']['num_q_heads'],
                data['config']['num_kv_heads'],data['config']['head_dim'],data['config']['page_size'],
                data['config']['window'],data['config']['rope_theta']) == (32,8,16,2,64,1,0,0.)
            assert data['config']['sm_scale'] == config.sm_scale and data['config']['qk_scale_factor'] == config.qk_factor
            return data,dict(original_page_table_width=original_width,padded_page_table_shape=[32,PT_STRIDE],
                padding_value=-1,valid_prefix_preserved=True,cache_loc_matches_live_mapping=True,
                physical_cache_rows=capacity,all_live_page_entries_in_physical_range=True,all_unused_suffix_entries_negative_one=True,
                seq_lens_dtype=str(data['seq_lens'].dtype),page_table_dtype=str(padded.dtype))

        class Runner:
            def __init__(self,name,inputs):
                self.name,self.inputs = name,inputs
                self.work = helpers.clone_inputs(inputs[0])
                self.graph,self.out = helpers.capture(functions[name],self.work,inputs[0])
                self.addresses = {key:value.data_ptr() for key,value in self.work.items() if isinstance(value,t.Tensor)}
                self.out_address = self.out.data_ptr()
                pool = getattr(self.graph,'pool',None)
                self.pool = tuple(pool()) if callable(pool) else None
                self.expected = {}
            def evidence(self):
                return dict(output_address=self.out_address,k_cache_address=self.work['k_cache'].data_ptr(),
                    v_cache_address=self.work['v_cache'].data_ptr(),graph_private_pool=None if self.pool is None else list(self.pool),
                    internal_scratch_pointers_observed=False)
            def check(self,version,oracle):
                raw = ref.collect(self.work,self.out)
                stable = self.addresses == {key:value.data_ptr() for key,value in self.work.items() if isinstance(value,t.Tensor)} and self.out_address == self.out.data_ptr()
                intact,why = helpers.unchanged(self.work,self.inputs[version])
                exact = byte_row('own_output_newkv',raw,self.expected[version])['exact'] if version in self.expected else None
                finite = bool(t.isfinite(self.out).all())
                if oracle:
                    okay,message = ref.check_implementation(self.work,raw)
                    okay = okay and exact is not False
                    self.expected[version] = raw.clone()
                else:
                    okay,message = exact is True,'Exact own previously original-oracle-checked bytes'
                return dict(passed=bool(okay and stable and intact and finite),stable_addresses=stable,
                    immutable_inputs_and_untouched_rows=intact,immutable_message=why,finite_output=finite,
                    official_oracle_called=oracle,oracle_message=message,exact_output_newkv_return=exact)
            def replay(self,version,oracle=True):
                helpers.restore(self.work,self.inputs[version]); self.graph.replay(); t.cuda.synchronize()
                return self.check(version,oracle)
            def timed(self,version,eviction):
                helpers.restore(self.work,self.inputs[version])
                value = helpers.timed(self.graph,eviction)
                return value,self.check(version,False)

        retained = None
        for case in CASES:
            before = time.monotonic(); gate()
            pairs = [fresh(case,version) for version in (0,1)]
            inputs = [pair[0] for pair in pairs]
            for key in ['seq_lens','qkv','page_table','k_cache','v_cache']:
                assert not byte_row(key,inputs[0][key],inputs[1][key])['exact']
            runners = {name:Runner(name,inputs) for name in NAMES}
            item = dict(case=case,config=dict(inputs[0]['config']),input_lengths=[value['seq_lens'].cpu().tolist() for value in inputs],
                padding_evidence=[pair[1] for pair in pairs],capture_evidence={name:value.evidence() for name,value in runners.items()},
                records=[],cache_cross_arm=[],prototype_output_cross_arm=[],newkv_cross_arm=[])
            report['cases'].append(item)
            assert len({value.out_address for value in runners.values()}) == 3
            known_pools = [value.pool for value in runners.values() if value.pool is not None]
            assert len(set(known_pools)) == len(known_pools)
            for key in ['k_cache','v_cache']:
                assert len({value.work[key].data_ptr() for value in runners.values()}) == 3
            for label,version in [('A1',0),('B',1),('A2',0)]:
                for name in NAMES:
                    gate(); row=dict(runner=name,label=label,input_version=version,**runners[name].replay(version))
                    item['records'].append(row)
                    if not row['passed']:
                        raise AssertionError('Original oracle/ABA/immutability gate failed; timing forbidden')
                for name in NAMES[1:]:
                    comparisons = [byte_row(key,runners['fa3'].work[key],runners[name].work[key]) for key in ['k_cache','v_cache']]
                    item['cache_cross_arm'].append(dict(label=label,input_version=version,other=name,comparisons=comparisons))
                    baseline_raw = ref.collect(runners['fa3'].work,runners['fa3'].out)
                    prototype_raw = ref.collect(runners[name].work,runners[name].out)
                    output_bytes = runners['fa3'].out.numel()*runners['fa3'].out.element_size()
                    newkv = byte_row('newKV',baseline_raw[output_bytes:],prototype_raw[output_bytes:])
                    item['newkv_cross_arm'].append(dict(label=label,other=name,**newkv))
                    if not all(value['exact'] for value in comparisons) or not newkv['exact']:
                        raise AssertionError('All-arm whole K/V and newKV byte gate failed')
                comparison = byte_row('prototype_output_newkv',ref.collect(runners['v_after_wait'].work,runners['v_after_wait'].out),
                    ref.collect(runners['v_before_wait'].work,runners['v_before_wait'].out))
                item['prototype_output_cross_arm'].append(dict(label=label,input_version=version,**comparison))
                if not comparison['exact']:
                    raise AssertionError('Prototype-pair output/newKV bytes differ')
            if case is BOUNDARY_UNIT:
                first = runners['v_before_wait']; second = Runner('v_before_wait',inputs)
                assert first.out_address != second.out_address
                if first.pool is not None and second.pool is not None:
                    assert first.pool != second.pool
                for key in ['k_cache','v_cache']:
                    assert first.work[key].data_ptr() != second.work[key].data_ptr()
                setup = dict(input_version=0,**second.replay(0,True)); report['independent_setup_checks'].append(setup)
                if not setup['passed']:
                    raise AssertionError('Independent own-A setup oracle failed')
                helpers.restore(first.work,inputs[0]); helpers.restore(second.work,inputs[1])
                streams = [t.cuda.Stream(),t.cuda.Stream()]; ready=t.cuda.Event(); ready.record()
                for stream,runner in zip(streams,[first,second]):
                    stream.wait_event(ready)
                    with t.cuda.stream(stream): runner.graph.replay()
                for stream in streams: t.cuda.current_stream().wait_stream(stream)
                t.cuda.synchronize()
                concurrent = dict(checks=[first.check(0,True),second.check(1,True)],isolation_checks=[],
                    capture_evidence=[first.evidence(),second.evidence()],actual_hardware_overlap_measured=False,
                    internal_scratch_pointers_observed=False,disjoint_outputs_and_writable_caches=True)
                report['concurrent_submissions'].append(concurrent)
                for active,version,inactive,retained_version in [(first,1,second,1),(second,0,first,1)]:
                    gate()
                    concurrent['isolation_checks'].append(dict(action='replay_active',**active.replay(version,False)))
                    concurrent['isolation_checks'].append(dict(action='check_inactive_without_replay',**inactive.check(retained_version,False)))
                if not all(row['passed'] for row in concurrent['checks']+concurrent['isolation_checks']):
                    raise AssertionError('Independent graph submission/reuse isolation failed')
                concurrent['passed']=True
                del first,second,streams,active,inactive,runner
            item['passed']=True
            if case is TARGET: retained=runners
            del runners,inputs,pairs
            phase(case['name']+'_correctness',before)
        originals = [row for case in report['cases'] for row in case['records']]
        extra = [row for group in report['concurrent_submissions'] for row in group['checks']]
        isolation = [row for group in report['concurrent_submissions'] for row in group['isolation_checks']]
        report['counts'] = dict(original_oracle_calls=sum(row['official_oracle_called'] for row in originals+extra+report['independent_setup_checks']),
            independent_setup_oracle_calls=len(report['independent_setup_checks']),concurrent_original_oracle_calls=len(extra),
            exact_return_A_checks=sum(row['label']=='A2' and row['exact_output_newkv_return'] is True for row in originals),
            cross_arm_cache_byte_comparisons=sum(len(group['comparisons']) for case in report['cases'] for group in case['cache_cross_arm']),
            prototype_output_newkv_ABA_comparisons=sum(len(case['prototype_output_cross_arm']) for case in report['cases']),
            cross_arm_newkv_ABA_comparisons=sum(len(case['newkv_cross_arm']) for case in report['cases']),
            post_concurrent_isolation_checks=len(isolation),extra_candidate_replays=sum(row['action']=='replay_active' for row in isolation),
            checked_warmup_measured_replays=0,prototype_output_newkv_timing_comparisons=0,cross_arm_newkv_timing_comparisons=0)
        for key,expected in dict(original_oracle_calls=30,exact_return_A_checks=9,cross_arm_cache_byte_comparisons=36,
                prototype_output_newkv_ABA_comparisons=9,cross_arm_newkv_ABA_comparisons=18,
                post_concurrent_isolation_checks=4,extra_candidate_replays=2).items():
            assert report['counts'][key] == expected
        report['correctness_complete']=True; gate()
        eviction=t.ones(256*1024*1024//4,device='cuda')
        timing=dict(case=TARGET,warmups=[],samples=[]); report['timings'].append(timing)
        before=time.monotonic()
        for index in range(WARMUP_SAMPLES+MEASURED_SAMPLES):
            order,version=order_and_version(index)
            sample=dict(index=index,order=order,input_version=version,raw_us={},checks={},newkv_cross_arm=[])
            (timing['warmups'] if index<WARMUP_SAMPLES else timing['samples']).append(sample)
            for name in order:
                gate(); value,check=retained[name].timed(version,eviction)
                sample['raw_us'][name],sample['checks'][name]=value,check
                report['counts']['checked_warmup_measured_replays']+=1
                if not check['passed']:
                    raise AssertionError('Timing replay failed exact own original-oracle bytes')
            raw={name:ref.collect(retained[name].work,retained[name].out) for name in NAMES}
            pair=byte_row('prototype_output_newkv',raw['v_after_wait'],raw['v_before_wait'])
            sample['prototype_pair']=pair; report['counts']['prototype_output_newkv_timing_comparisons']+=1
            if not pair['exact']: raise AssertionError('Timed prototype pair bytes differ')
            output_bytes=retained['fa3'].out.numel()*retained['fa3'].out.element_size()
            for name in NAMES[1:]:
                comparison=byte_row('newKV',raw['fa3'][output_bytes:],raw[name][output_bytes:])
                sample['newkv_cross_arm'].append(dict(other=name,**comparison))
                report['counts']['cross_arm_newkv_timing_comparisons']+=1
                if not comparison['exact']: raise AssertionError('Timed all-arm newKV bytes differ')
        assert report['counts']['checked_warmup_measured_replays']==48
        assert report['counts']['prototype_output_newkv_timing_comparisons']==16
        assert report['counts']['cross_arm_newkv_timing_comparisons']==32
        report['benefit']=benefit_gate(timing['samples'])
        report.update(status='complete',passed=True,timing_complete=True,outcome='timed')
        phase('uninstrumented_three_arm_wholecall_timing',before); save(); return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed',passed=False,error_type=type(exc).__name__,error=str(exc)[:1500])
        save(); return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Execute only after independent native gates and parent serialization admission')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--deadline',type=float,default=160.)
    args=parser.parse_args()
    if args.execute:
        if args.output is None: parser.error('--execute requires --output')
        raise SystemExit(main(args))
    print(json.dumps(dict(status='offline-preflight',verified_hashes=source_gate(),native_execution=False,official_check_admitted=False),indent=2))
