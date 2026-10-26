"""Bounded physical-page-locality diagnostic; default is CPU source preflight.

No service, credentials, GPU request, installation, counters or candidate edit.
Only separately admitted --execute loads Torch and executes the fixed source.
"""
import argparse
from array import array
import base64
from datetime import datetime, timezone
import gzip
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
V2 = HERE.parent
SOURCE_HASHES = {
    'fusion-lookahead-probe/candidate.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
    'locality-probe/fixture.py': '203240b0f4de073b4c6222c62daeeaff20ce0b3de32b519f8d9f59c5528e0e28',
}
SEEDS = (135831, 135832)
ARMS = ('random_pages', 'contiguous_pages')
CASES = (
    dict(name='B1_L1_sliding', batch=1, q_len=1, ctx=80000, heads='tp8',
         layer='sliding', page=1, kv_scale=1., spread='fixed', dist='soft'),
    dict(name='B16_L8_full', batch=16, q_len=8, ctx=80000, heads='tp8',
         layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent'),
    dict(name='B32_L1_full', batch=32, q_len=1, ctx=80000, heads='tp8',
         layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent'),
)
WARMUP_ROUNDS, REVERSE_PAIRS = 4, 6
MAX_SECONDS = 160
MAX_RAW_TRACE = 8 * 1024 * 1024
MAX_TRANSPORT = 2 * 1024 * 1024
EVICTION_BYTES = 256 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def source_gate(root=V2):
    observed = {}
    for name, expected in SOURCE_HASHES.items():
        p = Path(root)/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 256*1024:
            raise ValueError('Frozen source path guard: ' + name)
        raw = p.read_bytes()
        if digest(raw) != expected:
            raise ValueError('Frozen source hash mismatch: ' + name)
        compile(raw, name, 'exec')
        observed[name] = expected
    return observed


def measured_order(index):
    if type(index) is not int or not 0 <= index < 2*REVERSE_PAIRS:
        raise ValueError('Exactly twelve measured rounds are declared')
    pair, half = divmod(index, 2)
    return list(ARMS if (pair+half) % 2 == 0 else reversed(ARMS))


def summarize_samples(samples):
    if len(samples) != 2*REVERSE_PAIRS:
        raise ValueError('Six complete reverse-order pairs required')
    for index, row in enumerate(samples):
        if row.get('index') != index or row.get('order') != measured_order(index):
            raise ValueError('Measured order changed')
        values = row.get('raw_us', {})
        if set(values) != set(ARMS) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError('Missing, nonfinite or nonpositive raw event time')
    ratios = [row['raw_us'][ARMS[0]]/row['raw_us'][ARMS[1]] for row in samples]
    pairs = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, len(ratios), 2)]
    median = statistics.median(pairs)
    return dict(raw_median_us={name:statistics.median(row['raw_us'][name] for row in samples) for name in ARMS},
        reverse_order_pair_ratios=pairs, median_pair_ratio_random_over_contiguous=median,
        contiguous_latency_reduction_percent=100*(1-1/median),
        favorable_contiguous_pairs=sum(value > 1 for value in pairs),
        qualification='Same graph/source and same logical fixture; raw whole-call events, no replay-cost subtraction, no score or hardware-bottleneck inference')


def locality_contrast(plan, layer):
    """Prove actual attended/history page movement, not just a changed unused tail."""
    if layer not in ('sliding','full'):raise ValueError('Unknown locality case layer')
    moved=moved_history=attended=edges=old_contiguous=new_contiguous=0
    h_a,h_b=hashlib.sha256(),hashlib.sha256()
    for row_a,row_b,total in zip(plan.page_table_a,plan.page_table_b,plan.cache_seqlens):
        for row,d in ((row_a,h_a),(row_b,h_b)):
            values=array('i',row)
            if values.itemsize!=4:raise ValueError('int32 page-table encoding required')
            d.update(values.tobytes())
        start=max(0,total-2048) if layer=='sliding' else 0
        for j in range(start,total):
            changed=row_a[j]!=row_b[j]
            moved+=changed;moved_history+=changed and j<total-plan.q_len;attended+=1
            if j>start:
                edges+=1;old_contiguous+=row_a[j]==row_a[j-1]+1
                new_contiguous+=row_b[j]==row_b[j-1]+1
    if moved_history==0 or new_contiguous!=edges or old_contiguous>=new_contiguous:
        raise ValueError('No proven attended historical locality contrast')
    return dict(attended_pages=attended,moved_attended_pages=moved,
        moved_attended_history_pages=moved_history,adjacent_attended_pairs=edges,
        random_layout_contiguous_adjacent_pairs=old_contiguous,
        canonical_layout_contiguous_adjacent_pairs=new_contiguous,
        page_table_A_sha256=h_a.hexdigest(),page_table_B_sha256=h_b.hexdigest(),
        page_table_hash_encoding='native-endian int32; '+sys.byteorder,
        qualification='Actual fixture address-map contrast, not measured HBM transactions or hardware locality')


def expected_counts(activity=False):
    fixtures = len(CASES)*len(SEEDS)
    oracle = fixtures*len(ARMS)
    warmup = fixtures*WARMUP_ROUNDS*len(ARMS)
    measured = fixtures*2*REVERSE_PAIRS*len(ARMS)
    profiled = len(CASES)*len(ARMS) if activity else 0
    return dict(fixtures=fixtures, original_oracle_calls=oracle, warmup_checked_replays=warmup,
        measured_checked_replays=measured, activity_checked_replays=profiled,
        total_checked_graph_replays=oracle+warmup+measured+profiled,
        original_oracle_calls_attempted=oracle,graph_replays_attempted=oracle+warmup+measured+profiled,
        graph_replays_completed=oracle+warmup+measured+profiled,
        capture_warmup_function_calls_excluded=len(CASES)*3,
        capture_function_calls_excluded=len(CASES))


def parse_activity(trace, labels):
    """Fenced CPU ranges contain actual CUDA kernel events; no residency estimate."""
    events = trace.get('traceEvents', [])
    groups, total_symbols = {}, 0
    for label in labels:
        ranges = [e for e in events if e.get('name') == label and e.get('ph') == 'X'
                  and e.get('cat') == 'user_annotation']
        if len(ranges) != 1:
            raise ValueError('Exactly one fenced range required: ' + label)
        start, duration = ranges[0].get('ts'), ranges[0].get('dur')
        if not isinstance(start, (int,float)) or not isinstance(duration, (int,float)) or duration <= 0:
            raise ValueError('Invalid activity range')
        kernels = []
        for event in events:
            if event.get('cat') != 'kernel' or event.get('ph') != 'X':
                continue
            ts, dur = event.get('ts'), event.get('dur')
            if not isinstance(ts, (int,float)) or not isinstance(dur, (int,float)):
                continue
            if start <= ts and ts+dur <= start+duration+1e-6:
                name = event.get('name')
                if not isinstance(name, str) or not name.strip(): raise ValueError('Missing native symbol')
                total_symbols += len(name.encode())
                if len(kernels) >= 64 or total_symbols > 256*1024:
                    raise ValueError('Activity symbol bound; no truncation')
                args = event.get('args') or {}
                kernels.append(dict(name=name, symbol_sha256=digest(name.encode()),
                    instrumented_duration_us=dur,
                    launch={key:args.get(key) for key in ['grid','block','registers per thread','shared memory','graph id','graph node id','stream']}))
        if not kernels: raise ValueError('No fenced CUDA kernel evidence: ' + label)
        groups[label] = dict(kernels=kernels, kernel_count=len(kernels))
    return dict(groups=groups, full_symbol_utf8_bytes=total_symbols,
        qualification='Instrumented fenced activities and static launch fields; not benchmark times, CTA residency, DRAM bytes or stall counters')


def trace_artifact(raw):
    if len(raw) > MAX_RAW_TRACE: raise ValueError('Raw activity trace exceeds8MiB; no truncation')
    compressed = gzip.compress(raw, mtime=0)
    payload = base64.b64encode(compressed).decode('ascii')
    if len(payload.encode()) > MAX_TRANSPORT:
        raise ValueError('Compressed activity transport exceeds2MiB; no truncation')
    return dict(name='activity-trace.json', raw_bytes=len(raw), raw_sha256=digest(raw),
        encoding='gzip+base64', compressed_sha256=digest(compressed), data_b64=payload)


def native_identity_comparison(left, right):
    fields=('grid','block','registers per thread','shared memory','graph id','graph node id')
    missing=[]
    for arm,rows in zip(ARMS,(left,right)):
        for index,row in enumerate(rows):
            for field in fields:
                value=row.get('launch',{}).get(field)
                valid=(isinstance(value,(list,tuple)) and len(value)==3
                       and all(type(i) is int and i>0 for i in value)) if field in ('grid','block') else (
                       type(value) is int and value >= (1 if field=='registers per thread' else 0))
                if not valid:
                    missing.append(dict(arm=arm,kernel_index=index,field=field,reason='missing_or_invalid'))
    if missing:
        return dict(status='UNKNOWN',same_symbol_and_launch_fields=None,missing_fields=missing,
                    qualification='Missing launch metadata does not prove resource/geometry parity')
    def identity(rows):
        # PDL/parallel event start order is not node identity; compare multisets.
        return sorted(encoded(dict(symbol=r['symbol_sha256'],
            launch={key:r['launch'][key] for key in fields})).decode() for r in rows)
    same=identity(left)==identity(right)
    return dict(status='observed_equal' if same else 'observed_different',same_symbol_and_launch_fields=same,
                qualification='Full symbols and reported node/launch metadata, not instruction-level disassembly')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_inputs(t, data):
    return {k:v.clone() if isinstance(v,t.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in data.items()}


def tensor_state(t, data):
    return {k:dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype),
                   device=str(v.device), pointer=v.data_ptr()) for k,v in data.items() if isinstance(v,t.Tensor)}


def restore(t, work, fresh):
    if work.keys() != fresh.keys(): raise ValueError('Input keys changed')
    for key, value in fresh.items():
        if isinstance(value,t.Tensor):
            dest = work[key]
            if dest.shape != value.shape or dest.dtype != value.dtype or dest.stride() != value.stride():
                raise ValueError('Graph tensor layout changed: ' + key)
            dest.copy_(value)
        elif work[key] != value: raise ValueError('Config/None changed: ' + key)


def byte_equal(t, a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(t.equal(
        a.contiguous().reshape(-1).view(t.uint8), b.contiguous().reshape(-1).view(t.uint8)))


def ownership(t, work, fresh):
    for key, value in fresh.items():
        if key in ('k_cache','v_cache'): continue
        if isinstance(value,t.Tensor):
            if not byte_equal(t, work[key], value): raise ValueError('Immutable input changed: ' + key)
        elif work[key] != value: raise ValueError('Immutable metadata changed: ' + key)
    locations = fresh['cache_loc'].long()
    hkv, d = fresh['config']['num_kv_heads'], fresh['config']['head_dim']
    for key in ('k_cache','v_cache'):
        got, original = [x.view(-1,hkv*d).view(t.uint8) for x in (work[key],fresh[key])]
        saved = got[locations].clone()
        try:
            got[locations] = original[locations]
            if not bool(t.equal(got,original)): raise ValueError('Historical/unused cache row changed: ' + key)
        finally: got[locations] = saved


def new_rows(t, work):
    cfg = work['config']; loc = work['cache_loc'].long()
    return {key:work[key].view(-1,cfg['num_kv_heads']*cfg['head_dim']).view(t.uint8).index_select(0,loc).clone()
            for key in ('k_cache','v_cache')}


def canonical_fixture(t, ref, helper, case, seed):
    a = ref.generate_input(**{k:v for k,v in case.items() if k != 'name'}, seed=seed,device='cuda')
    if a['config']['page_size'] != 1: raise ValueError('Page1 only')
    plan = helper.build_plan(a['page_table'].cpu().tolist(),a['cache_seqlens'].cpu().tolist(),
        a['cache_loc'].cpu().tolist(),case['q_len'],a['k_cache'].shape[0])
    b = clone_inputs(t,a)
    nto = t.frombuffer(plan.new_to_old,dtype=t.int64).to(device='cuda')
    otn = t.frombuffer(plan.old_to_new,dtype=t.int64).to(device='cuda')
    for key in ('k_cache','v_cache'):
        b[key].view(t.uint8).copy_(a[key].view(t.uint8).index_select(0,nto))
        if not byte_equal(t,a[key],b[key].view(t.uint8).index_select(0,otn).view(a[key].dtype)):
            raise ValueError('Full encoded-cache inverse permutation failed')
    b['page_table'].copy_(t.tensor(plan.page_table_b,dtype=a['page_table'].dtype,device='cuda'))
    b['cache_loc'].copy_(t.tensor(plan.cache_loc_b,dtype=a['cache_loc'].dtype,device='cuda'))
    if tensor_state(t,a).keys() != tensor_state(t,b).keys(): raise ValueError('Fixture tensor keys changed')
    for key in a:
        if key in ('k_cache','v_cache','page_table','cache_loc'): continue
        if isinstance(a[key],t.Tensor):
            if not byte_equal(t,a[key],b[key]): raise ValueError('Nonlocality input difference: ' + key)
        elif a[key] != b[key]: raise ValueError('Config/None differs')
    return {ARMS[0]:a,ARMS[1]:b}, plan, otn


def execute(args, verified):
    started = time.monotonic()
    report = dict(kind='Selected Attention source: random versus canonical contiguous physical pages',
        status='starting', passed=False, started_utc=datetime.now(timezone.utc).isoformat(),
        worker_sha256=digest(Path(__file__).read_bytes()), verified_hashes=verified,
        correctness_complete=False,timing_complete=False, cases=[],counts={},
        activity={'status':'pending' if args.activity else 'not_requested'}, protocol=dict(cases=CASES,seeds=SEEDS,arms=ARMS,
            warmup_rounds_per_fixture=WARMUP_ROUNDS,reverse_pairs_per_fixture=REVERSE_PAIRS,
            same_graph_per_case=True,private_default_capture_pools=True,
            cold_l2_eviction_bytes=EVICTION_BYTES,raw_whole_call=True,replay_cost_subtracted=False,
            candidate_source_changed=False,page_table_always_present=True,
            provided_decode_metadata_preserved=True,diagnostic_only=True,
            expected_counts=expected_counts(args.activity)))
    counts = dict(original_oracle_calls=0,warmup_checked_replays=0,
                  measured_checked_replays=0,activity_checked_replays=0,total_checked_graph_replays=0,
                  original_oracle_calls_attempted=0,graph_replays_attempted=0,graph_replays_completed=0)
    active = []
    def save():
        report['counts'] = dict(counts); report['worker_wall_seconds'] = time.monotonic()-started
        raw = encoded(report)
        if len(raw) > MAX_TRANSPORT: raise ValueError('Result exceeds2MiB transport; no truncation')
        temp = args.output.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(args.output)
    def gate():
        if time.monotonic()-started >= args.deadline: raise TimeoutError('Locality worker deadline')
    def alarm(*_): raise TimeoutError('Locality worker alarm')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.deadline);save()
    try:
        import torch as t
        import triton
        if not t.cuda.is_available() or 'H200' not in t.cuda.get_device_name() or triton.__version__ != '3.8.0':
            raise ValueError('Pinned native H200/Triton3.8.0 required')
        report['environment'] = dict(torch=t.__version__,cuda=t.version.cuda,triton=triton.__version__,
            gpu=t.cuda.get_device_name(),capability=list(t.cuda.get_device_capability()),python=platform.python_version())
        sys.path.insert(0,str(V2/'official'))
        ref = load(V2/'official/reference.py','locality_reference')
        selected = load(V2/'fusion-lookahead-probe/candidate.py','locality_selected3da')
        helper = load(HERE/'fixture.py','locality_fixture')
        eviction = t.empty(EVICTION_BYTES//4,dtype=t.float32,device='cuda').fill_(1.)
        def run_graph(graph):
            counts['graph_replays_attempted']+=1
            graph.replay();t.cuda.synchronize()
            counts['graph_replays_completed']+=1
        def check_state(work,fresh,out,expected,state):
            if tensor_state(t,work) != state: raise ValueError('Captured input addresses/layout changed')
            if tensor_state(t,{'output':out})['output'] != expected['captured_output_state']:
                raise ValueError('Captured output address/layout changed')
            ownership(t,work,fresh)
            if not byte_equal(t,out,expected['output']): raise ValueError('Own oracle-checked output changed')
            rows = new_rows(t,work)
            if any(not byte_equal(t,rows[k],expected['new_rows'][k]) for k in rows):
                raise ValueError('Own oracle-checked new KV changed')
        for case in CASES:
            case_row = dict(case=case,fixtures=[],graph_captures=0)
            report['cases'].append(case_row);work=graph=out=state=None
            for seed in SEEDS:
                report['current_context']=dict(phase='fixture_generation',case=case['name'],seed=seed)
                gate();fixtures,plan,inverse = canonical_fixture(t,ref,helper,case,seed)
                fixture_row = dict(seed=seed,plan=helper.summarize_plan(plan),
                    locality_contrast=locality_contrast(plan,case['layer']),oracle=[],warmup=[],measured=[])
                case_row['fixtures'].append(fixture_row)
                if work is None:
                    report['current_context']=dict(phase='graph_capture',case=case['name'],seed=seed)
                    work=clone_inputs(t,fixtures[ARMS[0]]);state=tensor_state(t,work)
                    stream=t.cuda.Stream();stream.wait_stream(t.cuda.current_stream())
                    with t.cuda.stream(stream):
                        for _ in range(3): selected.kernel_fn(work)
                    t.cuda.current_stream().wait_stream(stream);restore(t,work,fixtures[ARMS[0]]);t.cuda.synchronize()
                    graph=t.cuda.CUDAGraph()
                    with t.cuda.graph(graph):out=selected.kernel_fn(work)
                    case_row['graph_captures']+=1
                    case_row['graph_pool_handle']=list(graph.pool())
                    case_row['captured_input_state']=state
                    case_row['captured_output_state']=tensor_state(t,{'output':out})['output']
                expected={};after_a={}
                for arm in ARMS:
                    report['current_context']=dict(phase='original_oracle',case=case['name'],seed=seed,arm=arm)
                    restore(t,work,fixtures[arm]);run_graph(graph);gate()
                    raw=ref.collect(work,out);counts['original_oracle_calls_attempted']+=1
                    ok,message=ref.check_implementation(work,raw)
                    if not ok: raise ValueError('Original oracle failed: '+message)
                    ownership(t,work,fixtures[arm])
                    if tensor_state(t,work)!=state: raise ValueError('Oracle replay changed tensor addresses')
                    if tensor_state(t,{'output':out})['output'] != case_row['captured_output_state']:
                        raise ValueError('Oracle replay changed captured output address/layout')
                    expected[arm]={'output':out.clone(),'new_rows':new_rows(t,work),
                                   'captured_output_state':case_row['captured_output_state']}
                    fixture_row['oracle'].append(dict(arm=arm,passed=True,message=message,ownership=True,stable_addresses=True))
                    counts['original_oracle_calls']+=1;counts['total_checked_graph_replays']+=1
                    if arm==ARMS[0]:
                        after_a={key:work[key].clone() for key in ('k_cache','v_cache')}
                    else:
                        for key in after_a:
                            recovered=work[key].view(t.uint8).index_select(0,inverse).view(after_a[key].dtype)
                            if not byte_equal(t,recovered,after_a[key]): raise ValueError('Post-call inverse logical cache bytes differ')
                fixture_row['post_call_full_cache_inverse_exact']=True
                fixture_row['cross_layout_output_exact']=byte_equal(t,expected[ARMS[0]]['output'],expected[ARMS[1]]['output'])
                del after_a
                for phase,rounds in [('warmup',WARMUP_ROUNDS),('measured',2*REVERSE_PAIRS)]:
                    for index in range(rounds):
                        order=list(ARMS if index%2==0 else reversed(ARMS)) if phase=='warmup' else measured_order(index)
                        row=dict(index=index,order=order,raw_us={})
                        for arm in order:
                            report['current_context']=dict(phase=phase,case=case['name'],seed=seed,arm=arm,round=index)
                            gate();restore(t,work,fixtures[arm]);eviction.sum();t.cuda.synchronize()
                            begin,end=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                            counts['graph_replays_attempted']+=1
                            begin.record();graph.replay();end.record();end.synchronize()
                            counts['graph_replays_completed']+=1
                            elapsed=begin.elapsed_time(end)*1000
                            if not math.isfinite(elapsed) or elapsed<=0: raise ValueError('Invalid raw event time')
                            row['raw_us'][arm]=elapsed
                            check_state(work,fixtures[arm],out,expected[arm],state)
                            counts[phase+'_checked_replays']+=1;counts['total_checked_graph_replays']+=1
                        fixture_row[phase].append(row)
                        save()
                fixture_row['timing_summary']=summarize_samples(fixture_row['measured'])
                active_state=(case,work,graph,out,state,fixtures,expected)
                save()
            active.append(active_state)
        pools=[tuple(row['graph_pool_handle']) for row in report['cases']]
        report['default_graph_pool_handles_distinct']=len(set(pools))==len(CASES)
        if not report['default_graph_pool_handles_distinct']:
            raise ValueError('Default private graph-pool handles are not distinct')
        report['correctness_complete']=True;report['timing_complete']=True;save()
        if args.activity:
            gate();labels=[]
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,t.profiler.ProfilerActivity.CUDA]) as prof:
                for case,work,graph,out,state,fixtures,expected in active:
                    for arm in ARMS:
                        report['current_context']=dict(phase='activity',case=case['name'],seed=SEEDS[-1],arm=arm)
                        gate();restore(t,work,fixtures[arm]);eviction.sum();t.cuda.synchronize()
                        label='LOCALITY_'+case['name']+'_'+arm;labels.append(label)
                        with t.profiler.record_function(label):run_graph(graph)
                        check_state(work,fixtures[arm],out,expected[arm],state)
                        counts['activity_checked_replays']+=1;counts['total_checked_graph_replays']+=1
            trace_path=args.output.parent/'activity-trace.json'
            prof.export_chrome_trace(str(trace_path));gate()
            if trace_path.stat().st_size>MAX_RAW_TRACE:raise ValueError('Raw trace exceeds8MiB; no truncation')
            raw=trace_path.read_bytes();trace=json.loads(raw)
            report['activity']=dict(status='complete',seed=SEEDS[-1],sessions=1,
                summary=parse_activity(trace,labels),artifact=trace_artifact(raw))
            # Same captured graph for A/B; retain actual native identity comparisons separately.
            comparisons=[]
            for case in CASES:
                groups=report['activity']['summary']['groups']
                left=groups['LOCALITY_'+case['name']+'_'+ARMS[0]]['kernels']
                right=groups['LOCALITY_'+case['name']+'_'+ARMS[1]]['kernels']
                comparisons.append(dict(case=case['name'],**native_identity_comparison(left,right)))
            report['activity']['cross_layout_native_identity']=comparisons
            report['activity']['native_identity_fully_observed']=all(r['same_symbol_and_launch_fields'] is True for r in comparisons)
            if any(r['same_symbol_and_launch_fields'] is False for r in comparisons):
                raise ValueError('Observed graph kernel identity/geometry differs across layouts')
        wanted=expected_counts(args.activity)
        if any(counts[k]!=wanted[k] for k in counts):raise ValueError('Declared replay/check count mismatch')
        report.update(status='complete',passed=True,current_context=None,
            qualification='Physical locality sensitivity with unchanged selected source and original oracle. No official score, production candidate, SASS/counters, achieved occupancy or HBM-bottleneck conclusion.')
        gate();save();return 0
    except Exception as exc:
        report.update(status='failed',passed=False,error_type=type(exc).__name__,error=str(exc)[:1024])
        try:save()
        except ValueError:
            minimal=dict(status='failed',passed=False,error_type='transport_bound',
                reason='Full result not transported; strict2MiB bound',worker_sha256=report['worker_sha256'],
                verified_hashes=verified,counts=counts,correctness_complete=report['correctness_complete'],timing_complete=report['timing_complete'])
            args.output.write_bytes(encoded(minimal))
        return 1
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--activity',action='store_true')
    p.add_argument('--deadline',type=int,default=160)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not 1<=args.deadline<=MAX_SECONDS:p.error('deadline must be1..160seconds')
    if args.activity and not args.execute:p.error('--activity requires explicit--execute')
    verified=source_gate()
    if not args.execute:
        report=dict(status='CPU_preflight_only',verified_hashes=verified,vendor_imports_executed=False,
                    kernel_launches=0,service_calls=0,expected_counts_without_activity=expected_counts(False),
                    expected_counts_with_activity=expected_counts(True))
        raw=encoded(report)
        if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(raw)
        else:print(raw.decode(),end='')
        return 0
    if args.output is None:p.error('--execute requires --output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    return execute(args,verified)


if __name__=='__main__':raise SystemExit(main())
