"""Frozen source-bound, serialized v298/champion H200 comparison.

No official runner, counter profiler, external model or network call occurs here.
CPU imports perform no Torch or candidate imports. Native compilation is warmed
outside timed calls, matching the allowed compilation exception.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

SOURCES = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'v298.py': '0cc4a0668acf7553c0d2c6d5c3a2e7d5c85374a70ad774463e9d2fd9e5d0f883',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
SEEDS = (135901, 135902)
ARMS = ('champion', 'v298')
EVICTION_BYTES = 256 * 1024 * 1024
TEST_CASES = (
    dict(batch=1, ctx=10, heads='tp8', layer='sliding'),
    dict(batch=3, ctx=2047, heads='tp8', layer='sliding'),
    dict(batch=5, ctx=5000, heads='tp8', layer='sliding', q_len=8, spread='mixed', page=1),
    dict(batch=4, ctx=3000, heads='tp8', layer='full'),
    dict(batch=8, ctx=8192, heads='dp', layer='full', spread='mixed'),
    dict(batch=2, ctx=100000, heads='tp8', layer='full', q_len=8, page=1),
    dict(batch=4, ctx=6000, heads='tp8', layer='sliding', dist='edges', page=1),
    dict(batch=4, ctx=6000, heads='tp8', layer='full', q_len=8, dist='edges'),
    dict(batch=1, ctx=1048576, heads='tp8', layer='full', q_len=8),
    dict(batch=3, ctx=4000, heads='tp8', layer='sliding', q_len=8, page=1, kv_scale=.015625),
    dict(batch=2, ctx=9000, heads='tp8', layer='full', kv_scale=.015625),
    dict(batch=4, ctx=80000, heads='tp8', layer='full', spread='mixed', dist='soft'),
    dict(batch=4, ctx=6000, heads='tp8', layer='sliding', q_len=8, dist='soft', page=1),
    dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='soft'),
    dict(batch=4, ctx=80000, heads='tp8', layer='full', spread='mixed', dist='coherent', page=1),
    dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='coherent'),
)


def bind_sources(root):
    found = {}
    for name, wanted in SOURCES.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or sha(path) != wanted:
            raise ValueError('Source binding failed: ' + name)
        found[name] = wanted
    return found


def case(batch, length=1, layer='full'):
    return dict(batch=batch, q_len=length, layer=layer, ctx=80000, heads='tp8',
                spread='fixed' if batch == 1 else 'agentic', page=1, kv_scale=1.,
                dist='coherent' if layer == 'full' else 'soft')


def scored_cases():
    return [case(batch, length, layer) for length in (1, 8)
            for batch in (1, 16, 32) for layer in ('sliding', 'full')]


def order(index):
    pair, half = divmod(index, 2)
    return ARMS if (pair + half) % 2 == 0 else ARMS[::-1]


def paired_summary(samples):
    if len(samples) != 12 or any(set(x) != set(ARMS) for x in samples):
        raise ValueError('Six complete reverse-order pairs required')
    ratios = []
    for row in samples:
        if any(not math.isfinite(v) or v <= 0 for v in row.values()):
            raise ValueError('Finite positive raw graph times required')
        ratios.append(row['champion'] / row['v298'])
    pairs = [math.sqrt(ratios[i] * ratios[i+1]) for i in range(0, 12, 2)]
    return dict(champion_median_us=statistics.median(x['champion'] for x in samples),
                v298_median_us=statistics.median(x['v298'] for x in samples),
                reverse_pair_ratios=pairs, median_pair_speedup=statistics.median(pairs),
                favorable_pairs=sum(x > 1 for x in pairs))


def whole_suite(rows, seed):
    selected = [row for row in rows if row['seed'] == seed]
    wanted = {(b,l,k) for b in (1,16,32) for l in (1,8) for k in ('sliding','full')}
    keys = [(r['case']['batch'], r['case']['q_len'], r['case']['layer']) for r in selected]
    if len(keys) != 12 or set(keys) != wanted:
        raise ValueError('Exactly twelve distinct scored rows required per fixture')
    bykey = dict(zip(keys, selected)); loads=[]
    for length in (1,8):
        for batch in (1,16,32):
            c, v = 0., 0.
            for layer, weight in (('sliding',46),('full',16)):
                summary = bykey[(batch,length,layer)]['summary']
                control = summary['champion_median_us']
                speedup = summary['median_pair_speedup']
                if not math.isfinite(control) or control <= 0 or not math.isfinite(speedup) or speedup <= 0:
                    raise ValueError('Invalid whole-suite timing inputs')
                c += weight * control; v += weight * control / speedup
            loads.append(dict(batch=batch, q_len=length, champion_weighted_us=c,
                              v298_reference_normalized_weighted_us=v, speedup=c/v))
    return dict(seed=seed, loads=loads,
                relative_champion_geomean=math.prod(r['speedup'] for r in loads)**(1/6),
                qualification='Raw whole-call graph times; no fixed replay subtraction. Relative to selected champion, not an official score or SGLang comparison.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n'); temporary.replace(path)

def load(root, name, filename):
    spec = importlib.util.spec_from_file_location(name, root/filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def clone(torch, data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else dict(v) if isinstance(v, dict) else v
            for k, v in data.items()}

def restore(torch, work, fresh):
    if work.keys() != fresh.keys(): raise ValueError('Input dictionary keys changed')
    for key, value in fresh.items():
        if isinstance(value, torch.Tensor):
            if work[key].shape != value.shape or work[key].stride() != value.stride() or work[key].dtype != value.dtype:
                raise ValueError('Fixed-address layout changed: ' + key)
            work[key].copy_(value)
        elif work[key] != value:
            raise ValueError('Input metadata changed: ' + key)

def pointers(torch, data):
    return {k: (v.data_ptr(), tuple(v.shape), tuple(v.stride()), str(v.dtype))
            for k, v in data.items() if isinstance(v, torch.Tensor)}

def mutated_fixture(torch, fresh):
    """Untimed, same-geometry live data mutation; never used for timing."""
    if fresh['page_table'].shape[1] < 2 or bool((fresh['seq_lens'] < 2).any()):
        raise ValueError('Declared fixture lacks two historical pages per request')
    mutated = clone(torch, fresh)
    mutated['qkv'].mul_(0.93)
    first = mutated['page_table'][:, 0].clone()
    mutated['page_table'][:, 0].copy_(mutated['page_table'][:, 1])
    mutated['page_table'][:, 1].copy_(first)
    if not bytes_equal(torch, mutated['cache_loc'], fresh['cache_loc']):
        raise ValueError('Mutation changed new KV write ownership')
    return mutated

def bytes_equal(torch, a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))

def check_ownership(torch, work, fresh):
    if work.keys() != fresh.keys(): raise ValueError('Input dictionary keys changed')
    for key, value in fresh.items():
        if key in ('k_cache', 'v_cache'):
            continue
        if isinstance(value, torch.Tensor):
            if not bytes_equal(torch, work[key], value):
                raise ValueError('Immutable input changed: ' + key)
        elif work[key] != value:
            raise ValueError('Immutable metadata changed: ' + key)
    locations = fresh['cache_loc'].long()
    width = fresh['config']['num_kv_heads'] * fresh['config']['head_dim']
    for key in ('k_cache', 'v_cache'):
        got, original = [x.view(-1, width).view(torch.uint8) for x in (work[key], fresh[key])]
        new = got[locations].clone()
        try:
            got[locations] = original[locations]
            if not bool(torch.equal(got, original)):
                raise ValueError('Historical cache rows changed: ' + key)
        finally:
            got[locations] = new

def oracle(torch, ref, work, fresh, output, state):
    torch.cuda.synchronize()
    raw = ref.collect(work, output)
    passed, message = ref.check_implementation(work, raw)
    if not passed:
        raise ValueError('Original oracle: ' + message)
    check_ownership(torch, work, fresh)
    if pointers(torch, work) != state:
        raise ValueError('Captured addresses changed')
    return dict(passed=True, message=message, ownership=True, fixed_addresses=True)

def make_graph(torch, fn, fresh):
    work = clone(torch, fresh)
    state = pointers(torch, work)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(2):
            fn(work)
    torch.cuda.current_stream().wait_stream(warm); torch.cuda.synchronize()
    restore(torch, work, fresh)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(work)
    return work, state, graph, output

def timed(torch, graph, eviction):
    eviction.zero_()  # Outside events, >H200 L2; never subtract profile time.
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(); graph.replay(); end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000

def cache_bytes(torch, ref, work, out):
    raw = ref.collect(work, out)
    return raw[out.numel() * out.element_size():].clone()


def assert_replay(torch, ref, work, out, state, expected_cache):
    if pointers(torch, work) != state:
        raise ValueError('Captured addresses changed')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('Graph replay output contains NaN/Inf')
    if not bytes_equal(torch, cache_bytes(torch, ref, work, out), expected_cache):
        raise ValueError('New cache bytes changed across identical replays')


def native_plan(module, data):
    cfg = data['config']; device=data['qkv'].device
    p = module._plan(cfg['batch'], cfg['num_kv_heads']//2, cfg['q_len'], cfg['window'],
                     data['page_table'].shape[1], cfg['page_size'], device)
    override=module._TUNED.get((cfg['batch'], cfg['q_len'], bool(cfg['window']), cfg['num_kv_heads']//2))
    return dict(base_shape_plan=p, default_tuned_override=override,
                supported=bool(module._supported(cfg) and module._fits(data)),
                qualification='Unmodified source planning metadata; actual launch grid not profiled')


def run(root, output, seconds):
    started=time.monotonic()
    report=dict(status='running', worker_sha256=sha(__file__), verified_hashes=bind_sources(root),
        prerequisite_cases=[], paired_cases=[], whole_suite=[], correctness_complete=False,
        timing_complete=False, candidate_changed=False, official_check=False,
        ranked_submission=False,
        protocol=dict(seeds=list(SEEDS), reverse_pairs=6, eviction_bytes=EVICTION_BYTES,
            graph_timing='raw whole-call CUDA event graph replay; no fixed cost subtraction',
            serialization='Both candidate global scratch and all GPU calls are serialized; no concurrent-stream claim',
            counters=False, promotion='not assessed; selected champion preserved'))

    def phase(name):
        report['phase']=name; report['elapsed_seconds']=time.monotonic()-started
        save(output, report); print('PROGRESS='+name, flush=True)
        if report['elapsed_seconds'] > seconds:
            raise TimeoutError('Worker deadline reached at '+name)

    try:
        phase('import-native-modules')
        import torch
        import triton
        ref=load(root,'reference','reference.py')
        modules={'champion':load(root,'selected','selected.py'), 'v298':load(root,'v298','v298.py')}
        if modules['champion']._INTERP:
            raise ValueError('Native probe forbids Triton interpret')
        props=torch.cuda.get_device_properties(0)
        if 'H200' not in props.name or (props.major,props.minor)!=(9,0):
            raise ValueError('Expected actual H200 SM90, got '+props.name)
        report['runtime']=dict(gpu=props.name,sm_count=props.multi_processor_count,
            torch=torch.__version__,triton=triton.__version__,device_capability=[props.major,props.minor])
        phase('v298-native-build-begin')
        built=modules['v298']._module()
        phase('v298-native-build-complete')
        binary=Path(built.__file__)
        report['runtime']['v298_binary']=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size)
        for index,spec in enumerate(TEST_CASES):
            phase('prerequisite-'+str(index+1)+'-of-16')
            seed=135800+index
            fresh=ref.generate_input(**spec,seed=seed,device='cuda')
            row=dict(index=index+1,case=spec,seed=seed,arms={})
            report['prerequisite_cases'].append(row)
            for arm in ARMS:
                phase('prerequisite-'+str(index+1)+'-'+arm)
                work=clone(torch,fresh);state=pointers(torch,work)
                out=modules[arm].kernel_fn(work)
                row['arms'][arm]=oracle(torch,ref,work,fresh,out,state)
                del work,state,out
            del fresh
            gc.collect();torch.cuda.empty_cache()
        report['prerequisites_complete']=True
        phase('sixteen-prerequisites-both-arms-pass')
        eviction=torch.empty(EVICTION_BYTES,dtype=torch.uint8,device='cuda')
        for seed in SEEDS:
            for spec in scored_cases():
                key=f"b{spec['batch']}-l{spec['q_len']}-{spec['layer']}-seed{seed}"
                phase('paired-'+key)
                fresh=ref.generate_input(**spec,seed=seed,device='cuda')
                lengths=fresh['seq_lens'].cpu().tolist()
                row=dict(case=spec,seed=seed,oracle={},mutation_oracle={},post_oracle={},
                    samples=[],fixture=dict(seq_lens=lengths,seq_len_sum=sum(lengths),
                    page_table_shape=list(fresh['page_table'].shape)),
                    v298_plan=native_plan(modules['v298'],fresh))
                report['paired_cases'].append(row);arms={}
                for arm in ARMS:
                    phase('graph-'+key+'-'+arm)
                    work,state,graph,out=make_graph(torch,modules[arm].kernel_fn,fresh)
                    restore(torch,work,fresh);graph.replay()
                    row['oracle'][arm]=oracle(torch,ref,work,fresh,out,state)
                    expected_cache=cache_bytes(torch,ref,work,out)
                    changed=mutated_fixture(torch,fresh)
                    restore(torch,work,changed);graph.replay()
                    row['mutation_oracle'][arm]=dict(timed=False,
                        kind='qkv *=0.93; first two historical page indices swapped; fixed captured addresses',
                        **oracle(torch,ref,work,changed,out,state))
                    restore(torch,work,fresh);graph.replay()
                    assert_replay(torch,ref,work,out,state,expected_cache)
                    arms[arm]=(work,state,graph,out,expected_cache)
                    del changed
                if arms['champion'][3].data_ptr()==arms['v298'][3].data_ptr():
                    raise ValueError('Arm output alias')
                phase('timing-'+key)
                for index in range(12):
                    sample={}
                    for arm in order(index):
                        work,state,graph,out,expected_cache=arms[arm]
                        restore(torch,work,fresh)
                        sample[arm]=timed(torch,graph,eviction)
                        assert_replay(torch,ref,work,out,state,expected_cache)
                    row['samples'].append(sample)
                row['summary']=paired_summary(row['samples'])
                for arm in ARMS:
                    work,state,graph,out,expected_cache=arms[arm]
                    row['post_oracle'][arm]=oracle(torch,ref,work,fresh,out,state)
                del arms,work,state,graph,out,expected_cache,fresh
                gc.collect();torch.cuda.empty_cache()
                phase('completed-'+key)
            report['whole_suite'].append(whole_suite(report['paired_cases'],seed))
        report['correctness_complete']=True;report['timing_complete']=True
        report['status']='complete';phase('complete')
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        save(output,report)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;save(output,report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--payload-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=int,default=540)
    args=p.parse_args()
    if args.deadline<=0 or args.deadline>540:
        raise ValueError('Bounded positive worker deadline <=540s required')
    run(args.payload_root.resolve(),args.output.resolve(),args.deadline)


if __name__=='__main__':
    main()
