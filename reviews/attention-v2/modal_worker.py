"""Allowlisted attention harness. Runs only inside separately authorized H200 app.

No Modal credentials, network or competing repository code is used here.
Raw graph times and independently empty-graph-calibrated diagnostics are separate.
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def make_entry():
    def run(mode,deadline=210):
        import json,os,signal,subprocess,sys,time
        from pathlib import Path
        root=Path('/opt/attention-bundle/reviews/attention-v2')
        result=Path('/tmp/attention-result.json')
        started=time.monotonic()
        with Path('/tmp/attention-worker.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(root/'modal_worker.py'),'--mode',mode,
                '--deadline',str(deadline),'--output',str(result)],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=deadline+10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait();code=-9
        return dict(exit_code=code,worker_wall_seconds=time.monotonic()-started,
            report=json.loads(result.read_text()) if result.exists() else None,
            log=Path('/tmp/attention-worker.log').read_text()[-16000:])
    return run


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def clone_inputs(data):
    import torch
    return {k:(v.clone() if isinstance(v,torch.Tensor) else (dict(v) if isinstance(v,dict) else v))
            for k,v in data.items()}


def restore(work,fresh):
    import torch
    if work.keys()!=fresh.keys():raise ValueError('Input keys changed')
    for k,v in fresh.items():
        if isinstance(v,torch.Tensor):
            if work[k].shape!=v.shape or work[k].dtype!=v.dtype:
                raise ValueError('Fresh tensor metadata changed: '+k)
            work[k].copy_(v)
        elif work[k]!=v:raise ValueError('Fresh config/None changed: '+k)


def unchanged(work,fresh):
    """Only new cache rows may differ. Byte comparisons support FP8 tensors."""
    import torch
    for key,v in work.items():
        if not isinstance(v,torch.Tensor):
            if v!=fresh[key]:return False,'modified immutable metadata '+key
            continue
        if key in {'k_cache','v_cache'}:continue
        if not torch.equal(v.contiguous().reshape(-1).view(torch.uint8),fresh[key].contiguous().reshape(-1).view(torch.uint8)):
            return False,'modified immutable input '+key
    cfg=work['config'];h,d=cfg['num_kv_heads'],cfg['head_dim']
    slots=fresh['cache_loc'].long()
    for key in ['k_cache','v_cache']:
        got=work[key].view(-1,h*d).view(torch.uint8)
        original=fresh[key].view(-1,h*d).view(torch.uint8)
        saved=got[slots].clone();got[slots]=original[slots]
        ok=torch.equal(got,original);got[slots]=saved
        if not ok:return False,'modified past/unused cache rows '+key
    return True,'immutable inputs and non-new cache rows unchanged'


def capture(fn,work,fresh):
    import torch
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn(work)
    torch.cuda.current_stream().wait_stream(stream)
    restore(work,fresh);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(work)
    return graph,out


def timed(graph,eviction):
    import torch
    eviction.sum()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();graph.replay();end.record();end.synchronize()
    return begin.elapsed_time(end)*1000


def check(reference,work,out,fresh):
    raw=reference.collect(work,out)
    ok,message=reference.check_implementation(work,raw)
    intact,details=unchanged(work,fresh)
    return dict(passed=bool(ok and intact),oracle_message=message,immutability=details)


def resources(module,case,data):
    """Read compiled cache metadata through documented warmup pattern, not counters."""
    import torch,triton
    cfg=data['config'];b,l,hq,hkv,d=[cfg[k] for k in
        ['batch','q_len','num_q_heads','num_kv_heads','head_dim']]
    x=data['qkv'].reshape(b*l,-1);g=hq//hkv;bm=max(16,triton.next_power_of_2(l*g))
    span=cfg['window'] or data['page_table'].shape[1]*cfg['page_size']
    splits=module._num_splits(b,hkv,span,64)
    out=torch.empty((b*l,hq*d),device=x.device,dtype=torch.bfloat16)
    n=b*hkv*splits*bm
    po=torch.empty((n,d),device=x.device) if splits>1 else out
    pm=torch.empty((n,),device=x.device) if splits>1 else out
    pl=torch.empty_like(pm);tickets=module._tickets(x.device,b*hkv)
    kernel=module._fused.warmup(x,data['seq_lens'],data['page_table'],data['cache_loc'],
        data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],po,pm,pl,tickets,out,
        float(cfg['rope_theta']) if cfg['rope_theta'] else 1.,module.RMS_EPS,
        float(cfg['qk_scale_factor']),float(cfg['sm_scale']),cfg['window'],cfg['page_size'],
        x.stride(0),data['page_table'].stride(0),L=l,G=g,HQ=hq,HKV=hkv,D=d,
        BLOCK_M=bm,BLOCK_N=64,BLOCK_L=16,SPLITS=splits,ROPE=bool(cfg['rope_theta']),
        WINDOWED=bool(cfg['window']),DOT=module._DOT,num_warps=4,num_stages=3,grid=(b,hkv,splits))
    kernel._init_handles();ptx=kernel.asm.get('ptx','')
    return dict(case=case,splits=splits,registers_per_thread=kernel.n_regs,
        n_spills_driver_raw=kernel.n_spills,shared_bytes=kernel.metadata.shared,
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),ptx=ptx,
        qualification='Static compiler/driver metadata, not achieved occupancy or measured DRAM traffic')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['baseline','merge'],required=True)
    parser.add_argument('--deadline',type=int,default=210)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch,triton
    started=time.monotonic();deadline=started+args.deadline
    report=dict(kind='Attention Modal H200 diagnostic, not official score',mode=args.mode,
        tests=[],benchmarks=[],resources=[],passed=False,
        timing=dict(warmups=2,steady=6,orders='alternate two runners, reverse-order pairs',
            eviction_bytes=256*1024*1024,raw='CUDA events around whole kernel_fn graph',
            calibrated='Subtract separately measured same-case empty-graph median from both sides; no clipping, not judge calibration'),
        concurrency='Sequential isolated graph replays only; original global-ticket overlap defect remains')
    def save():
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)
    save()
    def gate():
        save()
        if time.monotonic()>deadline:raise TimeoutError('Internal attention deadline')
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        report['blocked']='H200 required';save();return 2
    props=torch.cuda.get_device_properties(0)
    report['environment']=dict(torch=torch.__version__,triton=triton.__version__,gpu=props.name,
        sm_count=props.multi_processor_count,python=sys.version,public_image_not_private_judge=True)
    manifest=json.loads((HERE/'bundle-manifest.json').read_text())
    for path,sha in manifest['sha256'].items():
        if hashlib.sha256((HERE/path).read_bytes()).hexdigest()!=sha:raise ValueError('Bundle changed: '+path)
    report['bundle']=manifest
    sys.path.insert(0,str(HERE/'official'))
    reference=load(HERE/'official/reference.py','attention_reference')
    baseline=load(HERE/'submission.py','attention_v2')
    if args.mode=='baseline':
        other=load(HERE/'official/baseline.py','attention_sglang');names=['v2','sglang']
    else:
        other=load(HERE/'separate_merge.py','attention_merge');names=['merge','v2']
    modules={'v2':baseline,names[0] if args.mode=='merge' else 'sglang':other}
    fixtures=json.loads((HERE/'fixtures.json').read_text())
    report['source_sha256']={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                             for name,module in modules.items()}
    # All sixteen prerequisite shapes; each captured and checked, plus scored replays below.
    for index,case in enumerate(fixtures['tests']):
        gate();fresh=reference.generate_input(**case,device='cuda');work=clone_inputs(fresh)
        for name in names:
            restore(work,fresh);graph,out=capture(modules[name].kernel_fn,work,fresh)
            restore(work,fresh);graph.replay();result=check(reference,work,out,fresh)
            report['tests'].append(dict(index=index,case=case,runner=name,**result));gate()
            if not result['passed']:return 1
        del graph,out,work,fresh
    eviction=torch.ones(report['timing']['eviction_bytes']//4,device='cuda')
    for index,case in enumerate(fixtures['benchmarks']):
        gate();fresh=reference.generate_input(**case,device='cuda');work=clone_inputs(fresh)
        graphs={};outputs={}
        for name in names:
            restore(work,fresh);graphs[name],outputs[name]=capture(modules[name].kernel_fn,work,fresh)
        item=dict(case=case,warmup=[],samples=[],calibration=[]);report['benchmarks'].append(item)
        try:
            empty=torch.cuda.CUDAGraph()
            with torch.cuda.graph(empty):pass
            for _ in range(2):empty.replay()
            item['calibration']=[timed(empty,eviction) for _ in range(3)]
        except Exception as error:
            item['calibration_unavailable']=dict(type=type(error).__name__,message=str(error)[:500])
            empty=None
        for i in range(8):
            gate();seed=case['seed']+7000000+i
            fresh=reference.generate_input(**{**case,'seed':seed},device='cuda')
            order=names if i%2==0 else list(reversed(names));row=dict(seed=seed,order=order,raw_us={},checks={})
            for name in order:
                restore(work,fresh);row['raw_us'][name]=timed(graphs[name],eviction)
                row['checks'][name]=check(reference,work,outputs[name],fresh)
                if not row['checks'][name]['passed']:
                    item['failed']=row;save();return 1
            item['warmup' if i<2 else 'samples'].append(row);gate()
        if empty is not None:item['calibration'] += [timed(empty,eviction) for _ in range(3)]
        if item['calibration']:
            offset=statistics.median(item['calibration']);item['empty_graph_median_us']=offset
            for row in item['warmup']+item['samples']:
                row['calibrated_us']={name:value-offset for name,value in row['raw_us'].items()}
            item['calibration_resolves_both']=all(value>0 for row in item['samples'] for value in row['calibrated_us'].values())
        if case['layer']=='full' and case.get('q_len',1)==8 and case['batch'] in [16,32]:
            report['resources'].append(resources(baseline,case,work))
        gate();del graphs,outputs,work,fresh
    report['passed']=True;report['wall_seconds']=time.monotonic()-started;save();return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc();raise
