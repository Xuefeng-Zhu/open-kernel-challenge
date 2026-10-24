"""Focused A/B/A graph replay and compiler inspection, not a performance benchmark."""
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent


def make_entry():
    def run(mode,deadline=70):
        import json,os,signal,subprocess,sys,time
        from pathlib import Path
        root=Path('/opt/attention-bundle/reviews/attention-v2');output=Path('/tmp/replay-probe.json')
        started=time.monotonic()
        with Path('/tmp/replay-probe.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(root/'replay_probe.py'),str(output),str(deadline)],
                cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=deadline+10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait();code=-9
        return dict(exit_code=code,worker_wall_seconds=time.monotonic()-started,
            report=json.loads(output.read_text()) if output.exists() else None,
            log=Path('/tmp/replay-probe.log').read_text()[-16000:])
    return run


def compiled(module,data):
    import torch,triton
    cfg=data['config'];b,l,hq,hkv,d=[cfg[k] for k in ['batch','q_len','num_q_heads','num_kv_heads','head_dim']]
    g=hq//hkv;bm=max(16,triton.next_power_of_2(l*g));x=data['qkv'].reshape(b*l,-1)
    span=data['page_table'].shape[1]*cfg['page_size'];kw={}
    if 'CHUNK_SIZE' in inspect.signature(module._fused.fn).parameters:
        size=8192 if cfg['window']==0 and b in (16,32) else 0;kw['CHUNK_SIZE']=size
        splits=(span+size-1)//size if size else module._num_splits(b,hkv,span,64)
    else:splits=module._num_splits(b,hkv,span,64)
    n=b*hkv*splits*bm;out=torch.empty((b*l,hq*d),device=x.device,dtype=torch.bfloat16)
    po=torch.empty((n,d),device=x.device);pm=torch.empty((n,),device=x.device);pl=torch.empty_like(pm)
    k=module._fused.warmup(x,data['seq_lens'],data['page_table'],data['cache_loc'],data['k_cache'],
        data['v_cache'],data['k_scale'],data['v_scale'],po,pm,pl,module._tickets(x.device,b*hkv),out,
        1.,module.RMS_EPS,float(cfg['qk_scale_factor']),float(cfg['sm_scale']),cfg['window'],cfg['page_size'],
        x.stride(0),data['page_table'].stride(0),L=l,G=g,HQ=hq,HKV=hkv,D=d,BLOCK_M=bm,
        BLOCK_N=64,BLOCK_L=16,SPLITS=splits,ROPE=False,WINDOWED=False,DOT=module._DOT,
        num_warps=4,num_stages=3,grid=(b,hkv,splits),**kw)
    k._init_handles();ptx=k.asm.get('ptx','')
    canonical='\n'.join(line for line in ptx.splitlines() if not line.strip().startswith(('.file','.loc','//')))
    canonical=canonical.replace(k.metadata.name,'KERNEL')
    return dict(splits=splits,registers=k.n_regs,n_spills_driver_raw=k.n_spills,shared=k.metadata.shared,
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),canonical_ptx_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        ptx=ptx,ttgir=k.asm.get('ttgir',''),instruction_counts={s:ptx.count(s) for s in
          ['wgmma.mma_async','mma.sync','cp.async','ld.global','ld.shared','bar.sync']})


def main():
    import torch,triton
    from modal_worker import load,clone_inputs,restore,capture,check
    output=Path(sys.argv[1]);deadline=time.monotonic()+int(sys.argv[2])
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required, no fallback')
    manifest=json.loads((HERE/'bundle-manifest.json').read_text());plan=manifest['probe']
    report=dict(kind='Focused same-address A/B/A correctness and compiler inspection; no speed claim',
        passed=False,cases=[],environment=dict(torch=torch.__version__,triton=triton.__version__,
        gpu=torch.cuda.get_device_name()),plan=plan)
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(output)
    save()
    for name,sha in manifest['sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen bundle changed')
    sys.path.insert(0,str(HERE/'official'));reference=load(HERE/'official/reference.py','aba_reference')
    modules={name:load(HERE/file,'aba_'+name) for name,file in [('parent',plan['parent']),('candidate',plan['candidate'])]}
    # B16 mixed lengths include neutral non-owner chunks. B32 jointly tests page64,
    # non-unit scales and L8, absent from the previous cycle's replay coverage.
    cases=[dict(batch=16,ctx=80000,heads='tp8',layer='full',q_len=8,page=1,spread='mixed',dist='coherent'),
           dict(batch=32,ctx=80000,heads='tp8',layer='full',q_len=8,page=64,kv_scale=1/64,spread='agentic',dist='normal')]
    for i,case in enumerate(cases):
        if time.monotonic()>deadline:raise TimeoutError('Probe deadline')
        a=reference.generate_input(**case,seed=910000+i*100,device='cuda')
        b=reference.generate_input(**case,seed=910001+i*100,device='cuda')
        work=clone_inputs(a);pointers={k:v.data_ptr() for k,v in work.items() if isinstance(v,torch.Tensor)}
        lengths_a=a['seq_lens'].cpu().tolist();lengths_b=b['seq_lens'].cpu().tolist()
        assert lengths_a!=lengths_b
        item=dict(case=case,lengths_A=lengths_a,lengths_B=lengths_b,runners=[]);report['cases'].append(item)
        for name,module in modules.items():
            restore(work,a);graph,out=capture(module.kernel_fn,work,a)
            row=dict(runner=name,graph_captures=1,replays=[],compiler=compiled(module,a));item['runners'].append(row)
            first=None
            for label,fresh in [('A1',a),('B',b),('A2',a)]:
                restore(work,fresh)
                stable=all(work[k].data_ptr()==ptr for k,ptr in pointers.items())
                graph.replay();result=check(reference,work,out,fresh);raw=reference.collect(work,out).clone()
                if label=='A1':first=raw
                equal=bool(torch.equal(raw,first)) if label=='A2' else None
                row['replays'].append(dict(label=label,same_buffer_addresses=stable,
                    actual_seq_lens=work['seq_lens'].cpu().tolist(),A_return_output_and_newKV_bitwise_equal=equal,**result));save()
                if not stable or not result['passed'] or (label=='A2' and not equal):return 1
            del graph,out
        del a,b,work
    report['passed']=True;save();return 0


if __name__=='__main__':raise SystemExit(main())
