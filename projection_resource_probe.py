"""Inspect unchanged, already validated kernels; no hardware performance counters.

Run inside the bounded Modal worker. PTX counts/operand requests/resource ceilings
are static diagnostics, not measured DRAM traffic or achieved occupancy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent


def operand_requests(m, n, k, bm, bn):
    return dict(grid_ctas=math.ceil(m / bm) * math.ceil(n / bn),
                x_logical_bytes=m * k * 2 * math.ceil(n / bn),
                w_logical_bytes=n * k * 2 * math.ceil(m / bm),
                unique_x_bytes=m * k * 2, unique_w_bytes=n * k * 2,
                output_bytes=m * n * 2,
                qualification="Logical valid operand requests, not measured HBM/cache traffic")


def ptx_features(ptx):
    # Only instruction lines, not symbols, comments or compiler filenames.
    instructions=[]
    for line in ptx.splitlines():
        line=line.split('//',1)[0].strip()
        line=re.sub(r'^@!?%\w+\s+', '', line)
        if re.match(r'^[a-z][a-z0-9_.]*\s',line):
            instructions.append(line.split()[0])
    prefixes=('mma.sync','wgmma.mma_async','cp.async','cp.async.bulk',
              'ld.global','ld.local','st.local','ld.shared','st.shared','bar.sync')
    return dict(static_instruction_counts={p:sum(x.startswith(p) for x in instructions) for p in prefixes},
                qualification="Static PTX occurrences, not dynamic instruction counts or SASS analysis")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline-seconds',type=float,default=150)
    args=parser.parse_args()
    import torch
    import triton
    from benchmark import capture
    from harness import load_module
    started=time.monotonic()
    report=dict(kind="Unchanged-kernel resource/ordinary CUDA timeline diagnostic, no algorithm benchmark",
                source_sha256={},compiled=[],timelines=[],passed=False,
                counter_access="No Nsight/CUPTI hardware metrics requested; ordinary PyTorch CUDA activity only")
    def save():
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    def gate():
        if time.monotonic()-started>args.deadline_seconds:
            save();raise RuntimeError('Internal diagnostic deadline')
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        report['blocked']='H200 required';save();return 2
    props=torch.cuda.get_device_properties(0)
    report['hardware']=dict(name=props.name,sm_count=props.multi_processor_count,
                            compute_capability=list(torch.cuda.get_device_capability()),
                            torch=torch.__version__,triton=triton.__version__,
                            shared_per_sm_bytes=getattr(props,'shared_memory_per_multiprocessor',None))
    report['triton_device_properties']=triton.runtime.driver.active.utils.get_device_properties(0)
    reference=load_module(ROOT/'official/reference.py','probe_reference')
    files={'best':'projection_full_composition.py','bn128':'projection_pooler_grouped.py',
           'bn32':'projection_pooler_narrow.py'}
    modules={name:load_module(ROOT/'candidates'/file,'probe_'+name) for name,file in files.items()}
    report['source_sha256']={name:hashlib.sha256((ROOT/'candidates'/file).read_bytes()).hexdigest()
                             for name,file in files.items()}
    weights=reference.generate_weights(reference.WEIGHTS_SEED,device='cuda')
    states={name:module.prepare(weights) for name,module in modules.items()}
    # Compile the same launch parameters as each immutable candidate. No tuning.
    for name,bn in [('bn128',128),('bn32',32)]:
        for m in [128,256]:
            gate()
            data=reference.generate_input(m=m,layer='pooler',seed=2600000+m,device='cuda')
            x=data['x'];n,k=data['config']['n'],data['config']['k']
            out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            kernel=modules[name]._pooler_grouped.warmup(x,states[name]['pooler'],out,
                m,n,k,32,bn,64,8,num_warps=4,num_stages=3,
                grid=(triton.cdiv(m,32)*triton.cdiv(n,bn),))
            kernel._init_handles()  # Pattern documented by Triton's softmax tutorial.
            ptx=kernel.asm.get('ptx','')
            metadata={key:getattr(kernel.metadata,key,None) for key in
                      ['name','num_warps','num_stages','num_ctas','shared','maxnreg']}
            threads=32*metadata['num_warps']
            regs=kernel.n_regs;shared=metadata['shared']
            smem_sm=report['hardware']['shared_per_sm_bytes'] or 228*1024
            bounds=dict(registers=65536//(regs*threads) if regs else None,
                        shared_memory=smem_sm//shared if shared else 32,
                        threads=2048//threads,blocks=32)
            report['compiled'].append(dict(runner=name,m=m,n=n,k=k,bm=32,bn=bn,bk=64,
                metadata=metadata,n_regs_per_thread=regs,n_spills_driver_raw=kernel.n_spills,
                n_max_threads=kernel.n_max_threads,
                optimistic_resident_cta_bounds=bounds,
                occupancy_qualification="Static upper ceilings ignore allocation granularity, scheduling and other limits; NOT achieved occupancy",
                operand_requests=operand_requests(m,n,k,32,bn),
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),ptx=ptx,
                ttgir=kernel.asm.get('ttgir',''),**ptx_features(ptx)))
            save()
    eviction=torch.ones(256*1024*1024//4,device='cuda',dtype=torch.float32)
    for layer,m in [('pooler',128),('pooler',256),('qkv_gate',16),('qkv_gate',128),
                    ('unpooler',128),('unpooler',256)]:
        for name in (files if layer=='pooler' else ['best']):
            gate()
            data=reference.generate_input(m=m,layer=layer,seed=2700000+m,device='cuda')
            graph,out=capture(modules[name].kernel_fn,data,states[name])
            for _ in range(6):graph.replay()
            torch.cuda.synchronize()
            correctness=[]
            for replay in range(2):
                fresh=reference.generate_input(m=m,layer=layer,seed=2800000+m+replay,device='cuda')
                data['x'].copy_(fresh['x']);graph.replay()
                ok,message=reference.check_implementation(data,out,weights)
                correctness.append(dict(seed=2800000+m+replay,passed=ok,message=message))
            item=dict(runner=name,layer=layer,m=m,correctness=correctness)
            try:
                eviction.sum();torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=False,
                        with_stack=False,profile_memory=False) as prof:
                    graph.replay();torch.cuda.synchronize()
                item['cuda_activity_events']=[dict(name=e.name,duration_us=e.device_time_total)
                    for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                item['host_graph_launch_events']=[dict(name=e.name,cpu_duration_us=e.cpu_time_total)
                    for e in prof.events() if 'cudaGraphLaunch' in e.name]
                item['qualification']='Instrumented ordinary activity durations; not comparable to unprofiled score and never subtracted from graph times'
            except Exception as error:
                item['timeline_unavailable']=dict(type=type(error).__name__,message=str(error)[:1000],
                    action='No profiler retry, alternate counters or access bypass')
            report['timelines'].append(item);save()
            if not all(row['passed'] for row in correctness):return 1
    report['passed']=len(report['compiled'])==4 and len(report['timelines'])==10
    report['wall_seconds']=time.monotonic()-started
    save()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
