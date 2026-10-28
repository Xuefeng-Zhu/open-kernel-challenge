"""Pristine v298 serial graph replay stress; no scoring or concurrent execution."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import weakref
import helpers as h

SOURCES = dict(h.SOURCES, **{'helpers.py':'7fa92eba152f715c995b312b475cf54195aad5ca5a94ddc825727c3bfcd2262f'})
FAMILIES = (
 dict(name='single-decode-full',batch=1,q_len=1,ctx=80000,heads='tp8',layer='full',page=1,spread='fixed',dist='coherent'),
 dict(name='flat-decode-full',batch=16,q_len=1,ctx=80000,heads='tp8',layer='full',page=1,spread='agentic',dist='coherent'),
 dict(name='single-verify-full',batch=1,q_len=8,ctx=80000,heads='tp8',layer='full',page=1,spread='fixed',dist='coherent'),
 dict(name='flat-verify-full',batch=16,q_len=8,ctx=80000,heads='tp8',layer='full',page=1,spread='agentic',dist='coherent'),
 dict(name='single-decode-sliding',batch=1,q_len=1,ctx=6000,heads='tp8',layer='sliding',page=1,spread='fixed',dist='edges'),
 dict(name='verify-sliding',batch=16,q_len=8,ctx=6000,heads='tp8',layer='sliding',page=1,spread='agentic',dist='soft'),
)
GROWTH = dict(batch=33,q_len=8,ctx=64,heads='dp',layer='full',page=64,spread='fixed',dist='normal')
V_HISTORY_AMPLITUDE = 0.25  # Fixed robustness fixture scaling, never a timing/scoring input.
USED = ('qkv','seq_lens','cache_loc','k_cache','v_cache','page_table','k_scale','v_scale')

def bind_sources(root):
    found={}
    for name,want in SOURCES.items():
        p=root/name
        if p.is_symlink() or not p.is_file() or h.sha(p)!=want: raise ValueError('Source binding failed: '+name)
        found[name]=want
    return found


def metadata_cpu(table,lengths,length,page):
    """Pure-CPU model of the exact live metadata construction used below."""
    if not table or len(table)!=len(lengths) or length not in (1,8) or page<=0: raise ValueError('Invalid metadata geometry')
    width=len(table[0]);flat=[v for row in table for v in row]
    if any(len(row)!=width for row in table) or sorted(flat)!=list(range(len(flat))): raise ValueError('Whole-pool page bijection required')
    pos=[];loc=[]
    for row,n in zip(table,lengths):
        if type(n) is not int or n<0 or n+length>width*page: raise ValueError('Length outside fixed capacity')
        for j in range(length):
            q=n+j;pos.append(q);loc.append(row[q//page]*page+q%page)
    if len(set(loc))!=len(loc):raise ValueError('New cache writes alias')
    return dict(positions=pos,cache_loc=loc,cache_seqlens=[n+length for n in lengths],cu_seqlens_q=list(range(0,(len(table)+1)*length,length)))


def length_pattern(batch,capacity,length,variant):
    maximum=capacity-length
    choices=list(dict.fromkeys(min(maximum,n) for n in (0,1,63,64,65,2047,2048,2049,maximum)))
    return [choices[(variant+i)%len(choices)] for i in range(batch)]


def variant(torch,base,number):
    fresh=h.clone(torch,base);cfg=fresh['config'];device=fresh['qkv'].device
    gen=torch.Generator(device=device).manual_seed(136000+number)
    fresh['qkv'].copy_(torch.randn(fresh['qkv'].shape,generator=gen,device=device).to(fresh['qkv'].dtype))
    scale=1.0 if number%2==0 else 1/64
    for cache,key in (('k_cache','k_scale'),('v_cache','v_scale')):
        amplitude=V_HISTORY_AMPLITUDE if cache=='v_cache' else 1.0
        encoded=(base[cache].float()*base[key]*amplitude/scale).to(base[cache].dtype)
        if not bool(torch.isfinite(encoded.float()).all()):raise ValueError('Finite cache representation required')
        fresh[cache].copy_(encoded);fresh[key].fill_(scale)
    cfg['kv_scale']=scale
    # Full decode retains its correct step-shared FA3 scheduler metadata; v298 does not use it.
    mutate_lengths=cfg['q_len']==8 or cfg['window']>0
    if mutate_lengths:
        batch,width=fresh['page_table'].shape;page=cfg['page_size'];length=cfg['q_len']
        lens=length_pattern(batch,width*page,length,number)
        table=torch.randperm(batch*width,generator=gen,device=device).to(torch.int32).view(batch,width)
        n=torch.tensor(lens,dtype=torch.int32,device=device)
        positions=n.long()[:,None]+torch.arange(length,device=device)
        loc=torch.gather(table.long(),1,positions//page)*page+positions%page
        if len(torch.unique(loc))!=batch*length:raise ValueError('Runtime new cache writes alias')
        fresh['page_table'].copy_(table);fresh['seq_lens'].copy_(n)
        fresh['positions'].copy_(positions.reshape(-1));fresh['cache_loc'].copy_(loc.reshape(-1))
        fresh['cache_seqlens'].copy_(n+length)
        fresh['cu_seqlens_q'].copy_(torch.arange(0,(batch+1)*length,length,dtype=torch.int32,device=device))
    return fresh


def restore(torch,work,fresh):
    old=work['config'];new=fresh['config']
    if {k:v for k,v in old.items() if k!='kv_scale'}!={k:v for k,v in new.items() if k!='kv_scale'}:raise ValueError('Captured launch configuration changed')
    work['config']=dict(new)  # kv_scale is descriptive; v298 only consumes the live GPU scale tensors.
    h.restore(torch,work,fresh)


def used_pointers(torch,data):
    return {k:(data[k].data_ptr(),tuple(data[k].shape),tuple(data[k].stride()),str(data[k].dtype)) for k in USED}


def scratch_snapshot(torch,module,work):
    cfg=work['config'];plan=h.native_plan(module,work)
    actual=dict(plan['base_shape_plan']);actual.update(plan['default_tuned_override'] or {})
    rows=cfg['batch']*(cfg['num_kv_heads']//2)*actual['nc']*cfg['q_len']*16
    ws=module._WS[str(work['qkv'].device)]
    zero_acc=bool(torch.count_nonzero(ws[0][:rows*64])==0)
    zero_den=bool(torch.count_nonzero(ws[1][:rows])==0)
    if not zero_acc or not zero_den:raise ValueError('Used FP32 accumulator/denominator workspace was not reset')
    # Counters carry epochs, so record rather than demand zero.
    return dict(active_rows=rows,accumulator_zero=zero_acc,denominator_zero=zero_den,
                pointers=[x.data_ptr() for x in ws],numel=[x.numel() for x in ws],
                counter_epochs=ws[2][[24576,30020,30021,30022]].cpu().tolist(),plan=plan)


def unchanged_other_outputs(torch,ref,slots,except_slot):
    count=0
    for key,s in slots.items():
        if key==except_slot or 'last_raw' not in s:continue
        if not h.bytes_equal(torch,ref.collect(s['work'],s['out']),s['last_raw']):
            raise ValueError('Another retained graph output/new-cache was clobbered')
        count+=1
    return count


def run(root,output,seconds):
    start=time.monotonic();report=dict(status='running',worker_sha256=h.sha(__file__),verified_hashes=bind_sources(root),graph_cases=[],cross_shape_checks=[],optional_growth={},correctness_complete=False,candidate_changed=False,official_check=False,ranked_submission=False,concurrent_execution=False,
        protocol=dict(serial_only=True,main_families=6,independent_graphs_each=2,replay_order=['A','B','A','A','B','B','A','B'],bursts_each=16,performance_measurement=False,original_oracle=True,v_history_amplitude=V_HISTORY_AMPLITUDE,growth='Unscored API-supported extension only; no claim that the contest requires B33 DP'))
    def phase(name):
        report['phase']=name;report['elapsed_seconds']=time.monotonic()-start;h.save(output,report);print('PROGRESS='+name,flush=True)
        if report['elapsed_seconds']>seconds:raise TimeoutError('Worker deadline: '+name)
    try:
        import torch
        import triton
        ref=h.load(root,'reference','reference.py');module=h.load(root,'v298','v298.py')
        props=torch.cuda.get_device_properties(0)
        if 'H200' not in props.name or (props.major,props.minor)!=(9,0):raise ValueError('Expected H200 SM90')
        report['runtime']=dict(gpu=props.name,sm_count=props.multi_processor_count,torch=torch.__version__,triton=triton.__version__)
        phase('native-build');built=module._module();report['runtime']['binary_sha256']=h.sha(built.__file__)
        retained=[]
        for fi,spec in enumerate(FAMILIES):
            phase('capture-'+spec['name']);base=ref.generate_input(**{k:v for k,v in spec.items() if k!='name'},seed=135950+fi,device='cuda')
            row=dict(name=spec['name'],spec=spec,captures=[],serial_checks=[],burst_checks=[])
            report['graph_cases'].append(row);slots={}
            for slot,number in [('A',20+fi*2),('B',21+fi*2)]:
                fresh=variant(torch,base,number)
                work,state,graph,out=h.make_graph(torch,module.kernel_fn,fresh)
                slots[slot]=dict(work=work,state=state,used=used_pointers(torch,work),graph=graph,out=out)
                restore(torch,work,fresh);graph.replay();check=h.oracle(torch,ref,work,fresh,out,state)
                count=unchanged_other_outputs(torch,ref,slots,slot)
                slots[slot]['last_raw']=ref.collect(work,out).clone()
                row['captures'].append(dict(slot=slot,oracle=check,output_pointer=out.data_ptr(),other_outputs_unchanged=count,scratch=scratch_snapshot(torch,module,work)))
                del fresh
            if slots['A']['out'].data_ptr()==slots['B']['out'].data_ptr():raise ValueError('Independent graph output alias')
            for index,slot in enumerate(report['protocol']['replay_order']):
                phase('serial-'+spec['name']+'-'+str(index));fresh=variant(torch,base,100+fi*10+index)
                s=slots[slot];restore(torch,s['work'],fresh);s['graph'].replay()
                if used_pointers(torch,s['work'])!=s['used']:raise ValueError('Captured used input pointers changed')
                check=h.oracle(torch,ref,s['work'],fresh,s['out'],s['state'])
                count=unchanged_other_outputs(torch,ref,slots,slot);s['last_raw']=ref.collect(s['work'],s['out']).clone()
                row['serial_checks'].append(dict(slot=slot,variant=100+fi*10+index,oracle=check,other_outputs_unchanged=count,scales=[float(fresh['k_scale']),float(fresh['v_scale'])],seq_lens=fresh['seq_lens'].cpu().tolist(),scratch=scratch_snapshot(torch,module,s['work'])))
                del fresh
            # Same-stream bursts; no forced host synchronization between enqueued replays.
            for slot,s in slots.items():
                fresh=variant(torch,base,300+fi*2+(slot=='B'));restore(torch,s['work'],fresh)
                for _ in range(16):s['graph'].replay()
                check=h.oracle(torch,ref,s['work'],fresh,s['out'],s['state'])
                count=unchanged_other_outputs(torch,ref,slots,slot);s['last_raw']=ref.collect(s['work'],s['out']).clone()
                row['burst_checks'].append(dict(slot=slot,replays=16,oracle=check,other_outputs_unchanged=count,scratch=scratch_snapshot(torch,module,s['work'])))
                del fresh
            retained.append(dict(base=base,slots=slots,row=row))
            phase('complete-'+spec['name'])
        # All graph pairs remain live while shapes take turns sharing the same scratch.
        schedule=[0,3,0,5,1,4,2,3,2,5,4,1]
        for index,family in enumerate(schedule):
            phase('cross-shape-'+str(index));r=retained[family];slot='A' if index%2==0 else 'B';s=r['slots'][slot]
            fresh=variant(torch,r['base'],500+index);restore(torch,s['work'],fresh);s['graph'].replay()
            check=h.oracle(torch,ref,s['work'],fresh,s['out'],s['state'])
            others={f'{ri}-{name}':x for ri,rr in enumerate(retained) for name,x in rr['slots'].items()}
            excluded=f'{family}-{slot}';count=unchanged_other_outputs(torch,ref,others,excluded);s['last_raw']=ref.collect(s['work'],s['out']).clone()
            report['cross_shape_checks'].append(dict(family=r['row']['name'],slot=slot,oracle=check,other_outputs_unchanged=count,scratch=scratch_snapshot(torch,module,s['work'])))
            del fresh
        report['serial_complete']=True;phase('main-serial-complete')
        # Growth is an optional, separately labelled API stress, after all scored-family checks.
        old=retained[3]['slots']['A'];device=str(old['work']['qkv'].device)
        report['optional_growth']=dict(status='running',spec=GROWTH,claimed_contest_requirement=False,
            old_workspace_pointers=[x.data_ptr() for x in module._WS[device]],old_workspace_numel=[x.numel() for x in module._WS[device]])
        old_weak=[weakref.ref(x) for x in module._WS[device]]  # Do not pin the old scratch Tensor objects.
        phase('optional-growth-capture');large=ref.generate_input(**GROWTH,seed=136777,device='cuda')
        lw,ls,lg,lo=h.make_graph(torch,module.kernel_fn,large);restore(torch,lw,large);lg.replay()
        growth=report['optional_growth'];growth['oracle']=h.oracle(torch,ref,lw,large,lo,ls)
        growth['new_workspace_pointers']=[x.data_ptr() for x in module._WS[device]];growth['new_workspace_numel']=[x.numel() for x in module._WS[device]]
        if growth['new_workspace_pointers']==growth['old_workspace_pointers']:raise ValueError('Optional workspace growth not achieved')
        gc.collect();growth['old_python_scratch_alive']=[w() is not None for w in old_weak]
        # Ordinary new valid inputs create allocator activity without explicitly poisoning memory.
        pressure=ref.generate_input(batch=16,q_len=8,ctx=1024,heads='tp8',layer='full',page=1,seed=136778,device='cuda')
        phase('optional-old-graph-after-growth');fresh=variant(torch,retained[3]['base'],800)
        restore(torch,old['work'],fresh);old['graph'].replay()
        growth['old_graph_oracle']=h.oracle(torch,ref,old['work'],fresh,old['out'],old['state'])
        growth['old_used_pointers_unchanged']=used_pointers(torch,old['work'])==old['used']
        if not growth['old_used_pointers_unchanged']:raise ValueError('Old graph inputs moved')
        growth['status']='pass';report['correctness_complete']=True;report['status']='complete';phase('complete')
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());h.save(output,report);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start;h.save(output,report)


def main():
    p=argparse.ArgumentParser();p.add_argument('--payload-root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=int,default=540);a=p.parse_args()
    if not 0<a.deadline<=540:raise ValueError('Deadline must be positive and <=540s')
    run(a.payload_root.resolve(),a.output.resolve(),a.deadline)

if __name__=='__main__':main()
