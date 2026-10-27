"""Offline P043 frozen score/storage model; no GPU speed claim."""
import hashlib,json,math
from pathlib import Path
from projection_p041_plan import ROOT,read_rows,score

def build():
 rows=read_rows();old,loads=score(rows)
 targets=[r for r in rows if r['layer']=='pooler' and r['m'] in (8,16,32)]
 storage=json.loads((ROOT/'results/router-p043-codec-storage-cpu.json').read_text())
 reduction=min(r['byte_reduction'] for r in storage['results'])
 fractions={str(r['m']):r['weight']*r['candidate_us']/loads[r['load']]['candidate_us'] for r in targets}
 scenarios=[]
 for cut in (.05,.08,.10,.20,reduction):
  factor=math.prod(1-f*cut for f in fractions.values())**(-1/6)
  scenarios.append(dict(uniform_target_latency_cut=cut,whole_score_factor=factor,score=old*factor))
 # Historical raw R06 was not the exact new M-tiled control and used a different parent.
 penalty=1.0649883974/.9269575464
 return dict(kind='Conditional storage/arithmetic/score model, not measured performance',
  parent_score=old,parent_job='d5636ac8d1404abd',targets=targets,fractions=fractions,
  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
   [ROOT/'candidates/projection_pooler_lossless_codec.py',ROOT/'candidates/projection_pooler_lossless_raw_control.py']},
  all_stream_byte_reduction_lower_of_three_seeds=reduction,scenarios=scenarios,
  historical_R06_m8_normalized_penalty=penalty,
  historical_R06_m8_own_latency_cut_to_break_even=1-1/penalty,
  optimistic_m8_cut_if_all_raw_time_scales_with_weight_bytes_and_zero_decode_cost=1-penalty*(1-reduction),
  geometry=dict(splits=8,BM=16,BN=64,BK=64,warps=4,stages=2),
  source_cost_model=dict(normal_weight_element='one sign/mantissa uint8 load plus shared packed-delta byte load, exponent add, variable shift, masks/OR, exact uint16-to-BF16 bitcast',
    escape_weight_element='one sign/mantissa uint8 load and one full exponent uint8 load; delta read masked off',
    block_metadata='uint8 base plus int32 escape index per N row per BK64 iteration; K256 metadata can be reread four times; L2 reuse not assumed',
    partial_scratch_bytes={str(m):8*m*4096*4 for m in (8,16,32)},
    note='Scratch is written then read; M32 has two M tiles and rereads weights. Logical loads are not physical DRAM transactions or stall evidence.'),
  gates=dict(encoding='all BF16 patterns bit exact; actual full-weight reduction>=22% or P039 fallback',
   correctness='all9 prerequisites and30 scored shapes pass official oracle; preserve precision and original weights',
   candidate='each target normalized latency reduction>=10%, no target regression>5%, target-only frozen-parent modeled score gain>=1.5%',
   controls='report every unchanged path shift>5%; large drift/overlapping aggregate intervals forbid causal overall-gain claims',
   retain='never select on aggregate point alone; failure or gate miss preserves P039; no rescue sweep'),
  limitations=['Current raw matched control is only locally tested; official single candidate check cannot isolate compression causality',
   'No native compiler resources/PTX/DRAM counters for this candidate; decode issue/register costs unknown',
   'Ideal uniform storage-proportional scenario is optimistic conditional value, not a physical upper bound on all possible schedules',
   'CPU same-address A-B-A is not captured CUDA graph or overlapping-stream evidence'])

if __name__=='__main__':
 report=build();(ROOT/'results/router-p043-admission-model-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'parent_score':report['parent_score'],'scenarios':report['scenarios'],'target_fractions':report['fractions']}))
