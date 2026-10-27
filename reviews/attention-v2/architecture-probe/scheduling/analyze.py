"""Bounded CPU scheduling model from recorded H200 inputs; no CUDA/client calls.

Run with the original workspace .venv/bin/python. The primary vectors are actual
historical CUDA-generated input lengths. Optional CPU _lengths samples are only
distribution examples: PyTorch CPU and CUDA RNG sequences need not coincide.
"""
from __future__ import annotations

import ast
import hashlib
import heapq
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FROZEN = {
    "mixed_fa3_fp8_qk.py": "e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e",
    "a215/paired-result.json": "92b5ba197689c3e17ec3c86cefdefdb1edd4bfe0222171b0bd63bd67f000e867",
    "official/reference.py": "386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b",
}
BN, CHUNK, HKV, SM = 64, 8192, 2, 132


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs(lengths, capacity, q_len=1):
    """CUDA grid convention x=b, y=kvh, z=split; order is a model, not a trace."""
    splits = math.ceil(capacity / CHUNK)
    return [dict(batch=b, kv_head=h, split=s,
                 historical_tiles=math.ceil(max(0, min(CHUNK, n-s*CHUNK))/BN),
                 owns_new_tokens=(s == splits-1))
            for s in range(splits) for h in range(HKV)
            for b, n in enumerate(lengths)]


def list_schedule(costs, slots):
    queue = [0] * slots
    heapq.heapify(queue)
    for cost in costs:
        ready = heapq.heappop(queue)
        heapq.heappush(queue, ready + cost)
    return max(queue, default=0)


def model(lengths, capacity):
    tasks = jobs(lengths, capacity)
    hist = [j['historical_tiles'] for j in tasks]
    full = sum(t == CHUNK//BN for t in hist)
    empty = [j for j in tasks if j['historical_tiles'] == 0]
    sweeps = []
    for residency in range(1, 6):
        slots = SM*residency
        for prologue in (0, 1, 4, 16):
            # Hypothetical unit costs; new-token ownership must never disappear.
            costs = [j['historical_tiles'] + prologue + int(j['owns_new_tokens'])
                     for j in tasks]
            fifo = list_schedule(costs, slots)
            lpt = list_schedule(sorted(costs, reverse=True), slots)
            total_work_bound = sum(costs)/slots
            full_task_bound = math.ceil(full/slots)*(CHUNK//BN+prologue)
            assert fifo >= full_task_bound and lpt >= full_task_bound
            sweeps.append(dict(resident_ctas_per_sm_assumed=residency,
                independent_slots=slots, prologue_tile_equivalents=prologue,
                new_token_tile_equivalents=1, fifo_makespan=fifo,
                longest_processing_first_makespan=lpt,
                total_work_lower_bound=total_work_bound,
                full_task_pigeonhole_lower_bound=full_task_bound,
                existing_chunks_improvement_upper_bound=max(0,1-full_task_bound/fifo),
                arbitrary_divisible_work_improvement_upper_bound=max(0,1-total_work_bound/fifo)))
    m = 16
    partial = m*(64+2)*4
    return dict(lengths=lengths, capacity_tokens=capacity, splits=math.ceil(capacity/CHUNK),
        ctas=len(tasks), historical_tiles=sum(hist), full_history_ctas=full,
        partial_history_ctas=sum(0<t<128 for t in hist), empty_history_ctas=len(empty),
        removable_empty_ctas=sum(not j['owns_new_tokens'] for j in empty),
        empty_history_split_ids=sorted({j['split'] for j in empty}),
        min_length=min(lengths), max_length=max(lengths), mean_length=sum(lengths)/len(lengths),
        partial_bytes_per_cta_m16=partial, partial_bytes_total_m16=len(tasks)*partial,
        history_tile_jobs_if_one_partial_per_tile=sum(hist),
        partial_bytes_if_one_per_history_tile=sum(hist)*partial,
        model=sweeps)


def main():
    provenance = {}
    for rel, expected in FROZEN.items():
        actual = digest(BASE/rel)
        assert actual == expected, (rel, actual)
        provenance[rel] = actual
    receipt = json.loads((BASE/'a215/paired-result.json').read_text())
    actual = []
    for case in receipt['cases']:
        if case['case']['name'] != 'full_decode32_scored_family':
            continue
        for label, seed in [('A1',135701), ('B',135702)]:
            record = next(r for r in case['records'] if r['runner']=='a213' and r['label']==label)
            row = model(record['actual_lengths'], 86001)
            row.update(label=label, seed=seed, case=case['case'],
                provenance='Actual H200 CUDA input lengths recorded by prior paired worker; not regenerated')
            assert row['ctas']==704 and row['removable_empty_ctas']==0
            assert all(x['existing_chunks_improvement_upper_bound']==0 and
                       x['longest_processing_first_makespan']==x['full_task_pigeonhole_lower_bound']
                       for x in row['model'])
            actual.append(row)
    assert len(actual)==2
    # Check empty history still owns the new token for the last split. This is
    # why a scheduler must not prune a task solely from historical length.
    zero = jobs([0],86001)
    assert sum(x['owns_new_tokens'] for x in zero)==HKV
    assert sum(x['historical_tiles'] for x in zero)==0

    import torch
    tree = ast.parse((BASE/'official/reference.py').read_text())
    defs = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_lengths','_longest')]
    namespace = dict(torch=torch, math=math, AGENTIC_SPREAD=0.075)
    exec(compile(ast.Module(body=defs,type_ignores=[]),'official/reference.py::_lengths','exec'), namespace)
    fixtures = json.loads((BASE/'fixtures.json').read_text())
    cpu = []
    for case in fixtures['benchmarks']:
        if case['layer']!='full' or case['batch'] not in (16,32):
            continue
        gen = torch.Generator(device='cpu').manual_seed(case['seed'])
        values = namespace['_lengths'](gen,case['batch'],case['ctx'],case['spread'],'cpu').tolist()
        cpu.append(dict(case=case,lengths=values,device='cpu',
            qualification='Exact saved official length function on CPU. These are not measured CUDA vectors or official timing repetitions.'))
    output = dict(kind='Local cost/scheduling model; no new GPU execution or performance result',
        source_sha256=provenance, torch_cpu_version=torch.__version__,
        constants=dict(block_n=BN, chunk_tokens=CHUNK, kv_heads=HKV, h200_sm_count=SM),
        primary_actual_cuda_vectors=actual, supplementary_cpu_generator_samples=cpu,
        assumptions=['Grid order x-fastest used for idealized FIFO; actual dispatch order and CTA trace unknown',
          'Each SM assumed to provide 1..5 independent identical service slots; real occupancy/residency unknown',
          'Service costs measured in hypothetical historical-64-token-tile units; no instruction or wall-time model',
          'No measured prologue, page-table gather, assign/descriptor, partial/merge, synchronization or bandwidth costs',
          'One new-token unit charged to last split; prologue sensitivity 0,1,4,16 units, not fitted measurements',
          'Total-work bound requires arbitrarily divisible work and excludes extra planner/aggregation work'],
        checks_passed=['Frozen parent/receipt/reference hashes', 'Both actual vectors and 704 task counts',
          'No empty-history task removable in scored family', 'All 40 model cases reach fixed-full-task pigeonhole bound',
          'Zero-history last-split new-token ownership', 'Official CPU function extraction on four saved cases'])
    (HERE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(actual_cases=len(actual),cpu_examples=len(cpu),checks='passed',
                         result_sha256=digest(HERE/'results.json')),indent=2))


if __name__=='__main__':
    main()
