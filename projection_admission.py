"""Offline frozen-timing scenario, never a new GPU benchmark or predicted score."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/remote-P2-02-full-composition.csv'


def main():
    rows=list(csv.DictReader(SOURCE.open()))
    assert len(rows)==30 and {r['status'] for r in rows}=={'pass'}
    loads=sorted({r['load'] for r in rows});assert len(loads)==6
    totals={l:sum(int(r['weight'])*float(r['candidate_us']) for r in rows if r['load']==l) for l in loads}
    baseline={l:sum(int(r['weight'])*float(r['sglang_us']) for r in rows if r['load']==l) for l in loads}
    score=math.exp(sum(math.log(baseline[l]/totals[l]) for l in loads)/6)
    assert math.isclose(score,1.1486499523461586,rel_tol=1e-8)
    target=next(r for r in rows if r['layer']=='pooler' and int(r['m'])==256)
    latency=float(target['candidate_us']);share=int(target['weight'])*latency/totals[target['load']]
    def ratio(reduction):return (1/(1-share*reduction))**(1/6)
    required=(1-1/(1.015**6))/share
    scenarios=[dict(assumed_target_latency_reduction=r,assumed_latency_us=latency*(1-r),
                    frozen_scenario_relative_score=ratio(r)) for r in [.10,.20,required]]
    print(json.dumps(dict(method='Frozen official weighted-load scenario only: change pooler m256 latency, hold all other times fixed. Not a prediction or new measured score.',
        job_id='48b1357016134e52',receipt_csv_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        reproduced_official_score=score,target_latency_us=latency,target_share_of_verify32_load=share,
        target_latency_reduction_for_1_5pct_aggregate=required,scenarios=scenarios,
        optimistic_floor_scenario=dict(published_floor_us=17.4,relative_score=ratio(1-17.4/latency)),
        new_gpu_experiments=0,new_official_checks=0),indent=2))


if __name__=='__main__':main()
