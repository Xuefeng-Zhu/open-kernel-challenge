"""Offline P050 six-load sensitivity; never a measured/predicted speedup."""
import csv
import hashlib
import json
from pathlib import Path
from projection_p041_plan import score, compulsory_model

ROOT=Path(__file__).resolve().parent
CSV=ROOT/'results/remote-P2-09-pooler-column-batch2-hybrid.csv'


def build():
    raw=list(csv.DictReader(CSV.open()))
    rows=[dict(load=r['load'],layer=r['layer'],m=int(r['m']),weight=int(r['weight']),
               candidate_us=float(r['candidate_us']),reference_us=float(r['sglang_us'])) for r in raw]
    assert len(rows)==30 and len({(r['load'],r['layer']) for r in rows})==30
    assert all(r['weight']==62 for r in rows)
    parent,loads=score(rows)
    targets=lambda r:r['layer']=='qkv_gate' and r['m']>1
    scenarios=[]
    for cut in (.05,.10,.20):
        changed=[{**r,'candidate_us':r['candidate_us']*(1-cut)} if targets(r) else r for r in rows]
        hypothetical,_=score(changed)
        scenarios.append(dict(uniform_target_latency_cut=cut,modeled_relative_score_gain=hypothetical/parent-1,modeled_score=hypothetical))
    observed_cut=0.03780396199333225
    scenarios.append(dict(label='P049 raw M128 cut hypothetically applied to M128 only',
        target_latency_cut=observed_cut,modeled_relative_score_gain=score([
        {**r,'candidate_us':r['candidate_us']*(1-observed_cut)} if targets(r) and r['m']==128 else r
        for r in rows])[0]/parent-1,qualification='Cross-environment raw-to-official-weight sensitivity, not an achieved score'))
    required=[]
    for gain in (.01,.015):
        lo,hi=0.,1.
        for _ in range(60):
            mid=(lo+hi)/2
            relative=score([{**r,'candidate_us':r['candidate_us']*(1-mid)} if targets(r) else r for r in rows])[0]/parent-1
            if relative<gain:lo=mid
            else:hi=mid
        required.append(dict(target_whole_score_gain=gain,required_uniform_target_cut=(lo+hi)/2))
    floors=[{**r,'candidate_us':compulsory_model(r['layer'],r['m'])['official_model_floor_us']} if targets(r) else r for r in rows]
    return dict(parent_job='41ba778763a5433b',parent_score=parent,receipt=str(CSV.relative_to(ROOT)),
      receipt_sha256=hashlib.sha256(CSV.read_bytes()).hexdigest(),target_count=sum(targets(r) for r in rows),
      scoring='Six equally weighted load ratios; five shape weights each62 inside each load',
      scenarios=scenarios,required_uniform_cuts=required,model_floor_relative_score_gain=score(floors)[0]/parent-1,
      target_cases=[dict(layer=r['layer'],m=r['m'],parent_us=r['candidate_us'],
                         hypothetical_10pct_cut_us=r['candidate_us']*.1) for r in rows if targets(r)],
      interpretation='Sensitivity only, holding all other values fixed. No benefit is predicted; source dispatch is not hardware proof. Floors omit launch/workspace costs and are not achievable forecasts.',gpu_executed=False)


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
