"""Offline P049 six-load sensitivity; never a measured/predicted speedup."""
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
    targets=lambda r:r['layer'] in ('qkv_gate','unpooler') and r['m']>1
    scenarios=[]
    for cut in (.05,.10,.20):
        changed=[{**r,'candidate_us':r['candidate_us']*(1-cut)} if targets(r) else r for r in rows]
        hypothetical,_=score(changed)
        scenarios.append(dict(uniform_target_latency_cut=cut,modeled_relative_score_gain=hypothetical/parent-1,modeled_score=hypothetical))
    floors=[{**r,'candidate_us':compulsory_model(r['layer'],r['m'])['official_model_floor_us']} if targets(r) else r for r in rows]
    return dict(parent_job='41ba778763a5433b',parent_score=parent,receipt=str(CSV.relative_to(ROOT)),
      receipt_sha256=hashlib.sha256(CSV.read_bytes()).hexdigest(),target_count=sum(targets(r) for r in rows),
      scoring='Six equally weighted load ratios; five shape weights each62 inside each load',
      scenarios=scenarios,model_floor_relative_score_gain=score(floors)[0]/parent-1,
      target_cases=[dict(layer=r['layer'],m=r['m'],parent_us=r['candidate_us'],
                         hypothetical_10pct_cut_us=r['candidate_us']*.1) for r in rows if targets(r)],
      interpretation='Sensitivity only, holding all other values fixed. No benefit is predicted; source dispatch is not hardware proof. Floors omit launch/workspace costs and are not achievable forecasts.',gpu_executed=False)


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
