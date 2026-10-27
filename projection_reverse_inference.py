"""P055 offline score requirements; no inference of competitor implementation."""
import argparse
import json
from pathlib import Path
import projection_gap_model as model


def build(root=Path(__file__).resolve().parent):
    rows=model.read_rows(root)
    selected,loads=model.score(rows)
    public=json.loads((root/'results/router-p055-public-leader.json').read_text())
    target=public['leader']['speedup']
    fractions={load:{f:sum(r['weight']*r['candidate_us'] for r in rows if r['load']==load and r['layer']==f)/total['candidate_us']
        for f in model.FAMILIES} for load,total in loads.items()}
    groups=[model.MAIN,('qkv_gate','unpooler'),('pooler',),('o','router')]
    scenarios=[dict(families=group,minimum_m=minimum_m,cut=cut,
                   modeled_score=model.score(model.change(rows,group,cut,minimum_m))[0],
                   score_gain=model.score(model.change(rows,group,cut,minimum_m))[0]/selected-1)
        for group in groups for minimum_m in (1,2) for cut in (.1,.2,.3)]
    # Scaling every latency is analytic; a 100% all-path boundary would make
    # the denominator zero and is not a valid point for the subset solver.
    required={'all_paths':1-selected/target}
    required.update({label:model.required_cut(rows,group,target,minimum_m) for label,group,minimum_m in
        [('three_main_all_m',model.MAIN,1),('three_main_m_gt_1',model.MAIN,2)]})
    for label,group,minimum_m in [('all_paths',model.FAMILIES,1),('three_main_all_m',model.MAIN,1),('three_main_m_gt_1',model.MAIN,2)]:
        assert abs(model.score(model.change(rows,group,required[label],minimum_m))[0]-target)<1e-10
    return dict(kind='P055 conditional score arithmetic, not achieved speedups',observed_utc=public['observed_utc'],
        selected_score=selected,leader=public['leader'],leader_over_selected=target/selected,
        required_latency_cuts=required,per_load_fractions=fractions,
        equal_load_mean_fractions={f:sum(v[f] for v in fractions.values())/6 for f in model.FAMILIES},
        scenarios=scenarios,source_sha256={p:model.sha(root/p) for p in (model.CSV,'results/router-p055-public-leader.json')},
        limitations=['Public leaderboard rounded and provisional; not a same-run paired comparison',
            'Each hypothetical holds all untargeted paths fixed; no competitor architecture inferred',
            'No new GPU run, official check or score'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected_score','leader','required_latency_cuts','equal_load_mean_fractions')},indent=2))
