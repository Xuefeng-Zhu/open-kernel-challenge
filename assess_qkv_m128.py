"""Offline description of M36; no network, GPU or invented official score."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def main():
    report=json.loads((ROOT/'results/remote-modal-M36-qkv-m128-singlepass.json').read_text())
    assert report['attempt']['correctness_and_graph_passed']
    target=next(x for x in report['summaries'] if x['case']['layer']=='qkv_gate' and x['case']['m']==128)
    loads=report['aggregate']['loads']
    matched=[]
    for i in range(15):
        pairs=[load['opposite_order_pairs'][i] for load in loads]
        assert all(p['indices']==pairs[0]['indices'] for p in pairs)
        matched.append(math.exp(statistics.mean(math.log(p['geometric_control_over_candidate']['best'])
                                                for p in pairs)))
    target_wins=sum(p['geometric_control_over_candidate']['best']>1 for p in target['opposite_order_pairs'])
    result=dict(kind='Descriptive offline analysis, no official score/CI',
                source_commit='26189baa01e72d6e629f130a85f776347ca15a3f',
                source_sha256=report['attempt']['submission_sha256'],
                app_id=report['attempt']['app_id'],
                candidate=target['gross_us_distribution']['candidate'],
                best=target['gross_us_distribution']['best'],
                target_paired_best_over_candidate=target['median_paired_gross_ratio']['best'],
                target_winning_pairs=target_wins,
                aggregate_gm_of_load_median_ratios=report['aggregate']['geometric_mean_over_loads_control_over_candidate']['best'],
                descriptive_matched_pair_gm=matched,
                descriptive_matched_pair_median=statistics.median(matched),
                descriptive_matched_pair_wins=sum(x>1 for x in matched),
                qualification='Matched-pair aggregation is a post-hoc descriptive view, not a confidence interval or new promotion criterion',
                meets_predeclared_target_gate=target_wins>=13,
                decision='Reject; no official check',
                compiled={k:v for k,v in report['result']['records']['nearby-correctness.json']['candidate_compiled'].items() if k!='ptx'})
    assert not result['meets_predeclared_target_gate']
    (ROOT/'results/modal-M36-screen-assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
