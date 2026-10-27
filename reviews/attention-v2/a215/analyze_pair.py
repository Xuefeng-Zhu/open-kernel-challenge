"""Audit saved paired evidence and label the six-load projection as a model."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def main():
    raw = (HERE/'paired-result.json').read_bytes()
    run = json.loads((HERE/'paired-modal-run.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == run['result_sha256']
    r = json.loads(raw)
    assert r['passed'] and r['correctness_complete'] and r['timing_complete']
    assert len(r['cases']) == 2 and all(len(c['records']) == 6 for c in r['cases'])
    checks = [x for c in r['cases'] for x in c['records']]
    assert all(x['passed'] and x['stable_addresses'] and x['immutable_inputs_and_untouched_rows'] for x in checks)
    assert sum(x['official_oracle_called'] for x in checks) == 8
    assert sum(x['exact_output_newkv_return'] is True for x in checks) == 4
    t = r['timing']; samples = t['samples']; warmups = t['warmups']
    assert len(samples) == 12 and len(warmups) == 4
    assert all(x['passed'] for row in samples+warmups for x in row['checks'].values())
    ratios = [x['raw_us']['a213']/x['raw_us']['a215'] for x in samples]
    balanced = []
    for i in range(0, 12, 2):
        assert samples[i]['order'] == list(reversed(samples[i+1]['order']))
        assert samples[i]['input_version'] == samples[i+1]['input_version']
        balanced.append(math.sqrt(ratios[i]*ratios[i+1]))
    ratio = statistics.median(balanced)
    cut = 1-1/ratio
    offset = statistics.median(t['calibration_us'])
    adjusted = [{n: row['raw_us'][n]-offset for n in ['a213','a215']} for row in samples]
    assert all(v > 0 for row in adjusted for v in row.values())
    adj_ratios = [row['a213']/row['a215'] for row in adjusted]
    adj_pair = [math.sqrt(adj_ratios[i]*adj_ratios[i+1]) for i in range(0,12,2)]
    history = json.loads((HERE.parent/'strategy/weighted_analysis.json').read_text())
    d32 = [x for x in history['comparison']['shapes'] if x['load']=='decode 32']
    full = next(x for x in d32 if x['layer']=='full')
    full_share = full['weight']*full['candidate_adjusted_us']/sum(x['weight']*x['candidate_adjusted_us'] for x in d32)
    factor = (1/(1-full_share*cut))**(1/6)
    result = dict(kind='Offline paired audit; hypothetical aggregate is not an achieved score',
        result_sha256=hashlib.sha256(raw).hexdigest(), app_id=run['app_id'],
        graph_captures=4, checked_aba_replays=12, official_oracle_calls=8,
        exact_A_return_checks=4, timed_warmup_calls=8, measured_calls=24,
        timing_calls_checked_against_oracle_validated_bytes=32,
        raw_median_us={n:statistics.median(x['raw_us'][n] for x in samples) for n in ['a213','a215']},
        raw_min_max_us={n:[min(x['raw_us'][n] for x in samples),max(x['raw_us'][n] for x in samples)] for n in ['a213','a215']},
        raw_ratio_parent_over_candidate=ratio, target_latency_reduction_percent=100*cut,
        sample_wins=sum(x>1 for x in ratios), reverse_order_pair_ratios=balanced,
        empty_graph_median_us=offset,
        calibrated_median_us={n:statistics.median(x[n] for x in adjusted) for n in ['a213','a215']},
        calibrated_ratio_parent_over_candidate=statistics.median(adj_pair),
        predeclared_target_latency_reduction_percent=10, predeclared_target_met=cut>=.10,
        model=dict(basis='Transfer same relative target cut to historical official A2-13; all other paths fixed',
            historical_full_D32_load_share=full_share, equation='(1/(1-full_share*target_cut))**(1/6)',
            aggregate_factor=factor, aggregate_gain_percent=100*(factor-1),
            achieved_score=False, caveat='Different environment and calibration; one session, two fixtures, no controls or independent repeat'),
        decision='Retain as modest promising experiment; do not replace selected A2-11 or claim official improvement')
    (HERE/'paired-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
