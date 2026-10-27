"""Frozen-evidence composition sensitivity; no GPU or official actions."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
EVIDENCE_HASHES = {
    'fusion-lookahead-probe/baseline.py': '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0',
    'fusion-lookahead-probe/candidate.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'fused-preparation-probe/candidate.py': '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0',
    'fused-preparation-probe/official-check/official-result.json': '46f9818a029a68995d6bad8b99c79a72f390ab09e464c01eb9a79b64bb160001',
    'mixed_fa3_page_lookahead.py': 'fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c',
    'mixed_fa3_fp8_qk.py': 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',
    'a215/paired-result.json': '92b5ba197689c3e17ec3c86cefdefdb1edd4bfe0222171b0bd63bd67f000e867',
    'a215/paired-analysis.json': 'a6ee815651dadc1172b2c9663b424d532235b9b4fe3be389f3349ba63e7dfbef',
}


def evidence(root=V2):
    observed = {}
    for name, expected in EVIDENCE_HASHES.items():
        observed[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if observed[name] != expected:
            raise ValueError('Frozen composition evidence mismatch: '+name)
    official = json.loads((root/'fused-preparation-probe/official-check/official-result.json').read_bytes())
    prior = json.loads((root/'a215/paired-result.json').read_bytes())
    assert official['id'] == '2f5f1fb54b48486d' and official['mode'] == 'check'
    assert official['status'] == 'done' and official['result']['passed']
    assert prior['passed'] and prior['correctness_complete'] and prior['timing_complete']
    assert prior['source_sha256'] == {'a213': EVIDENCE_HASHES['mixed_fa3_fp8_qk.py'],
                                     'a215': EVIDENCE_HASHES['mixed_fa3_page_lookahead.py']}
    return official, prior, observed


def score_factor(full_share, relative_target_cut):
    if not math.isfinite(full_share) or not 0 < full_share < 1:
        raise ValueError('Full-path share must lie strictly within a load')
    if not math.isfinite(relative_target_cut) or 1-full_share*relative_target_cut <= 0:
        raise ValueError('Nonpositive or nonfinite resulting load')
    return (1-full_share*relative_target_cut)**(-1/6)


def required_cut(full_share, gain):
    if not math.isfinite(gain) or gain <= -1:
        raise ValueError('Invalid aggregate gain')
    return (1-(1+gain)**(-6))/full_share


def build(root=V2):
    official, prior, hashes = evidence(root)
    rows = [row for row in official['result']['records'] if row['event'] == 'benchmark']
    assert len(rows) == 12
    by_key = {(x['case']['load'], x['case']['layer']): x for x in rows}
    assert len(by_key) == 12
    factors = []
    for load in sorted({key[0] for key in by_key}):
        ref = sum(by_key[(load, layer)]['baseline_ns']*weight for layer, weight in [('sliding',46),('full',16)])
        cand = sum(by_key[(load, layer)]['median_ns']*weight for layer, weight in [('sliding',46),('full',16)])
        factors.append(ref/cand)
    reconstructed = math.exp(statistics.mean(math.log(x) for x in factors))
    assert math.isclose(reconstructed, official['result']['speedup'], rel_tol=1e-12)
    load = 'decode 32'
    full_us = by_key[(load,'full')]['median_ns']/1000
    sliding_us = by_key[(load,'sliding')]['median_ns']/1000
    cost = 46*sliding_us+16*full_us
    share = 16*full_us/cost
    samples = prior['timing']['samples']
    assert len(samples) == 12
    ratios = [x['raw_us']['a213']/x['raw_us']['a215'] for x in samples]
    balanced = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0,12,2)]
    ratio = statistics.median(balanced)
    cut = 1-1/ratio
    return dict(kind='Conditional composition score sensitivity; no achieved composition result',
        frozen_evidence_sha256=hashes, fusion_official_job_id=official['id'],
        composition_source_sha256=EVIDENCE_HASHES['fusion-lookahead-probe/candidate.py'],
        fusion_official_speedup=official['result']['speedup'], reconstructed_speedup=reconstructed,
        target='full decode B32 only', fusion_target_us=full_us, fusion_sliding_us=sliding_us,
        decode32_load_cost_us=cost, full_target_load_share=share,
        old_a215=dict(raw_reverse_order_pair_ratios=balanced, median_ratio=ratio,
            target_cut_percent=100*cut, all_twelve_comparisons_favorable=all(x>1 for x in ratios),
            original_goal_percent=10, original_goal_met=cut>=.10,
            qualification='One prior session, two fixtures, no unchanged-path controls; edit induced both address scheduling and K/V buffering changes'),
        transfer_old_relative_cut=dict(aggregate_gain_percent=100*(score_factor(share,cut)-1),
            hypothetical_speedup=official['result']['speedup']*score_factor(share,cut),
            qualification='Freeze every other current-fusion official path; changed environment/calibration and composition make transfer conditional'),
        scenarios=[dict(assumed_target_cut_percent=100*x,
                        conditional_aggregate_gain_percent=100*(score_factor(share,x)-1))
                   for x in [0,.02,.03,.04,.05,.10]],
        required_target_cut_percent={str(100*gain):100*required_cut(share,gain)
                                     for gain in [.005,.01,.015]},
        proposed_new_composition_gate=dict(median_paired_target_cut_percent=4,
            reverse_order_pairs_favorable=6, unchanged_full_B16_verify_max_regression_percent=5,
            conditional_minimum_aggregate_gain_percent=100*(score_factor(share,.04)-1),
            purpose='Replicate useful component gain on fusion; the failed historical10% architectural goal remains failed'),
        compute_started=False, official_check_started=False, achieved_composition_score=False)


if __name__ == '__main__':
    result = build()
    (HERE/'model.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
