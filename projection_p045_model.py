"""P045 score sensitivity from the immutable P039 receipt; no GPU measurement."""
import hashlib
import json
from projection_p041_plan import ROOT, read_rows, score


def build():
    receipt = ROOT / 'results/remote-P2-04-router-m1-fp32-gemv.csv'
    rows = read_rows()
    original, loads = score(rows)
    scenarios = []
    for cut in (0.05, 0.10, 0.20, 1.0):
        changed = [{**r, 'candidate_us': r['candidate_us'] * (1.0 - cut)}
                   if r['layer'] == 'pooler' and r['m'] in (8, 16, 32, 128)
                   else r for r in rows]
        hypothetical, _ = score(changed)
        scenarios.append({'target_latency_cut_fraction': cut,
                          'modeled_score': hypothetical,
                          'modeled_relative_score_gain': hypothetical / original - 1.0})
    return {
        'source_receipt': str(receipt.relative_to(ROOT)),
        'source_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'parent_job': 'd5636ac8d1404abd', 'parent_score': original,
        'target_layer': 'pooler', 'target_scored_m': [8, 16, 32, 128],
        'scoring': 'Six equally weighted loads; within each load all five families have weight 62.',
        'scenarios': scenarios,
        'interpretation': ('These hold every other time fixed and predict no actual speedup. '
                           'The 100% cut is an impossible zero-latency algebraic ceiling, '
                           'not a physically achievable or expected result.'),
        'gpu_executed': False,
    }


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
