"""Source-bound conditional score model; no GPU execution or speed forecast."""
from pathlib import Path
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
BASELINE_SHA = '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0'
OFFICIAL_SHA = '46f9818a029a68995d6bad8b99c79a72f390ab09e464c01eb9a79b64bb160001'
PROFILE_SHA = '3ab2cf3685af1fba292b657e134ea1719176aee15c3cbbff147921759981c54b'


def checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Frozen evidence hash mismatch: '+path.name)
    return data


def score_factor(load_costs, savings):
    """Freeze all other paths; signed savings are in each complete target call."""
    factors = []
    for batch, old_cost in load_costs.items():
        new_cost = old_cost - 46*savings.get(batch, 0.)
        if not math.isfinite(new_cost) or new_cost <= 0:
            raise ValueError('Nonpositive or nonfinite modeled load cost')
        factors.append(old_cost/new_cost)
    return math.exp(sum(math.log(x) for x in factors)/6)


def equal_saving_for_gain(load_costs, gain):
    low, high = 0., min(load_costs.values())/46*.99
    for _ in range(80):
        middle = (low+high)/2
        if score_factor(load_costs, {b: middle for b in load_costs}) < 1+gain:
            low = middle
        else:
            high = middle
    return (low+high)/2


def main():
    baseline = V2/'fused-preparation-probe/candidate.py'
    receipt = V2/'fused-preparation-probe/official-check/official-result.json'
    profile = V2/'fused-preparation-probe/official-check/artifacts/candidate.2f5f1fb54b48486d.txt'
    checked_bytes(baseline, BASELINE_SHA)
    job = json.loads(checked_bytes(receipt, OFFICIAL_SHA))
    checked_bytes(profile, PROFILE_SHA)
    assert job['id'] == '2f5f1fb54b48486d' and job['mode'] == 'check'
    assert job['status'] == 'done' and job['result']['passed']
    records = job['result']['records']
    bench = {(row['case']['load'], row['case']['layer']): row
             for row in records if row['event'] == 'benchmark'}
    assert len(bench) == 12
    six_loads = sorted({key[0] for key in bench})
    ratios = []
    for load in six_loads:
        reference = sum(bench[(load, layer)]['baseline_ns']*weight
                        for layer, weight in [('sliding', 46), ('full', 16)])
        candidate = sum(bench[(load, layer)]['median_ns']*weight
                        for layer, weight in [('sliding', 46), ('full', 16)])
        ratios.append(reference/candidate)
    reconstructed = math.exp(sum(math.log(x) for x in ratios)/6)
    assert math.isclose(reconstructed, job['result']['speedup'], rel_tol=1e-12)
    targets = {}
    for batch in [16, 32]:
        load = f'verify {batch}'
        sliding, full = [bench[(load, layer)]['median_ns']/1000 for layer in ['sliding', 'full']]
        cost = 46*sliding+16*full
        prof = next(row['report'] for row in records if row['event'] == 'profile'
                    and row['case']['load'] == load and row['case']['layer'] == 'sliding')
        def duration(prefix):
            line = next(line for line in prof.splitlines() if prefix in line)
            return float(re.search(r'\s([0-9.]+) µs(?:\s|$)', line).group(1))
        targets[batch] = dict(sliding_call_us=sliding, full_call_us=full,
            weighted_load_us=cost, sliding_fraction_of_load=46*sliding/cost,
            official_profile_rounded_us=dict(mainloop=duration('enable_sm90_or_later'),
                combine=duration('FlashAttnFwdCombine'),
                scheduler_prepare=duration('prepare_varlen_num_blocks_kernel'),
                fused_prepare=duration('_fa3_prepare_fused'),
                gate=duration('norm_gate_kernel'),
                summed_activities=duration('total'), busy_union=duration('GPU busy')),
            profile_has_overlap='kernels overlap' in prof,
            conditional_useful_query_tasks=batch*2,
            conditional_tasks_per_132_sms=batch*2/132)
    costs = {batch: row['weighted_load_us'] for batch, row in targets.items()}
    scenarios = []
    for saving in [0, 1, 2, 3, 4, 5]:
        factor = score_factor(costs, {16: saving, 32: saving})
        scenarios.append(dict(complete_call_savings_us_each=saving,
            overall_score_factor=factor, overall_gain_percent=100*(factor-1)))
    output = dict(kind='Conditional mathematical model, not predicted or achieved speedup',
        baseline_commit='05c9cc08948c42fed9e09aae1cd2b41edf966c74',
        baseline_source_sha256=BASELINE_SHA, official_job_id=job['id'],
        official_receipt_sha256=OFFICIAL_SHA, official_profile_artifact_sha256=PROFILE_SHA,
        reconstructed_baseline_speedup=reconstructed, targets=targets,
        equal_complete_call_savings_scenarios=scenarios,
        equal_saving_us_required_for_overall_gain_percent={str(gain*100): equal_saving_for_gain(costs, gain)
                                                        for gain in [.005, .01, .015]},
        qualifications=[
            'Times used in score arithmetic are current official calibrated complete-call medians; profile activity times are separate rounded observations.',
            'The 4.2/4.4us combine activities overlap other work and cannot be subtracted from whole-call latency or treated as maximum recoverable latency.',
            'Scheduler preparation may remain with num_splits=1; its removal is not assumed.',
            'The 32/64 useful-query-task counts assume one packed M>=64 tile per batch/KV head for L8,GQA8,Hkv2. They are not measured CTA grids, active SMs, occupancy, or proof of the installed specialization.',
            'No forecast is made for the mainloop penalty of losing split parallelism; both whole-call signs must be measured.',
            'The model freezes all other paths, keeps weight46 sliding/16 full, and takes the geometric mean over six equally weighted loads.'])
    (HERE/'model.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'baseline_speedup': reconstructed, 'scenarios': scenarios,
                      'equal_saving_thresholds_us': output['equal_saving_us_required_for_overall_gain_percent']}, indent=2))
    return output


if __name__ == '__main__':
    main()
