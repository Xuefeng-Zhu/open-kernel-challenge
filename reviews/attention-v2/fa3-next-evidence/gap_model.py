"""Offline conditional Attention score model; never queries a service or GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
LOADS = [f'{kind} {batch}' for kind in ('decode', 'verify') for batch in (1, 16, 32)]
PATHS = [(load, layer) for load in LOADS for layer in ('sliding', 'full')]
WEIGHTS = {'sliding': 46, 'full': 16}
GROUPS = {
    'four_large_full': [(load, 'full') for load in LOADS if not load.endswith(' 1')],
    'three_large_FA3_full': [(load, 'full') for load in ('decode 16', 'verify 16', 'verify 32')],
    'all_six_full': [(load, 'full') for load in LOADS],
    'two_full_verify': [(load, 'full') for load in ('verify 16', 'verify 32')],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_floors(text):
    floors = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if len(cells) != 4 or cells[0] not in LOADS:
            continue
        layer = re.search(r'\blayer (sliding|full)\b', cells[1]).group(1)
        assert int(cells[2]) == WEIGHTS[layer]
        key = (cells[0], layer)
        assert key not in floors
        floors[key] = float(cells[3].split()[0])
    assert set(floors) == set(PATHS)
    return floors


def score(times, reference):
    ratios = []
    for load in LOADS:
        current = sum(WEIGHTS[layer] * times[(load, layer)] for layer in WEIGHTS)
        fixed_reference = sum(WEIGHTS[layer] * reference[(load, layer)] for layer in WEIGHTS)
        if current <= 0 or fixed_reference <= 0:
            raise ValueError('Every weighted load must retain positive time')
        ratios.append(fixed_reference / current)
    return math.exp(sum(map(math.log, ratios)) / len(ratios))


def scaled(times, paths, cut):
    if not 0 <= cut <= 1:
        raise ValueError('Latency reduction must be in [0,1]')
    selected = set(paths)
    return {key: value * (1-cut) if key in selected else value for key, value in times.items()}


def solve_cut(times, reference, paths, target):
    zero_selected = scaled(times, paths, 1.)
    zero_load = any(sum(WEIGHTS[layer] * zero_selected[(load, layer)] for layer in WEIGHTS) == 0
                    for load in LOADS)
    limit = math.inf if zero_load else score(zero_selected, reference)
    if limit < target:
        return None, limit
    lo, hi = 0., 1.
    for _ in range(100):
        middle = (lo+hi)/2
        if score(scaled(times, paths, middle), reference) < target:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2, limit


def floor_violations(times, floors):
    return [dict(load=load, layer=layer, modeled_us=times[(load, layer)],
                 task_floor_us=floors[(load, layer)])
            for load, layer in PATHS if times[(load, layer)] < floors[(load, layer)] - 1e-12]


def make_model(public_receipt):
    result_path = V2/'fusion-lookahead-probe/official-check/official-result.json'
    task_path = V2/'official/TASK.md'
    job = json.loads(result_path.read_text())
    assert job['id'] == '5090b83f1d63407b' and job['mode'] == 'check'
    assert job['status'] == 'done' and job['result']['passed']
    rows = [row for row in job['result']['records'] if row['event'] == 'benchmark']
    assert len(rows) == 12
    times = {(r['case']['load'], r['case']['layer']): r['median_ns']/1000 for r in rows}
    reference = {(r['case']['load'], r['case']['layer']): r['baseline_ns']/1000 for r in rows}
    assert set(times) == set(PATHS)
    floors = task_floors(task_path.read_text())
    current = score(times, reference)
    assert math.isclose(current, job['result']['speedup'], rel_tol=1e-12)
    receipt = json.loads(public_receipt.read_text())
    public = receipt['public_response']
    assert public['board'] == 'muse_attention_decode' and public['you'] is None
    leader = float(public['leader']['speedup'])
    ratio = leader/current
    group_models = {}
    for name, paths in GROUPS.items():
        cut, limit = solve_cut(times, reference, paths, leader)
        at_floors = {key: floors[key] if key in paths else value for key, value in times.items()}
        group_models[name] = dict(paths=[dict(load=a, layer=b) for a,b in paths],
            score_if_selected_paths_halve=score(scaled(times, paths, .5), reference),
            zero_selected_time_conditional_limit=limit,
            target_reachable_without_floor_constraint=cut is not None,
            uniform_cut_needed_percent=None if cut is None else 100*cut,
            required_cut_task_floor_violations=[] if cut is None else floor_violations(scaled(times, paths, cut), floors),
            score_if_selected_paths_at_TASK_floors=score(at_floors, reference))
    large = [key for key in PATHS if not key[0].endswith(' 1')]
    small = [key for key in PATHS if key[0].endswith(' 1')]
    broad_times = scaled(scaled(times, large, .5), small, .7)
    large_half = scaled(times, large, .5)
    required_small_cut, _ = solve_cut(large_half, reference, small, leader)
    closed_form_small_cut = 1 - 4 / ratio**3
    assert math.isclose(required_small_cut, closed_form_small_cut, abs_tol=1e-12)
    full_floor_times = {key: floors[key] if key[1] == 'full' else value for key,value in times.items()}
    sliding_paths = [key for key in PATHS if key[1] == 'sliding']
    sliding_cut, _ = solve_cut(full_floor_times, reference, sliding_paths, leader)
    return dict(
        scope='Offline fixed-reference conditional score arithmetic, not a forecast or hardware ceiling',
        public_summary=dict(observed_utc=receipt['observed_utc'], leader_team=public['leader']['team'],
                            leader_speedup_public_rounded=leader, ranked_count=public['ranked'], our_rank=None),
        provenance=dict(public_CLI_raw_sha256=receipt['raw_sha256'], CLI_binary_sha256=receipt['cli_sha256'],
                        local_public_receipt_sha256=digest(public_receipt),
                        own_official_result_sha256=digest(result_path), TASK_sha256=digest(task_path)),
        measured_current=dict(score=current, ci=job['result']['speedup_ci'], mode='check', ranked_entry=False,
                              candidate_sha256='3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'),
        formula='score = geomean_over_6_loads((46*R_sliding+16*R_full)/(46*C_sliding+16*C_full)); R fixed from own same job',
        leader_over_current=ratio, all_paths_uniform_cut_needed_percent=100*(1-1/ratio),
        path_inputs=[dict(load=a, layer=b, current_us=times[(a,b)], fixed_reference_us=reference[(a,b)],
                          weight=WEIGHTS[b], TASK_floor_us=floors[(a,b)]) for a,b in PATHS],
        group_models=group_models,
        broad_engineering_scenario=dict(large_eight_path_cut_percent=50., B1_four_path_cut_percent=70.,
            score=score(broad_times, reference), task_floor_violations=floor_violations(broad_times, floors),
            modeled_path_times=[dict(load=a, layer=b, time_us=broad_times[(a,b)]) for a,b in PATHS]),
        required_B1_after_large_half=dict(cut_percent=100*required_small_cut, closed_form='1 - 4/(leader/current)^3',
            score=score(scaled(large_half, small, required_small_cut), reference),
            task_floor_violations=floor_violations(scaled(large_half, small, required_small_cut), floors)),
        required_sliding_after_full_TASK_floors=dict(full_floor_only_score=score(full_floor_times, reference),
            sliding_uniform_cut_percent=100*sliding_cut,
            score=score(scaled(full_floor_times, sliding_paths, sliding_cut), reference),
            task_floor_violations=floor_violations(scaled(full_floor_times, sliding_paths, sliding_cut), floors)),
        limitations=[
            'Public leader is rounded; its confidence interval, per-shape times, kernel and private results were not requested or inferred.',
            'Our result is a nonranking check; public you:null supplies no rank for us.',
            'References and all untargeted paths are frozen at our one official run; future evaluator/input effects are not modeled.',
            'TASK floors are optimistic published bandwidth/compute estimates, not proven attainable whole-call lower bounds or a true hardware ceiling.',
            'Group zero-time limits are mathematical counterfactuals only; below-floor required cuts are not credible hardware targets.',
            'The broad scenario gives necessary scale and an internally consistent engineering target, not an algorithm, forecast or evidence of achievable performance.',
            'No new source candidate, GPU launch, service request, quota use or formal entry follows from this model.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-receipt', type=Path,
                        default=HERE.parents[3]/'attention-public-leaderboard-20261025.local.json')
    parser.add_argument('--output', type=Path, default=HERE/'gap_model.json')
    args = parser.parse_args()
    model = make_model(args.public_receipt)
    args.output.write_text(json.dumps(model, indent=2)+'\n')
    print(json.dumps({key:model[key] for key in ['leader_over_current', 'all_paths_uniform_cut_needed_percent',
        'broad_engineering_scenario', 'required_B1_after_large_half', 'required_sliding_after_full_TASK_floors']}, indent=2))


if __name__ == '__main__':
    main()
