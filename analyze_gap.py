"""Describe measured time shares and explicitly hypothetical task-floor bounds offline."""
import argparse
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    args = parser.parse_args()
    report = json.loads((ROOT / 'results' / ('remote-' + args.experiment + '.json')).read_text())
    assert report['result']['passed']
    rows = [r for r in report['result']['records'] if r.get('event') == 'benchmark']
    assert len(rows) == 30
    floors = {}
    for line in (ROOT / 'TASK.md').read_text().splitlines():
        match = re.match(r'\| ([^|]+) \| m (\d+), layer (\w+) \| (\d+) \| ([\d.]+) µs \|', line)
        if match:
            floors[match[3], int(match[2])] = float(match[5])
    assert len(floors) == 30
    loads = report['weighted_loads']
    families = []
    for layer in sorted({r['case']['layer'] for r in rows}):
        subset = [r for r in rows if r['case']['layer'] == layer]
        share = sum(r['median_ns']/1000*r['case']['weight'] /
                    loads[r['case']['load']]['candidate_load_us'] for r in subset) / len(loads)
        scenario = {name: value['candidate_load_us'] for name, value in loads.items()}
        for row in subset:
            case = row['case']
            scenario[case['load']] -= (row['median_ns']/1000-floors[layer, case['m']])*case['weight']
        bound = math.exp(sum(math.log(loads[name]['candidate_load_us']/value)
                             for name, value in scenario.items())/len(loads))
        families.append((share, layer, subset, bound))
    board = json.loads((ROOT / 'leaderboard_reference.json').read_text())
    leader = board['leader']
    leader_speedup = board['published_reference_us'] / leader['value']
    speedup = report['result']['speedup']
    print('# Gap diagnosis from actual H200 evidence')
    print(f'\nOur {args.experiment}: job `{report["admission"]["job_id"]}`, {speedup:.7f}× SGLang.')
    print(f'Leader snapshot 2026-10-24 16:06 UTC: {leader_speedup:.10f}× ({leader["team"]}).')
    print(f'At identical normalization, matching it requires {100*(1-speedup/leader_speedup):.2f}% '
          'further latency reduction. This compares independent runs, not paired inputs/GPUs.')
    print('\nOfficial scoring is the geometric mean of six SGLang/candidate load ratios. Each '
          'load sums its five shape latencies times weight62. Shares below average each '
          'family’s fraction within those six loads; they sum to100% and reflect local '
          'log-score sensitivity to proportional family changes, not a replacement score.')
    print('\n| Family | Measured mean per-load share | Measured latency range µs | TASK floor range µs | '
          'Optimistic additional speedup if only this whole family reached its floors |')
    print('|---|---|---|---|---|')
    for share, layer, subset, bound in sorted(families, reverse=True):
        values = [r['median_ns']/1000 for r in subset]
        low = [floors[layer, r['case']['m']] for r in subset]
        print(f'| {layer} | {100*share:.2f}% | {min(values):.2f}–{max(values):.2f} | '
              f'{min(low):.1f}–{max(low):.1f} | {bound:.4f}× |')
    without_router = {name: value['candidate_load_us'] for name, value in loads.items()}
    for row in rows:
        if row['case']['layer'] == 'router':
            without_router[row['case']['load']] -= row['median_ns']/1000*row['case']['weight']
    free_router = math.exp(sum(math.log(loads[name]['candidate_load_us']/value)
                               for name, value in without_router.items())/len(loads))
    print('\n**Floor columns are theoretical scenarios, not measured/predicted performance or '
          'candidate scores.** They assume the published optimistic traffic/compute floor '
          'is attainable for every shape of one family while all other times stay fixed. '
          'Launch costs, utilization, reductions, precision requirements and memory access '
          'can prevent that. Family bounds cannot be added together.')
    print(f'\nEven the impossible zero-time router scenario gives only {free_router:.4f}× further '
          f'speedup, or {speedup*free_router:.4f}× SGLang—below the leader snapshot. '
          'The gap therefore cannot be closed by router-only work on these measured loads.')
    if report['admission']['candidate'] == 'candidates/router_pooler_hybrid.py':
        print('\nThe final verified composition uses split32 native router at m<=8, wide hi+lo '
              'router above8, and single-row SIMT pooler. Other BF16 paths remain original. '
              'R06 pooler split8 regressed despite passing correctness; retain that failure. '
              'R08 validated router split32 and R09 validated the one-row SIMT pooler component; '
              'R10 checks their composition. QKV split8 stays CPU-only after the related R06 '
              'regression. Further BF16 GEMM/GEMV layout, tile and shape-dispatch work must '
              'start from profiles and preserve BF16 weights/activations. No unsupported '
              'cross-layer fusion: this contract supplies one layer/call, so caching '
              'activations is forbidden. All10 new checks are terminal; no further quota '
              'is authorized. A future GPU experiment needs a new explicit budget.')
    else:
        print('\nOptimization coverage is experiment-specific; inspect the exact committed source '
              'and remote experiment record. Do not infer competitors or other variants from these shares.')
    print('\nPublic-source limits: the official board returns participant aggregate score, '
          'attempts and a job ID, not code/per-shape records. The documented GET /jobs/{id} '
          'route is own-jobs-only. An unauthenticated read of leader job5987bf672867464e '
          'returned HTTP401; no token retry or access bypass was attempted. No verified '
          'public competitor source/per-shape details were found in the inspected official '
          'sources, so their implementation remains unknown. This is not a claim about '
          'unrelated public repositories elsewhere.')
    print('\nSources: [official task](https://openkernelchallenge.dev/problems/muse_bf16_proj), '
          '[official API/mode guide](https://openkernelchallenge.dev/llms.txt), '
          '[problem board](https://openkernelchallenge.dev/leaderboard?problem=muse_bf16_proj), '
          f'`results/remote-{args.experiment}.json` and CSV. Reproduce offline with '
          f'`.venv/bin/python analyze_gap.py {args.experiment}` (uses the committed sanitized '
          'public leader snapshot; no network/GPU call is made).')


if __name__ == '__main__':
    main()
