"""Offline analysis of one accepted attention cycle-two job; no native imports."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from assess_E6 import shapes, loads

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    exp = parser.parse_args().experiment
    ledger = json.loads((HERE/'attention-cycle2-budget.json').read_text())
    entry = next(r for r in ledger['requests'] if r['experiment'] == exp)
    plan = ledger['plans'][exp]
    assert entry['state'] == 'terminal'
    new = json.loads((HERE/f'{exp}-official-result.json').read_text())
    parent = json.loads((HERE/plan['parent_receipt']).read_text())
    assert new['id'] == entry['job_id'] and new['mode'] == 'check'
    assert hashlib.sha256((HERE/plan['candidate']).read_bytes()).hexdigest() == entry['source_sha256']
    targets = {tuple(v) for v in plan['target_shapes']}
    report = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        experiment=exp, job_id=new['id'], parent_job_id=parent['id'],
        candidate=plan['candidate'], source_commit=entry['source_commit'], source_sha256=entry['source_sha256'],
        passed=new['result']['passed'], status=new['status'],
        same_evaluator=new['result'].get('evaluator') == parent['result'].get('evaluator'),
        actual_platform_quota=entry.get('quota_after'), remaining_user_checks=ledger['remaining'],
        gpu=new['result'].get('gpu'), machine=new['result'].get('machine'),
        test_records=[r for r in new['result']['records'] if r['event'] == 'test'])
    if not report['passed']:
        report.update(error=new['result'].get('error'), log_tail=new['result'].get('log_tail'),
                      recommendation='Do not promote failure; preserve parent')
    else:
        before, after = shapes(parent), shapes(new)
        assert len(before) == len(after) == 12 and before.keys() == after.keys()
        bloads, aloads = loads(before), loads(after)
        score = lambda rows: math.exp(sum(math.log(r['ratio']) for r in rows.values())/6)
        oldG, G = score(bloads), score(aloads)
        assert math.isclose(G, new['result']['speedup'], rel_tol=1e-10)
        rows = []
        modeled_loads = {k: v['candidate_us'] for k, v in bloads.items()}
        for key, r in after.items():
            old = before[key]
            relative = r['paired_ratio']/old['paired_ratio']
            rows.append(dict(load=key[0], layer=key[1], target=key in targets,
                weight=r['case']['weight'], parent_adjusted_candidate_us=old['median_ns']/1000,
                parent_adjusted_sglang_us=old['baseline_ns']/1000,
                new_adjusted_candidate_us=r['median_ns']/1000,
                new_adjusted_sglang_us=r['baseline_ns']/1000,
                parent_raw_candidate_us=old['median_raw_ns']/1000,
                parent_raw_sglang_us=old['baseline_raw_ns']/1000,
                new_raw_candidate_us=r['median_raw_ns']/1000, new_raw_sglang_us=r['baseline_raw_ns']/1000,
                parent_paired_ratio=old['paired_ratio'], new_paired_ratio=r['paired_ratio'],
                relative_speedup=relative, normalized_latency_cut=1-1/relative))
            if key in targets:
                equivalent = old['baseline_ns']/r['paired_ratio']/1000
                modeled_loads[key[0]] -= r['case']['weight']*(old['median_ns']/1000-equivalent)
        assert sum(r['target'] for r in rows) == len(targets)
        regressions = [r for r in rows if not r['target'] and r['relative_speedup'] < 1/1.05]
        ci, oldci = new['result']['speedup_ci'], parent['result']['speedup_ci']
        overlaps = max(ci[0], oldci[0]) <= min(ci[1], oldci[1])
        goal = all(r['normalized_latency_cut'] >= plan.get('target_latency_cut', .2) for r in rows if r['target'])
        report.update(parent_score=oldG, score=G, relative_score_change=G/oldG-1,
            ci=ci, parent_ci=oldci, aggregate_ci_overlap=overlaps, target_goal_met=goal,
            shape_rows=rows, parent_loads=bloads, new_loads=aloads,
            unchanged_controls_over5pct_latency_regression=regressions,
            score_if_only_targets_observed_normalized_change=oldG*math.prod(
                bloads[k]['candidate_us']/v for k,v in modeled_loads.items())**(1/6),
            target_profiles=[r for r in new['result']['records']
                if r['event'] == 'profile' and (r['case']['load'], r['case']['layer']) in targets],
            recommendation=('adopt correct candidate with resolved gain over parent'
                if goal and not regressions and not overlaps and G > oldG
                else 'inspect target evidence and controls; total gain may be inconclusive'))
        with (HERE/f'{exp}-per-shape.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        text = [f'# {exp} actual official result', '',
            f"Job `{new['id']}`, source `{entry['source_commit']}`, SHA256 `{entry['source_sha256']}`.",
            f"Parent `{parent['id']}`; same evaluator: {report['same_evaluator']}.",
            f"Passed {len(report['test_records'])} prerequisites and {len(rows)} timed shapes.",
            f"Score {G:.10f}×, CI {ci}, vs parent {oldG:.10f}×, relative {(G/oldG-1)*100:+.4f}%.",
            f"Parent/new intervals overlap: {overlaps}; declared target goal met: {goal}.",
            f"Unchanged controls with >5% normalized latency regression: {len(regressions)}.", '',
            '| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |',
            '|---|---:|---:|---:|---:|']
        for r in rows:
            text.append(f"| {r['load']}/{r['layer']} | {r['parent_adjusted_candidate_us']:.3f} | {r['new_adjusted_candidate_us']:.3f} | {r['new_adjusted_sglang_us']:.3f} | {100*r['normalized_latency_cut']:+.3f}% |")
        text += ['', 'Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).', '',
                 '| Load | Weighted C µs | Weighted S µs | S/C |', '|---|---:|---:|---:|']
        for name,r in aloads.items():
            text.append(f"| {name} | {r['candidate_us']:.3f} | {r['sglang_us']:.3f} | {r['ratio']:.6f} |")
        text += ['', f"Target-only arithmetic using observed normalized change: {report['score_if_only_targets_observed_normalized_change']:.10f}×; NOT another achieved run.",
                 'Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.',
                 'Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.',
                 'Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.',
                 'CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.',
                 f"User attention checks remaining: {ledger['remaining']}; no push, ranked entry or Modal in this experiment."]
        (HERE/f'{exp}-RESULT.md').write_text('\n'.join(text)+'\n')
    (HERE/f'{exp}-assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['experiment','job_id','passed','score','relative_score_change','target_goal_met','recommendation']}))


if __name__ == '__main__':
    main()
