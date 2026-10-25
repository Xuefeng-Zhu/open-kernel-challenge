"""Offline E6 receipt comparison. Does not launch, poll or import candidate code."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def shapes(receipt):
    return {(r['case']['load'], r['case']['layer']): r for r in receipt['result']['records']
            if r['event'] == 'benchmark'}


def loads(rows):
    out = {}
    for (load, layer), row in rows.items():
        w = row['case']['weight']
        assert w == (46 if layer == 'sliding' else 16)
        assert math.isclose(row['baseline_ns']/row['median_ns'], row['paired_ratio'], rel_tol=1e-8)
        pair = out.setdefault(load, dict(candidate_us=0., sglang_us=0., layers=[]))
        pair['candidate_us'] += w*row['median_ns']/1000
        pair['sglang_us'] += w*row['baseline_ns']/1000
        pair['layers'].append(layer)
    assert len(out) == 6 and all(set(r['layers']) == {'sliding', 'full'} for r in out.values())
    for row in out.values():
        row['ratio'] = row['sglang_us']/row['candidate_us']
    return out


def main():
    parent = json.loads((HERE/'E5-official-result.json').read_text())
    new = json.loads((HERE/'E6-official-result.json').read_text())
    ledger = json.loads((HERE/'attention-cycle-budget.json').read_text())
    entry = next(e for e in ledger['requests'] if e['experiment'] == 'E6')
    assert entry['state'] == 'terminal' and ledger['remaining'] == 0
    assert new['id'] == entry['job_id'] and new['mode'] == 'check'
    source = (HERE/'mixed_fa3_sliding.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == entry['source_sha256']
    result = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
                  job_id=new['id'], parent_job_id=parent['id'],
                  source_commit=entry['source_commit'], source_sha256=entry['source_sha256'],
                  status=new['status'], passed=new['result']['passed'],
                  same_evaluator=new['result'].get('evaluator') == parent['result'].get('evaluator'),
                  user_attention_checks_remaining=ledger['remaining'],
                  actual_platform_quota=entry.get('quota_after'),
                  local_cpu_preflight='E6-cpu-preflight.json',
                  evidence_limits=['Separate official runs, not directly paired parent/candidate samples',
                      'Native timed-shape output/new-cache oracle and fresh-input CUDA graph timing only where receipt reports',
                      'No separate target GPU same-address A-B-A, page64/nonunit scales or untouched-cache-row assertion',
                      'No Modal app launched; unchanged E2 fallback retains simultaneous-call ticket limitation'])
    if not result['passed']:
        result.update(error=new['result'].get('error'), log_tail=new['result'].get('log_tail'),
                      selection='preserve parent E5; failed E6 is not selected')
    else:
        before, after = shapes(parent), shapes(new)
        assert len(before) == len(after) == 12 and before.keys() == after.keys()
        bloads, aloads = loads(before), loads(after)
        score = lambda rows: math.exp(sum(math.log(r['ratio']) for r in rows.values())/6)
        assert math.isclose(score(aloads), new['result']['speedup'], rel_tol=1e-10)
        assert math.isclose(score(bloads), parent['result']['speedup'], rel_tol=1e-10)
        target = ('verify 1', 'sliding')
        rows = []
        for key, row in after.items():
            old = before[key]
            relative = row['paired_ratio']/old['paired_ratio']
            rows.append(dict(load=key[0], layer=key[1], target=key == target,
                weight=row['case']['weight'], parent_raw_candidate_us=old['median_raw_ns']/1000,
                parent_raw_sglang_us=old['baseline_raw_ns']/1000,
                new_raw_candidate_us=row['median_raw_ns']/1000, new_raw_sglang_us=row['baseline_raw_ns']/1000,
                parent_adjusted_candidate_us=old['median_ns']/1000,
                parent_adjusted_sglang_us=old['baseline_ns']/1000,
                new_adjusted_candidate_us=row['median_ns']/1000,
                new_adjusted_sglang_us=row['baseline_ns']/1000,
                parent_paired_ratio=old['paired_ratio'], new_paired_ratio=row['paired_ratio'],
                relative_speedup=relative, normalized_latency_cut=1-1/relative))
        targetrow = next(r for r in rows if r['target'])
        regressions = [r for r in rows if not r['target'] and r['relative_speedup'] < 1/1.05]
        oldscore, newscore = score(bloads), score(aloads)
        oldtarget = before[target]
        equivalent_target_us = oldtarget['baseline_ns']/after[target]['paired_ratio']/1000
        model_load = bloads['verify 1']['candidate_us']-46*(oldtarget['median_ns']/1000-equivalent_target_us)
        model = oldscore*(bloads['verify 1']['candidate_us']/model_load)**(1/6)
        oldci, newci = parent['result']['speedup_ci'], new['result']['speedup_ci']
        intervals_overlap = max(oldci[0], newci[0]) <= min(oldci[1], newci[1])
        hypothesis = targetrow['normalized_latency_cut'] >= .20 and not regressions
        result.update(parent_score=oldscore, score=newscore, relative_score_change=newscore/oldscore-1,
            parent_ci=oldci, ci=newci, aggregate_ci_overlap=intervals_overlap,
            shape_rows=rows, parent_loads=bloads, new_loads=aloads, target=targetrow,
            controls_latency_regressions_over5pct=regressions,
            model_if_only_target_has_observed_normalized_change=model,
            prior_20pct_latency_cut_model=.911952106097244,
            prior_matching_sglang_model=.9462766259181346,
            declared_target_goal_supported=hypothesis,
            selection=('adopt E6' if hypothesis and not intervals_overlap and newscore > oldscore
                       else 'preserve E5 pending interpretation; E6 remains saved, no further check authorized'))
        with (HERE/'E6-per-shape.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (HERE/'E6-assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k in
                     {'job_id', 'passed', 'score', 'relative_score_change', 'aggregate_ci_overlap',
                      'declared_target_goal_supported', 'selection', 'controls_latency_regressions_over5pct'}}))


if __name__ == '__main__':
    main()
