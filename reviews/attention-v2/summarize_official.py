"""Reproduce six-load official scores offline; retain source receipts separately."""
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def describe(receipt):
    result=receipt['result']
    if not result.get('passed'):return dict(passed=False,error=result.get('error'))
    rows=[r for r in result['records'] if r['event']=='benchmark']
    if len(rows)!=12:raise ValueError('Expected all twelve official timing rows')
    shapes=[];loads=[]
    for r in rows:
        c=r['case'];expected=46 if c['layer']=='sliding' else 16
        if c['weight']!=expected:raise ValueError('Official workload weight changed')
        shapes.append(dict(case=c,official_adjusted_us=r['median_ns']/1000,
            raw_median_us=r['median_raw_ns']/1000,baseline_adjusted_us=r['baseline_ns']/1000,
            baseline_raw_us=r['baseline_raw_ns']/1000,paired_reference_over_candidate=r['paired_ratio'],
            runs=r['runs'],raw_sample_count=len(r['runs_ns']),
            fixed_overhead_ns=r['replay_overhead']['overhead_ns']))
    for name in sorted({r['case']['load'] for r in rows}):
        pair=[r for r in rows if r['case']['load']==name]
        if len(pair)!=2 or {r['case']['layer'] for r in pair}!={'sliding','full'}:
            raise ValueError('Each load must contain exactly sliding and full')
        base=sum(r['case']['weight']*r['baseline_ns'] for r in pair)
        cand=sum(r['case']['weight']*r['median_ns'] for r in pair)
        loads.append(dict(load=name,baseline_us=base/1000,candidate_us=cand/1000,speedup=base/cand))
    if len(loads)!=6:raise ValueError('Expected six equal-weight loads')
    reproduced=math.exp(statistics.mean(math.log(r['speedup']) for r in loads))
    if not math.isclose(reproduced,result['speedup'],rel_tol=1e-8):raise ValueError('Score did not reproduce')
    return dict(passed=True,speedup=result['speedup'],speedup_ci=result['speedup_ci'],
        reproduced_six_weighted_load_score=reproduced,shapes=shapes,loads=loads,
        tests_passed=sum(r['event']=='test' and r['status']=='pass' for r in result['records']),
        flags=result['flags'],evaluator=result['evaluator'],machine=result['machine'])


def main():
    ledger=json.loads((HERE/'attention-cycle-budget.json').read_text());results={}
    for e in ledger['requests']:
        if e['state']!='terminal':continue
        receipt=json.loads((HERE/(e['experiment']+'-official-result.json')).read_text())
        result=describe(receipt);result.update(job_id=e['job_id'],source_commit=e['source_commit'],
            source_sha256=e['source_sha256'],source=e['candidate'],receipt=e['experiment']+'-official-result.json')
        results[e['experiment']]=result
    if results.get('E0',{}).get('passed'):
        original=results['E0']
        for name,r in results.items():
            if not r.get('passed'):continue
            r['score_ratio_vs_E0']=r['speedup']/original['speedup']
            previous={s['case']['load']+':'+s['case']['layer']:s for s in original['shapes']}
            for s in r['shapes']:
                p=previous[s['case']['load']+':'+s['case']['layer']]
                s['relative_reference_normalized_speedup_vs_E0']=s['paired_reference_over_candidate']/p['paired_reference_over_candidate']
    report=dict(method='Six weighted loads (46 sliding + 16 full); geometric mean over loads. Official paired adjusted times and raw times kept separate.',
        qualification='Candidate comparisons across separate official runs are not directly paired. CI overlap and 0–2% changes are inconclusive; report any >5% regression.',
        checks_used=len(ledger['requests']),checks_remaining=ledger['remaining'],experiments=results)
    (HERE/'official-cycle-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in results.items():
        print(name,'passed',r['passed'],'score',r.get('speedup'),'CI',r.get('speedup_ci'))
        if r.get('passed'):
            for s in r['shapes']:
                print(s['case']['load'],s['case']['layer'],round(s['official_adjusted_us'],3),round(s['relative_reference_normalized_speedup_vs_E0'],4))


if __name__=='__main__':main()
