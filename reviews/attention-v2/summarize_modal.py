"""Offline paired summaries; raw and measured-empty-graph calibration stay separate."""
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def pair_ratios(rows,names,key):
    pairs=[]
    for at in range(0,len(rows),2):
        left,right=rows[at:at+2]
        if left['order']!=list(reversed(right['order'])):raise ValueError('Unbalanced order pair')
        values=[row[key][names[1]]/row[key][names[0]] for row in [left,right]]
        if not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Unresolved nonpositive time')
        pairs.append(math.sqrt(math.prod(values)))
    return pairs


def describe(evidence):
    report=evidence['result']['report'];mode=report['mode']
    names=['v2','sglang'] if mode=='baseline' else ['merge','v2']
    result=dict(mode=mode,names=names,passed=report['passed'],cases=[],loads=[],
        qualification='Three reverse-order pairs per shape only; no official score or confidence interval; calibrated diagnostic uses independently measured empty graphs')
    for row in report['benchmarks']:
        samples=row['samples'];raw=pair_ratios(samples,names,'raw_us')
        item=dict(case=row['case'],raw_median_us={n:statistics.median(x['raw_us'][n] for x in samples) for n in names},
            raw_pair_ratios=raw,raw_median_reference_over_candidate=statistics.median(raw),
            winning_pairs=sum(v>1 for v in raw),empty_graph_median_us=row.get('empty_graph_median_us'),
            raw_all_samples=samples)
        if row.get('calibration_resolves_both'):
            cal=pair_ratios(samples,names,'calibrated_us');item.update(calibrated_pair_ratios=cal,
                calibrated_median_us={n:statistics.median(x['calibrated_us'][n] for x in samples) for n in names},
                calibrated_median_reference_over_candidate=statistics.median(cal))
        result['cases'].append(item)
    for load in sorted({row['case']['load'] for row in report['benchmarks']}):
        component=[row for row in report['benchmarks'] if row['case']['load']==load]
        combined=[]
        for i in range(6):
            order=component[0]['samples'][i]['order']
            if any(x['samples'][i]['order']!=order for x in component):raise ValueError('Load order mismatch')
            combined.append(dict(order=order,raw_us={n:sum(x['case']['weight']*x['samples'][i]['raw_us'][n] for x in component) for n in names}))
        ratios=pair_ratios(combined,names,'raw_us')
        result['loads'].append(dict(load=load,raw_pair_ratios=ratios,raw_median_reference_over_candidate=statistics.median(ratios)))
    result['raw_weighted_load_geometric_ratio']=math.exp(statistics.mean(math.log(x['raw_median_reference_over_candidate']) for x in result['loads']))
    return result


def main():
    summaries={}
    for experiment in ['A03-baseline','A05-merge']:
        path=HERE/(experiment+'-result.json')
        if path.exists():
            e=json.loads(path.read_text())
            if e['attempt']['correctness_and_graph_passed']:summaries[experiment]=describe(e)
    (HERE/'modal-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    for key,s in summaries.items():
        print(key,'raw weighted ratio',s['raw_weighted_load_geometric_ratio'])
        for x in s['cases']:
            c=x['case']
            if c['layer']=='full' and c.get('q_len',1)==8:
                print(c['batch'],x['raw_median_us'],x['raw_pair_ratios'],x.get('calibrated_median_us'))


if __name__=='__main__':main()
