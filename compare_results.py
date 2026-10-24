"""Compare preserved official check records offline; never run or estimate a kernel."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(experiment):
    report = json.loads((ROOT / 'results' / ('remote-' + experiment + '.json')).read_text())
    if not report['result'].get('passed'):
        raise ValueError('Only terminal passing timing records can be compared')
    rows = {(r['case']['layer'], r['case']['m']): r for r in report['result']['records']
            if r.get('event') == 'benchmark' and r.get('status') == 'pass'}
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control')
    parser.add_argument('candidate')
    parser.add_argument('--layer', default='router')
    args = parser.parse_args()
    control, a = load(args.control)
    candidate, b = load(args.candidate)
    if a.keys() != b.keys():
        raise ValueError('Shape sets differ; do not compare partial workloads as full results')
    print(f"# {args.candidate} vs {args.control} — saved H200 records only")
    print()
    for experiment, report in [(args.control, control), (args.candidate, candidate)]:
        result = report['result']
        print(f"{experiment}: job {report['admission']['job_id']}, "
              f"{result['speedup']:.7f}x SGLang, interval {result['speedup_ci']}")
    print('\nIndependent runs, not paired same-input trials. The normalized ratio below divides\n'
          'each candidate/SGLang ratio by the control/SGLang ratio; it does not remove\n'
          'all run-to-run noise or establish statistical significance. Higher is better.\n')
    if control['result'].get('evaluator') != candidate['result'].get('evaluator'):
        print('Evaluator/image metadata differs: inspect the exact records before drawing conclusions.\n')
    print('| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |')
    print('|---|---|---|---|')
    for (layer, m), row in sorted(a.items()):
        if layer != args.layer:
            continue
        other = b[layer, m]
        print(f"| {m} | {row['median_ns']/1000:.4f} | {other['median_ns']/1000:.4f} | "
              f"{other['speedup']/row['speedup']:.5f} |")
    print('\n| load | control speedup vs SGLang | candidate speedup vs SGLang |')
    print('|---|---|---|')
    for name, value in control['weighted_loads'].items():
        print(f"| {name} | {value['speedup']:.6f} | {candidate['weighted_loads'][name]['speedup']:.6f} |")
    print('\nUntouched layers can vary too. Inspect all shape records rather than attributing\n'
          'a whole-workload change to the targeted kernel. No new performance is predicted.')


if __name__ == '__main__':
    main()
