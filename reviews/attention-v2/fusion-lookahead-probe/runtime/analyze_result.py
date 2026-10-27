"""Reconstruct source-bound composition correctness and paired timing gates."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FREEZE = 'c9062363c665e37535d3370c1b6bfb54e04fc7c8'
NAMES = ['baseline', 'candidate']
TARGET = 'full_B32_L1_scored'
CONTROL = 'full_B16_L8_fusion_unchanged'
BOUNDARY = 'full_B32_L1_boundary'


def analyze():
    raw = (HERE/'result.json').read_bytes()
    report = json.loads(raw)
    receipt = json.loads((HERE/'launch-receipt.json').read_bytes())
    assert receipt['source_commit'] == FREEZE
    assert receipt['report_sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['exit_code'] == 0 and receipt['worker_passed']
    for name, sha in receipt['payload_sha256'].items():
        assert hashlib.sha256((HERE.parents[3]/name).read_bytes()).hexdigest() == sha, name
    assert report['worker_sha256'] == hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest()
    assert report['passed'] and report['correctness_complete'] and report['timing_complete']
    assert report['protocol']['seeds'] == [135711, 135712]
    expected = {TARGET, CONTROL, BOUNDARY, 'full_B4_L1_fallback_unchanged', 'full_B32_L1_dp_smoke'}
    assert len(report['cases']) == 5 and {x['case']['name'] for x in report['cases']} == expected
    oracle_calls = 0
    for case in report['cases']:
        assert case['passed'] and len(case['records']) == 6
        assert {(x['runner'], x['label']) for x in case['records']} == {
            (name, label) for name in NAMES for label in ['A1', 'B', 'A2']}
        for row in case['records']:
            assert row['passed'] and row['official_oracle_called']
            assert row['stable_addresses'] and row['immutable_inputs_and_untouched_rows']
            if row['label'] == 'A2':
                assert row['exact_output_newkv_return'] is True
            oracle_calls += 1
        assert len(case['cache_cross_arm']) == 3
        for row in case['cache_cross_arm']:
            assert row['passed'] and len(row['comparisons']) == 2
            assert all(x['exact'] and x['unequal_byte_count'] == 0 for x in row['comparisons'])
    assert len(report['concurrent_submissions']) == 1
    concurrent = report['concurrent_submissions'][0]
    assert concurrent['case'] == BOUNDARY and concurrent['passed']
    assert concurrent['disjoint_writable_caches'] and concurrent['separate_streams']
    assert concurrent['actual_hardware_overlap_measured'] is False
    assert len(concurrent['checks']) == 2
    for row in concurrent['checks']:
        assert row['passed'] and row['official_oracle_called']
        assert row['stable_addresses'] and row['immutable_inputs_and_untouched_rows']
        oracle_calls += 1
    assert len(report['timings']) == 2 and {x['case']['name'] for x in report['timings']} == {TARGET, CONTROL}
    timings = []
    for row in report['timings']:
        assert len(row['warmups']) == 4 and len(row['samples']) == 12
        for index, sample in enumerate(row['warmups']+row['samples']):
            assert sample['index'] == index and sample['input_version'] == (index//2) % 2
            assert sample['order'] == (NAMES if index % 2 == 0 else NAMES[::-1])
            for name in NAMES:
                value, check = sample['raw_us'][name], sample['checks'][name]
                assert math.isfinite(value) and value > 0
                assert check['passed'] and check['exact_output_newkv_return'] is True
                assert check['stable_addresses'] and check['immutable_inputs_and_untouched_rows']
                assert check['official_oracle_called'] is False
        ratios = [s['raw_us']['baseline']/s['raw_us']['candidate'] for s in row['samples']]
        pairs = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, 12, 2)]
        median_ratio = statistics.median(pairs)
        timings.append(dict(case=row['case']['name'],
            raw_median_us={name: statistics.median(s['raw_us'][name] for s in row['samples']) for name in NAMES},
            reverse_order_pair_ratios=pairs, median_pair_ratio=median_ratio,
            paired_latency_reduction_percent=100*(1-1/median_ratio),
            paired_candidate_slowdown_percent=100*(1/median_ratio-1),
            positive_pairs=sum(x > 1 for x in pairs)))
    target = next(x for x in timings if x['case'] == TARGET)
    control = next(x for x in timings if x['case'] == CONTROL)
    admitted = target['paired_latency_reduction_percent'] >= 4 and target['positive_pairs'] == 6
    admitted = admitted and control['paired_candidate_slowdown_percent'] <= 5
    assert report['benefit']['admitted'] == report['official_check_admitted'] == admitted
    return dict(source_freeze=FREEZE, raw_result_sha256=hashlib.sha256(raw).hexdigest(),
        validation_passed=True, original_oracle_calls=oracle_calls, exact_return_A_checks=10,
        cross_arm_cache_byte_comparisons=30, checked_warmup_measured_replays=64,
        fresh_seeds=report['protocol']['seeds'], timings=timings,
        native_benefit_gate_passed=admitted,
        decision='Supports one complete nonranking official check after serialized-slot coordination',
        qualifications=['One new session and two fresh fixtures, not an independent-session confidence interval or official aggregate.',
                        'The old A2-15 10% goal remains unmet; 4% was a new predeclared composition/replication gate.',
                        'Exact A2-15 producer functions were composed; page-address scheduling is not separately isolated from generated buffering.',
                        'No profiler, hardware counters or actual overlap/CTA occupancy measurements were used.'])


if __name__ == '__main__':
    output = analyze()
    (HERE/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
