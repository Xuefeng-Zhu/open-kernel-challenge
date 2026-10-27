"""Reconstruct native gates and paired whole-call evidence from retained bytes."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
WORKER_SHA = 'b4b47a4729fb06b85995c115cb91492bc6d552f5a809653359e38a67cd35f08f'
SOURCE = {'baseline.py': 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',
          'candidate.py': '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0'}
SUPPORT = {'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
           'official/reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
           'official/checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055'}


def analyze():
    raw = (HERE/'result.json').read_bytes()
    report = json.loads(raw)
    receipt = json.loads((HERE/'launch-receipt.json').read_bytes())
    assert receipt['source_commit'] == '0280972637e600e5f82a45f35778aed3f72886b1'
    assert receipt['report_sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['exit_code'] == 0 and receipt['worker_passed']
    for name, sha in receipt['payload_sha256'].items():
        assert hashlib.sha256((HERE.parents[3]/name).read_bytes()).hexdigest() == sha, name
    assert hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest() == WORKER_SHA
    assert report['worker_sha256'] == WORKER_SHA
    assert report['source_sha256'] == SOURCE
    assert report['support_sha256'] == SUPPORT
    for name, sha in SUPPORT.items():
        assert hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest() == sha
        assert report['verified_hashes'][name] == sha
    for name, sha in SOURCE.items():
        assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest() == sha
        assert report['verified_hashes'][name] == sha
    result = dict(result_sha256=hashlib.sha256(raw).hexdigest(),
        worker_sha256=WORKER_SHA, source_sha256=SOURCE,
        status=report['status'], passed=report['passed'],
        native_preparation_comparisons=len(report['prep_comparisons']),
        cases=len(report['cases']), concurrent_submissions=len(report['concurrent_submissions']),
        worker_wall_seconds=report['worker_wall_seconds'], environment=report.get('environment'),
        error_type=report.get('error_type'), error=report.get('error'),
        timings=[], failed_gates=[])
    for row in report['prep_comparisons']:
        if not row['passed']:
            result['failed_gates'].append(row)
    if not report['passed'] or not report['timing_complete']:
        result['admission'] = 'Rejected or incomplete; no official performance check justified'
        return result
    assert report['preparation_complete'] and report['correctness_complete']
    assert len(report['prep_comparisons']) == 12
    assert len(report['cases']) == 6 and len(report['concurrent_submissions']) == 3
    target_names = {f'full_B{b}_L{length}_scored' for b, length in [(16, 1), (16, 8), (32, 8)]}
    boundary_names = {f'full_B{b}_L{length}_boundary' for b, length in [(16, 1), (16, 8), (32, 8)]}
    case_names = target_names | boundary_names
    assert {row['case'] for row in report['prep_comparisons']} == case_names
    assert {(row['case'], row['input_version']) for row in report['prep_comparisons']} == {
        (name, version) for name in case_names for version in [0, 1]}
    assert {row['case']['name'] for row in report['cases']} == case_names
    assert {row['case'] for row in report['concurrent_submissions']} == boundary_names
    for row in report['prep_comparisons']:
        assert row['passed'] and row['finite_queries']
        assert all(x['exact'] and x['unequal_byte_count'] == 0 for x in row['comparisons'])
        assert all(x['passed'] for x in row['immutable_and_untouched'])
    oracle_count = 0
    for case in report['cases']:
        assert case['passed'] and len(case['records']) == 6
        assert {(x['runner'], x['label']) for x in case['records']} == {
            (name, label) for name in SOURCE_NAMES for label in ['A1', 'B', 'A2']}
        for row in case['records']:
            assert row['passed'] and row['official_oracle_called']
            assert row['stable_addresses'] and row['immutable_inputs_and_untouched_rows']
            if row['label'] == 'A2':
                assert row['exact_output_newkv_return'] is True
            oracle_count += 1
    for row in report['concurrent_submissions']:
        assert row['passed'] and row['disjoint_writable_caches'] and row['separate_graph_captures']
        assert row['separate_streams'] and row['actual_hardware_overlap_measured'] is False
        assert len(row['checks']) == 2
        for check in row['checks']:
            assert check['passed'] and check['official_oracle_called']
            assert check['stable_addresses'] and check['immutable_inputs_and_untouched_rows']
            oracle_count += 1
    assert len(report['timings']) == 3
    assert {row['case']['name'] for row in report['timings']} == target_names
    measured_checks = 0
    for timing in report['timings']:
        assert len(timing['warmups']) == 4 and len(timing['samples']) == 12
        for index, sample in enumerate(timing['warmups']+timing['samples']):
            assert sample['index'] == index
            assert sample['input_version'] == (index//2) % 2
            assert sample['order'] == (SOURCE_NAMES if index % 2 == 0 else SOURCE_NAMES[::-1])
            for name in SOURCE_NAMES:
                value, check = sample['raw_us'][name], sample['checks'][name]
                assert math.isfinite(value) and value > 0
                assert check['passed'] and check['official_oracle_called'] is False
                assert check['exact_output_newkv_return'] is True
                assert check['stable_addresses'] and check['immutable_inputs_and_untouched_rows']
                measured_checks += 1
        ratios = [s['raw_us']['baseline']/s['raw_us']['candidate'] for s in timing['samples']]
        pairs = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, 12, 2)]
        median_ratio = statistics.median(pairs)
        medians = {name: statistics.median(s['raw_us'][name] for s in timing['samples'])
                   for name in SOURCE_NAMES}
        result['timings'].append(dict(case=timing['case']['name'], raw_median_us=medians,
            raw_median_difference_us=medians['baseline']-medians['candidate'],
            reverse_order_pair_ratios=pairs, median_pair_ratio=median_ratio,
            latency_reduction_from_median_ratio_percent=100*(1-1/median_ratio),
            positive_pairs=sum(value > 1 for value in pairs),
            per_input_version_raw_median_us={str(version): {
                name: statistics.median(s['raw_us'][name] for s in timing['samples']
                                        if s['input_version'] == version)
                for name in SOURCE_NAMES} for version in [0, 1]},
            qualification='One session, six paired ratios, two inputs; no independent-session interval or official score'))
    result.update(official_oracle_calls=oracle_count, checked_timing_replays=measured_checks,
                  admission='All native gates complete; assess complete-call benefit before any official check')
    return result


SOURCE_NAMES = ['baseline', 'candidate']

if __name__ == '__main__':
    output = analyze()
    (HERE/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
