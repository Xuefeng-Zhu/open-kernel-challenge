"""Verify retained native evidence and reconstruct the rejected benefit gate."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FREEZE = '5e494062a91a643df674887140125b9333f2fb92'
NAMES = ['baseline', 'candidate']


def analyze():
    raw = (HERE/'result.json').read_bytes()
    result = json.loads(raw)
    receipt = json.loads((HERE/'launch-receipt.json').read_bytes())
    assert receipt['source_commit'] == FREEZE
    assert receipt['report_sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['exit_code'] == 0 and result['passed'] and result['status'] == 'complete'
    for name, sha in receipt['payload_sha256'].items():
        assert hashlib.sha256((HERE.parents[3]/name).read_bytes()).hexdigest() == sha, name
    assert result['worker_sha256'] == hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest()
    identity = result['fa3_identity']
    assert identity['admitted'] and identity['exact_observed_extensions_verified']
    assert identity['same_callable_in_both_modules'] and identity['num_splits_default'] == '0'
    assert not identity['loaded_binary_paths_omitted']
    assert identity['loaded_library_discovery']['status'] == 'observed'
    assert identity['loaded_binaries'] and all(len(row['sha256']) == 64 for row in identity['loaded_binaries'])
    source = identity['source']
    assert hashlib.sha256(source['file_text'].encode()).hexdigest() == source['sha256']
    assert hashlib.sha256(source['function_source'].encode()).hexdigest() == source['function_source_sha256']
    targets = {f'sliding_B{b}_L8_scored' for b in [16, 32]}
    boundaries = {f'sliding_B{b}_L8_boundary' for b in [16, 32]}
    edges = {f'sliding_B{b}_L8_edges' for b in [16, 32]}
    control = 'full_B16_L8_unchanged_control'
    cases = targets | boundaries | edges | {control}
    assert result['preparation_complete'] and result['correctness_complete'] and result['mechanism_complete']
    assert len(result['prep_comparisons']) == 12
    assert {(x['case'], x['input_version']) for x in result['prep_comparisons']} == {
        (name, version) for name in targets | boundaries | edges for version in [0, 1]}
    for row in result['prep_comparisons']:
        assert row['passed'] and row['finite_queries']
        assert all(x['exact'] and x['unequal_byte_count'] == 0 for x in row['comparisons'])
        assert all(x['passed'] for x in row['immutable_and_untouched'])
    assert len(result['cases']) == 7 and {x['case']['name'] for x in result['cases']} == cases
    oracle_calls = 0
    for case in result['cases']:
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
        assert all(row['passed'] and all(x['exact'] for x in row['comparisons']) for row in case['cache_cross_arm'])
    assert len(result['concurrent_submissions']) == 2
    assert {x['case'] for x in result['concurrent_submissions']} == boundaries
    for row in result['concurrent_submissions']:
        assert row['passed'] and row['disjoint_writable_caches'] and row['separate_streams']
        assert row['actual_hardware_overlap_measured'] is False and len(row['checks']) == 2
        for check in row['checks']:
            assert check['passed'] and check['official_oracle_called']
            assert check['stable_addresses'] and check['immutable_inputs_and_untouched_rows']
            oracle_calls += 1
    activity = result['activity']
    assert activity['passed'] and activity['sessions'] == 1 and not activity['unassigned_cuda_events']
    assert not activity['event_names_truncated'] and len(activity['replay_checks']) == 4
    assert all(x['passed'] and x['exact_output_newkv_return'] for x in activity['replay_checks'])
    mechanism = []
    for case in sorted(targets):
        arms = {}
        for name in NAMES:
            label = 'UNSPLIT_MECHANISM_'+case+'_'+name
            events = activity['groups'][label]['events']
            physical = [x for x in events if x['name'] != label]
            counts = {kind: sum(x['kind'] == kind for x in physical)
                      for kind in ['mainloop', 'combine', 'schedule_prepare']}
            assert counts == dict(mainloop=1, combine=1 if name == 'baseline' else 0, schedule_prepare=1)
            arms[name] = dict(recognized_kernel_counts=counts, physical_kernel_count=len(physical),
                CUDA_marker_annotations=len(events)-len(physical),
                instrumented_mainloop_us=sum(x['end_us']-x['start_us'] for x in physical if x['kind'] == 'mainloop'))
        mechanism.append(dict(case=case, arms=arms))
    assert result['timing_complete'] and len(result['timings']) == 3
    assert {x['case']['name'] for x in result['timings']} == targets | {control}
    timings = []
    for row in result['timings']:
        assert len(row['warmups']) == 4 and len(row['samples']) == 12
        for i, sample in enumerate(row['warmups']+row['samples']):
            assert sample['index'] == i and sample['input_version'] == (i//2) % 2
            assert sample['order'] == (NAMES if i % 2 == 0 else NAMES[::-1])
            for name in NAMES:
                value, check = sample['raw_us'][name], sample['checks'][name]
                assert math.isfinite(value) and value > 0
                assert check['passed'] and check['exact_output_newkv_return'] is True
                assert not check['official_oracle_called'] and check['stable_addresses']
                assert check['immutable_inputs_and_untouched_rows']
        samples = row['samples']
        ratios = [s['raw_us']['baseline']/s['raw_us']['candidate'] for s in samples]
        pairs = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, 12, 2)]
        savings = [statistics.mean(samples[j]['raw_us']['baseline']-samples[j]['raw_us']['candidate']
                                   for j in [i, i+1]) for i in range(0, 12, 2)]
        ratio = statistics.median(pairs)
        timings.append(dict(case=row['case']['name'],
            raw_median_us={name: statistics.median(s['raw_us'][name] for s in samples) for name in NAMES},
            balanced_ratios=pairs, median_balanced_ratio=ratio,
            paired_candidate_slowdown_percent=100*(1/ratio-1),
            balanced_mean_savings_us=savings, median_balanced_mean_saving_us=statistics.median(savings),
            positive_pairs=sum(x > 1 for x in pairs)))
    target_admitted = all(x['median_balanced_mean_saving_us'] >= 2 and x['positive_pairs'] == 6
                          for x in timings if x['case'] in targets)
    control_admitted = next(x for x in timings if x['case'] == control)['paired_candidate_slowdown_percent'] <= 5
    admitted = target_admitted and control_admitted
    assert result['benefit']['admitted'] == admitted == result['official_check_admitted']
    return dict(source_freeze=FREEZE, raw_result_sha256=hashlib.sha256(raw).hexdigest(),
        validation_and_mechanism_passed=True, oracle_calls=oracle_calls, exact_return_A_checks=14,
        prep_groups=12, timed_replay_checks=96, activity_replay_checks=4,
        installed_wrapper_sha256=source['sha256'], observed_extension_hashes=identity['loaded_binaries'],
        mechanism=mechanism, timings=timings, official_check_admitted=admitted,
        decision='Reject combined sliding num_splits=1; no official check',
        qualifications=['One session, two fixtures and six balanced pairs per family; no independent-session confidence interval.',
                        'Profiler event groups contain CUDA marker annotations; physical kernel counts exclude them.',
                        'Mainloop activity grows, but no CTA/occupancy/stall cause was measured.',
                        'The installed Modal environment identity is not asserted identical to the official judge image.'])


if __name__ == '__main__':
    output = analyze()
    (HERE/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
