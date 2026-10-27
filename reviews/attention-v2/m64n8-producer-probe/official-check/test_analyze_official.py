"""CPU tests of raw/profile result analysis; no candidate import or service call."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('m64n8_official_analysis', HERE / 'analyze_official.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
SELECTED = HERE.parents[1] / 'fusion-lookahead-probe' / 'official-check'
SELECTED_ID = '5090b83f1d63407b'
CANDIDATE_SHA = '96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970'
TARGETS = {('decode 1', 'sliding'), ('decode 16', 'sliding'),
           ('decode 32', 'sliding'), ('decode 32', 'full')}
LOADS = tuple(f'{kind} {batch}' for kind in ('decode', 'verify') for batch in (1, 16, 32))


def row_key(row):
    return row['case']['load'], row['case']['layer']


def benchmark_rows(job):
    return {row_key(row): row for row in job['result']['records'] if row.get('event') == 'benchmark'}


def independent_score(rows, candidate_times=None):
    """Independent card arithmetic, with no analyzer implementation helpers."""
    speedups = []
    for load in LOADS:
        reference = sum(rows[(load, layer)]['baseline_ns'] * weight
                        for layer, weight in (('sliding', 46), ('full', 16)))
        candidate = sum((rows[(load, layer)]['median_ns'] if candidate_times is None
                         else candidate_times[(load, layer)]) * weight
                        for layer, weight in (('sliding', 46), ('full', 16)))
        speedups.append(reference / candidate)
    return math.exp(sum(math.log(value) for value in speedups) / 6)


def selected_fixture():
    return (json.loads((SELECTED / 'official-result.json').read_text()),
            json.loads((SELECTED / f'artifacts/candidate.{SELECTED_ID}.json').read_text()))


def changed_fixture(factors=None, reference_factors=None):
    """Pure copies of saved real shape/metadata receipts, plus synthetic times."""
    selected, old_profile = selected_fixture()
    job, profile = copy.deepcopy(selected), copy.deepcopy(old_profile)
    job['id'] = profile['job_id'] = '3e0a7ef51d4849c6'
    factors, reference_factors = factors or {}, reference_factors or {}
    for row in job['result']['records']:
        if row.get('event') != 'benchmark':
            continue
        key = row_key(row)
        candidate_factor, reference_factor = factors.get(key, 1), reference_factors.get(key, 1)
        row['median_ns'] *= candidate_factor
        row['baseline_ns'] *= reference_factor
        row['paired_ratio'] *= reference_factor / candidate_factor
    rows = benchmark_rows(job)
    for row in profile['benchmarks']:
        other = rows[row_key(row)]
        row.update({field: other[field] for field in ('median_ns', 'baseline_ns', 'paired_ratio')})
    job['result']['speedup'] = profile['summary']['speedup'] = independent_score(rows)
    request = {'job_id': job['id'], 'source_sha256': CANDIDATE_SHA}
    return job, profile, selected, old_profile, request


class OfficialAnalysis(unittest.TestCase):
    def test_exact_four_target_scope_and_same_baseline_zero_gain(self):
        self.assertEqual(a.TARGETS, TARGETS)
        self.assertEqual(a.CANDIDATE_SHA, CANDIDATE_SHA)
        out = a.analyze(*changed_fixture())
        self.assertAlmostEqual(out['official_speedup'], 1.2336330588627955, places=12)
        self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], 0, places=12)
        self.assertEqual(out['classification'], 'performance_rejected')
        self.assertEqual(len(out['shapes']), 12)
        self.assertEqual(sum(row['target'] for row in out['shapes']), 4)
        self.assertEqual(sum(not row['target'] for row in out['shapes']), 8)
        self.assertTrue(out['source_request_receipt_verified'])
        self.assertFalse(out['automatic_promotion'])

    def test_ten_percent_four_target_cuts_match_predeclared_weighted_model(self):
        args = changed_fixture({key: .9 for key in TARGETS})
        out = a.analyze(*args)
        old_rows = benchmark_rows(args[2])
        expected = 100 * (independent_score(old_rows, {
            key: row['median_ns'] * (.9 if key in TARGETS else 1)
            for key, row in old_rows.items()}) / independent_score(old_rows) - 1)
        self.assertAlmostEqual(expected, 3.0304768473572663, places=11)
        self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], expected, places=11)
        self.assertEqual(out['classification'], 'descriptive_gain_requires_confirmation')
        self.assertFalse(out['automatic_promotion'])

    def test_small_gain_and_any_changed_path_regression_fail_declared_gates(self):
        out = a.analyze(*changed_fixture({key: .95 for key in TARGETS}))
        self.assertFalse(out['gates']['target_only_modeled_gain_at_least3_percent'])
        self.assertEqual(out['classification'], 'performance_rejected')
        factors = {key: .5 for key in TARGETS}
        factors[('decode 1', 'sliding')] = 1.04
        out = a.analyze(*changed_fixture(factors))
        self.assertFalse(out['gates']['no_target_normalized_regression_above3_percent'])
        self.assertEqual(out['classification'], 'performance_rejected')

    def test_reference_normalization_uses_both_own_reference_times(self):
        factors = {key: 1.2 for key in TARGETS}
        out = a.analyze(*changed_fixture(factors, factors))
        self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], 0, places=12)
        for row in out['shapes']:
            self.assertAlmostEqual(row['historical_reference_normalized_latency_ratio'], 1, places=12)
        self.assertEqual(out['classification'], 'performance_rejected')

    def test_positive_and_negative_control_drift_flag_without_adding_model_gain(self):
        for factor in (1.04, .96):
            factors = {key: .8 for key in TARGETS}
            factors[('verify 16', 'full')] = factor
            out = a.analyze(*changed_fixture(factors))
            self.assertEqual(out['classification'], 'inconclusive_comparison')
            self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], 6.43825949355632, places=10)
            self.assertEqual(len(out['unchanged_control_drift_flags']), 1)
            self.assertFalse(out['gates']['no_unchanged_absolute_normalized_drift_above3_percent'])

    def test_entire_evaluator_dictionary_drift_is_not_hidden(self):
        for field in ('image', 'code', 'files'):
            args = changed_fixture({key: .8 for key in TARGETS})
            args[0]['result']['evaluator'][field] = {'synthetic_change': field}
            out = a.analyze(*args)
            self.assertEqual(out['classification'], 'inconclusive_comparison')
            self.assertFalse(out['gates']['same_evaluator'])
            self.assertTrue(out['evaluator']['differences'])

    def test_driver_drift_is_inconclusive_and_missing_gpu_identity_unknown(self):
        args = changed_fixture({key: .8 for key in TARGETS})
        args[0]['result']['machine']['gpu_before']['driver_version'] = 'different-driver'
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'inconclusive_comparison')
        self.assertFalse(out['gates']['compatible_reported_GPU'])
        args = changed_fixture({key: .8 for key in TARGETS})
        args[0]['result']['machine']['gpu_before'].pop('driver_version')
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'inconclusive_comparison')
        self.assertFalse(out['gpu_environment']['stable_fields_known'])

    def test_node_or_temperature_context_is_retained_without_identity_claim(self):
        args = changed_fixture({key: .8 for key in TARGETS})
        args[0]['result']['machine']['node'] = 'different-context-node'
        args[0]['result']['machine']['gpu_before']['temperature.gpu'] = 40
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'descriptive_gain_requires_confirmation')
        self.assertTrue(out['gpu_environment']['all_reported_machine_differences'])
        self.assertTrue(out['gates']['compatible_reported_GPU'])

    def test_original_prerequisite_failure_without_timings_keeps_source_proof(self):
        args = changed_fixture()
        failed = copy.deepcopy(next(row for row in args[0]['result']['records'] if row.get('event') == 'test'))
        failed.update(status='fail', message='original oracle mismatch')
        args[0]['result'].update(passed=False, returncode=1, error='original oracle mismatch', records=[failed])
        out = a.analyze(args[0], None, args[2], args[3], args[4])
        self.assertEqual(out['classification'], 'original_prerequisite_failure')
        self.assertTrue(out['source_request_receipt_verified'])
        self.assertEqual(out['correctness']['prerequisites_recorded'], 1)
        self.assertEqual(out['correctness']['prerequisites_passed'], 0)
        self.assertNotIn('shapes', out)
        self.assertNotIn('official_speedup', out)
        self.assertNotIn('target_only_fixed_reference_modeled_gain_percent', out)

    def test_benchmark_correctness_failure_is_early_and_not_a_timing_result(self):
        args = changed_fixture()
        tests = [row for row in args[0]['result']['records'] if row.get('event') == 'test']
        failed = copy.deepcopy(next(row for row in args[0]['result']['records'] if row.get('event') == 'benchmark'))
        failed.update(status='fail', message='benchmark fixture oracle mismatch')
        for field in ('median_ns', 'baseline_ns', 'paired_ratio'):
            failed.pop(field)
        args[0]['result'].update(passed=False, returncode=1, error='benchmark fixture oracle mismatch', records=tests + [failed])
        out = a.analyze(args[0], None, args[2], args[3], args[4])
        self.assertEqual(out['classification'], 'original_benchmark_failure')
        self.assertEqual(out['correctness']['benchmarks_recorded'], 1)
        self.assertEqual(out['correctness']['benchmarks_passed'], 0)
        self.assertEqual(out['correctness']['benchmarks_failed'], 1)
        self.assertEqual(out['correctness']['first_failed_benchmark_case'], failed['case'])
        self.assertTrue(out['source_request_receipt_verified'])
        self.assertNotIn('shapes', out)
        self.assertNotIn('official_speedup', out)

    def test_source_receipt_mismatch_is_invalid_even_before_failure_analysis(self):
        for field in ('job_id', 'source_sha256'):
            args = changed_fixture()
            args[4][field] = 'wrong-binding'
            args[0]['result'].update(passed=False, returncode=1, records=[])
            out = a.analyze(args[0], None, args[2], args[3], args[4])
            self.assertEqual(out['classification'], 'invalid_receipt')
            self.assertFalse(out['source_request_receipt_verified'])
            self.assertNotIn('shapes', out)

    def test_bound_receipt_for_another_job_cannot_replace_this_one_check(self):
        args = changed_fixture({key: .8 for key in TARGETS})
        args[0]['id'] = args[1]['job_id'] = args[4]['job_id'] = 'another-consistently-bound-job'
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'invalid_receipt')
        self.assertNotIn('shapes', out)
        self.assertFalse(out['automatic_promotion'])

    def test_unbound_gain_and_contradictory_profile_cannot_pass(self):
        args = changed_fixture({key: .8 for key in TARGETS})
        out = a.analyze(*args[:-1])
        self.assertEqual(out['classification'], 'inconclusive_comparison')
        self.assertIsNone(out['source_request_receipt_verified'])
        args[1]['summary']['passed'] = False
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'inconclusive')
        self.assertFalse(out['automatic_promotion'])

    def test_duplicate_prerequisites_and_flagged_timings_cannot_pass(self):
        args = changed_fixture({key: .8 for key in TARGETS})
        tests = [row for row in args[0]['result']['records'] if row.get('event') == 'test']
        args[0]['result']['records'] = [copy.deepcopy(tests[0]) for _ in range(16)] + list(benchmark_rows(args[0]).values())
        self.assertEqual(a.analyze(*args)['classification'], 'incomplete_or_flagged_evidence')
        args = changed_fixture({key: .8 for key in TARGETS})
        next(iter(benchmark_rows(args[0]).values()))['flags'] = ['synthetic_flag']
        self.assertEqual(a.analyze(*args)['classification'], 'incomplete_or_flagged_evidence')


if __name__ == '__main__':
    unittest.main()
