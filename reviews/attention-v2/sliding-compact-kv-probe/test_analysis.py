"""Offline result gates; no Torch imports or service calls."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('compact_official_analysis', HERE/'analyze_official.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def selected_fixture():
    return (json.loads((a.SELECTED/'official-result.json').read_text()),
            json.loads((a.SELECTED/'artifacts/candidate.5090b83f1d63407b.json').read_text()))


def changed_fixture(factors=None):
    selected, old_profile = selected_fixture()
    job, profile = copy.deepcopy(selected), copy.deepcopy(old_profile)
    job['id'] = profile['job_id'] = 'compact-synthetic-check'
    factors = factors or {}
    for row in job['result']['records']:
        if row.get('event') != 'benchmark':
            continue
        factor = factors.get(a.key(row), 1)
        row['median_ns'] *= factor
        row['paired_ratio'] /= factor
    rows = a.rows_from_job(job)
    for row in profile['benchmarks']:
        other = rows[a.key(row)]
        row.update(median_ns=other['median_ns'], paired_ratio=other['paired_ratio'])
    score, _ = a.score_from_rows(rows)
    job['result']['speedup'] = profile['summary']['speedup'] = score
    request = {'job_id': job['id'], 'source_sha256': a.CANDIDATE_SHA}
    return job, profile, selected, old_profile, request


class OfficialAnalysis(unittest.TestCase):
    def test_saved_selected_reconstructs_and_same_source_has_zero_gain(self):
        selected, profile = selected_fixture()
        out = a.analyze(selected, profile, selected, profile)
        self.assertAlmostEqual(out['official_speedup'], 1.2336330588627955, places=12)
        self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], 0, places=12)
        self.assertEqual(out['classification'], 'performance_rejected')
        self.assertEqual(len(out['shapes']), 12)
        self.assertFalse(out['automatic_promotion'])

    def test_three_target_quarter_cut_matches_prospective_card(self):
        out = a.analyze(*changed_fixture({k: .75 for k in a.TARGETS}))
        self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], 4.692787, places=5)
        self.assertEqual(out['classification'], 'descriptive_gain_requires_confirmation')
        self.assertEqual(sum(row['target'] for row in out['shapes']), 3)
        self.assertFalse(out['automatic_promotion'])

    def test_small_gain_and_single_target_regression_fail_declared_gates(self):
        out = a.analyze(*changed_fixture({k: .9 for k in a.TARGETS}))
        self.assertEqual(out['classification'], 'performance_rejected')
        self.assertFalse(out['gates']['target_only_modeled_gain_at_least3_percent'])
        factors = {k: .5 for k in a.TARGETS}
        factors[('verify 32', 'sliding')] = 1.04
        out = a.analyze(*changed_fixture(factors))
        self.assertEqual(out['classification'], 'performance_rejected')
        self.assertFalse(out['gates']['no_target_normalized_regression_above3_percent'])

    def test_control_drift_is_not_counted_as_candidate_benefit(self):
        factors = {k: .75 for k in a.TARGETS}
        factors[('decode 16', 'full')] = 1.04
        out = a.analyze(*changed_fixture(factors))
        self.assertEqual(out['classification'], 'inconclusive_comparison')
        self.assertAlmostEqual(out['target_only_fixed_reference_modeled_gain_percent'], 4.692787, places=5)
        self.assertEqual(len(out['unchanged_control_drift_flags']), 1)

    def test_evaluator_or_stable_gpu_changes_make_comparison_inconclusive(self):
        args = changed_fixture({k: .75 for k in a.TARGETS})
        args[0]['result']['evaluator']['image'] = 'different-image'
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'inconclusive_comparison')
        self.assertFalse(out['gates']['same_evaluator'])
        args = changed_fixture({k: .75 for k in a.TARGETS})
        args[0]['result']['machine']['gpu_before']['driver_version'] = 'different-driver'
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'inconclusive_comparison')
        self.assertFalse(out['gates']['compatible_reported_GPU'])

    def test_machine_context_is_reported_without_false_binary_identity(self):
        args = changed_fixture({k: .75 for k in a.TARGETS})
        args[0]['result']['machine']['node'] = 'different-node'
        args[0]['result']['machine']['gpu_before']['temperature.gpu'] = 40
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'descriptive_gain_requires_confirmation')
        self.assertTrue(out['gpu_environment']['all_reported_machine_differences'])
        self.assertEqual(out['new_metadata']['binary_identity'], 'UNKNOWN')

    def test_failed_prerequisite_without_any_timings_does_not_crash(self):
        selected, profile = selected_fixture()
        job = copy.deepcopy(selected)
        job['id'] = 'failed-compact'
        failure = copy.deepcopy(job['result']['records'][0])
        failure['status'] = 'fail'
        job['result'].update(passed=False, returncode=1, error='original oracle failed', records=[failure])
        out = a.analyze(job, None, selected, profile)
        self.assertEqual(out['classification'], 'original_prerequisite_failure')
        self.assertNotIn('official_speedup', out)
        self.assertEqual(out['correctness']['prerequisites_recorded'], 1)

    def test_failed_native_prerequisite_is_not_labeled_numerical_error(self):
        args = changed_fixture()
        test = copy.deepcopy(args[0]['result']['records'][0])
        test.update(status='fail', message='RuntimeError: unsupported contiguous native route')
        args[0]['result'].update(passed=False, returncode=1,
            error='RuntimeError: unsupported native route', records=[test])
        out = a.analyze(args[0], None, args[2], args[3], args[4])
        self.assertEqual(out['classification'], 'original_prerequisite_failure')
        self.assertTrue(out['source_request_receipt_verified'])
        self.assertIn('unsupported', out['correctness']['first_failed_message'])

    def test_duplicate_prerequisites_cannot_count_as_all16(self):
        args = changed_fixture({k: .75 for k in a.TARGETS})
        records = args[0]['result']['records']
        first = next(r for r in records if r['event'] == 'test')
        args[0]['result']['records'] = [copy.deepcopy(first) for _ in range(16)] + [r for r in records if r['event'] != 'test']
        self.assertEqual(a.analyze(*args)['classification'], 'incomplete_or_flagged_evidence')

    def test_contradictory_profile_summary_is_not_accepted(self):
        args = changed_fixture({k: .75 for k in a.TARGETS})
        args[1]['summary']['passed'] = False
        self.assertEqual(a.analyze(*args)['classification'], 'inconclusive')

    def test_native_failure_pending_and_duplicate_are_distinct(self):
        selected, profile = selected_fixture()
        job = copy.deepcopy(selected)
        job['result'].update(passed=False, returncode=1, error='unsupported native route', records=[])
        self.assertEqual(a.analyze(job, None, selected, profile)['classification'], 'native_or_execution_failure')
        job['status'] = 'running'
        self.assertEqual(a.analyze(job, None, selected, profile)['classification'], 'not_terminal')
        job['duplicate_of'] = 'old-job'
        self.assertEqual(a.analyze(job, None, selected, profile)['classification'], 'inconclusive')

    def test_profile_disagreement_and_unbound_source_block_clean_gain(self):
        args = changed_fixture({k: .75 for k in a.TARGETS})
        args[1]['benchmarks'][0]['median_ns'] += 1
        self.assertEqual(a.analyze(*args)['classification'], 'inconclusive')
        args = changed_fixture({k: .75 for k in a.TARGETS})
        args[-1]['source_sha256'] = 'wrong-source'
        out = a.analyze(*args)
        self.assertEqual(out['classification'], 'invalid_receipt')
        self.assertFalse(out['source_request_receipt_verified'])

    def test_metadata_is_bounded_and_never_resolves_pack_from_input_schema(self):
        marker = (a.MARKER + ' torch=2.14.1+cu130 cuda=13.0 '
                  'fwd_schema=sgl_kernel::fwd(Tensor q, bool? pack_gqa=None) -> Tensor[]')
        out = a.metadata_from_log('some log\n' + marker)
        self.assertEqual(out['status'], 'observed_API_metadata')
        self.assertEqual(out['resolved_pack_gqa'], 'UNKNOWN')
        self.assertEqual(out['resolved_splits'], 'UNKNOWN')
        self.assertEqual(out['internal_QK_PV_dtype'], 'UNKNOWN')
        self.assertEqual(a.metadata_from_log('') ['status'], 'UNKNOWN')
        self.assertEqual(a.metadata_from_log(marker + '\n' + marker)['identical_marker_observations'], 2)
        self.assertEqual(a.metadata_from_log(marker + '\n' + marker + 'different')['status'], 'UNKNOWN')
        self.assertEqual(a.metadata_from_log(a.MARKER + ' torch=t cuda=c fwd_schema=' + 'x'*4097)['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
