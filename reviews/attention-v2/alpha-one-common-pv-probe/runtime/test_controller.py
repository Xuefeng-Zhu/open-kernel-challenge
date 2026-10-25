"""Offline guard/transport tests. No Modal SDK import, auth read or GPU launch."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('common_pv_runtime_controller', HERE/'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ControllerGuards(unittest.TestCase):
    def setUp(self):
        self.current = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        self.head = 'a'*40
        self.slot = dict(state='granted', experiment=c.EXPERIMENT, source_commit=self.head,
            gpu='H200', granted_utc=self.current.isoformat(),
            expires_utc=(self.current+timedelta(minutes=10)).isoformat())
        self.receipt = dict(workspace='xuefeng-zhu', verified_net_spend_cap_usd='0',
            total_spend_usd='0', changes_made=False, observed_utc=self.current.isoformat(),
            verified_credit_balance_usd='10', total_usage_usd='2')
        self.billing = dict(billed_cost='0', metered_cost='2.1',
                            metered_cost_breakdown={'gpu': '2.123'})

    def test_slot_requires_exact_parent_grant_freshness_and_enough_time(self):
        c.validate_slot(self.slot, self.head, self.current)
        for changes in [dict(state='idle'), dict(experiment='projection'),
                        dict(source_commit='b'*40), dict(gpu='H100'),
                        dict(granted_utc=(self.current-timedelta(minutes=16)).isoformat()),
                        dict(expires_utc=(self.current+timedelta(seconds=200)).isoformat())]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_slot({**self.slot, **changes}, self.head, self.current)

    def test_budget_requires_free_credits_and_paid_zero(self):
        self.assertEqual(c.validate_budget(self.receipt, self.billing, Decimal('.7'), self.current),
                         Decimal('9.877'))
        for changes in [dict(verified_net_spend_cap_usd='1'), dict(total_spend_usd='.01'),
                        dict(changes_made=True), dict(workspace='other'),
                        dict(verified_credit_balance_usd='.5'),
                        dict(observed_utc=(self.current-timedelta(minutes=16)).isoformat())]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_budget({**self.receipt, **changes}, self.billing, Decimal('.7'), self.current)
        with self.assertRaises(ValueError):
            c.validate_budget(self.receipt, {**self.billing, 'billed_cost': '.01'}, Decimal('.7'), self.current)

    def test_reserve_uses_full_watchdog_double_rate_and_slack(self):
        rate = c.resource_rate(dict(gpu_hour_cost_h200='4.54', cpu_hour_cost='.0473',
                                    mem_gib_hour_cost='.008'))
        self.assertEqual(c.reserve(rate), Decimal('.58'))
        self.assertGreater(c.reserve(rate), Decimal(210)*rate*2+Decimal('.02'))
        with self.assertRaises(ValueError):
            c.reserve(rate*2)

    def test_stopped_rejects_stopping_and_nonzero_tasks(self):
        self.assertTrue(c.stopped([dict(state='stopped', tasks=0)]))
        self.assertFalse(c.stopped([dict(state='stopping', tasks=0)]))
        self.assertFalse(c.stopped([dict(state='stopped', tasks=1)]))

    def test_result_transport_rejects_tamper_overflow_and_nonobject(self):
        raw = b'{"passed":true}'
        reply = dict(exit_code=0, report_json=raw.decode(), report_sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(c.verified_result(reply), (raw, {'passed': True}))
        bad = [{**reply, 'report_sha256': '0'*64}, {**reply, 'error': 'missing'},
               {**reply, 'report_json': ' '*500001}, {**reply, 'exit_code': '0'}]
        for value in bad:
            with self.assertRaises(ValueError):
                c.verified_result(value)
        array = '[]'
        with self.assertRaises(ValueError):
            c.verified_result(dict(exit_code=0, report_json=array,
                                   report_sha256=hashlib.sha256(array.encode()).hexdigest()))

    def test_payload_and_remote_entry_do_not_capture_credentials_or_config(self):
        paths = c.payload()
        self.assertEqual(len(paths), 9)
        self.assertTrue(all(p.is_relative_to(c.REPO) for p in paths.values()))
        self.assertFalse(any('local-notes' in key or '.local.' in key for key in paths))
        self.assertEqual(paths[(c.PROBE/'candidate.py').relative_to(c.REPO).as_posix()],
                         c.ATTENTION/'alpha-one-common-pv-probe/candidate.py')
        self.assertIn((c.HERE/'worker.py').relative_to(c.REPO).as_posix(), paths)
        self.assertIsNone(c.make_remote_entry().__closure__)
        self.assertLess(c.WORKER_SECONDS, c.CHILD_SECONDS)
        self.assertLess(c.CHILD_SECONDS, c.FUNCTION_SECONDS)
        self.assertLess(c.FUNCTION_SECONDS+c.STARTUP_SECONDS, c.WATCHDOG_SECONDS)

    def test_offline_preflight_only_reads_committed_payload_and_compiles(self):
        calls = []
        paths = c.payload()
        controller_key = Path(c.__file__).relative_to(c.REPO).as_posix()
        def git_output(args, **kwargs):
            calls.append(args)
            if args == ['git', 'rev-parse', 'HEAD']:
                return self.head+'\n'
            if args[:2] == ['git', 'show']:
                name = args[2].split(':', 1)[1]
                return (Path(c.__file__) if name == controller_key else paths[name]).read_bytes()
            raise AssertionError(args)
        # The unit fixture keeps this guard test independent of a simultaneously
        # prepared worker. The real frozen-head command must still parse the
        # completed committed worker before any launch.
        with tempfile.TemporaryDirectory() as folder:
            worker = Path(folder)/'worker.py'
            worker.write_text('"""Offline payload fixture; never executed."""\n')
            paths[(c.HERE/'worker.py').relative_to(c.REPO).as_posix()] = worker
            with patch.object(c, 'payload', return_value=paths), \
                 patch.object(c.subprocess, 'check_output', side_effect=git_output), \
                 patch.object(c, 'credential_environment', side_effect=AssertionError('auth touched')), \
                 patch.object(c, 'cli', side_effect=AssertionError('network touched')):
                paths, hashes = c.frozen_payload(self.head)
        self.assertEqual(len(hashes), len(paths)+1)
        self.assertEqual(hashes[(c.PROBE/'candidate.py').relative_to(c.REPO).as_posix()], c.CANDIDATE_SHA)
        self.assertTrue(all(command[:2] in (['git', 'rev-parse'], ['git', 'show']) for command in calls))
        with self.assertRaises(ValueError):
            c.frozen_payload('HEAD')

    def completion_fixture(self):
        worker_key = (c.HERE/'worker.py').relative_to(c.REPO).as_posix()
        diagnostic_key = (c.HERE/'diagnostic.py').relative_to(c.REPO).as_posix()
        sources = {'baseline.py': c.BASE_SHA, 'candidate.py': c.CANDIDATE_SHA}
        report = dict(passed=True, status='complete', correctness_complete=True,
            diagnostic_complete=True, timing_complete=True,
            source_sha256=sources, support_sha256=c.SUPPORT_HASHES,
            diagnostic_sha256=c.DIAGNOSTIC_SHA,
            verified_hashes={**sources, **c.SUPPORT_HASHES, 'diagnostic.py': c.DIAGNOSTIC_SHA},
            worker_sha256='f'*64, counts=c.FULL_COUNTS,
            benefit={'admitted': True}, frequency_gate={'opportunity': True, 'scored_fast_counts': [10, 20]},
            timings=[{'case': 'target'}, {'case': 'unchanged_control'}])
        return {'exit_code': 0}, report, {worker_key: 'f'*64, diagnostic_key: c.DIAGNOSTIC_SHA}

    def test_claimed_completion_requires_exact_native_contract(self):
        reply, report, hashes = self.completion_fixture()
        c.validate_success(reply, report, hashes)
        for changes in [dict(status='partial'), dict(correctness_complete=False),
                        dict(diagnostic_complete=False), dict(source_sha256={}), dict(support_sha256={}),
                        dict(verified_hashes={}), dict(worker_sha256='a'*64), dict(diagnostic_sha256='a'*64),
                        dict(counts={}), dict(benefit={'admitted': 'true'}), dict(timings=[]),
                        dict(frequency_gate={'opportunity': False}),
                        dict(frequency_gate={'opportunity': True, 'scored_fast_counts': [0, 0]}),
                        dict(frequency_gate={'opportunity': True, 'scored_fast_counts': [True, 1]}),
                        dict(official_check_admitted=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_success(reply, {**report, **changes}, hashes)
        with self.assertRaises(ValueError):
            c.validate_success({'exit_code': 1}, report, hashes)
        c.validate_success({'exit_code': 124}, {'passed': False, 'status': 'partial'}, {})

    def test_negative_timed_benefit_is_valid_complete_not_execution_error(self):
        reply, report, hashes = self.completion_fixture()
        report['benefit'] = {'admitted': False, 'reason': 'Per-seed performance gate failed'}
        c.validate_success(reply, report, hashes)
        self.assertFalse(report['benefit']['admitted'])
        self.assertNotIn('official_check_admitted', report)

    def test_any_zero_frequency_can_skip_timing_only_as_explicit_negative(self):
        reply, report, hashes = self.completion_fixture()
        report.update(timing_complete=False, counts=c.NO_OPPORTUNITY_COUNTS, timings=[],
            frequency_gate={'opportunity': False, 'scored_fast_counts': [0, 0]}, benefit={'admitted': False})
        c.validate_success(reply, report, hashes)
        c.validate_success(reply, {**report,
            'frequency_gate': {'opportunity': False, 'scored_fast_counts': [0, 123]}}, hashes)
        for changes in [dict(benefit={'admitted': True}), dict(counts=c.FULL_COUNTS),
                        dict(frequency_gate={'opportunity': False, 'scored_fast_counts': [1, 1]}),
                        dict(timings=[{'invented': True}])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_success(reply, {**report, **changes}, hashes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
