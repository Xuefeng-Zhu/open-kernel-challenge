"""Offline guard/transport tests. No Modal SDK import, auth read or GPU launch."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unsplit_controller', HERE/'controller.py')
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
        self.assertIn((c.HERE/'identity.py').relative_to(c.REPO).as_posix(), paths)
        self.assertTrue(all(p.is_relative_to(c.REPO) for p in paths.values()))
        self.assertFalse(any('local-notes' in key or '.local.' in key for key in paths))
        self.assertIsNone(c.make_remote_entry().__closure__)
        self.assertLess(c.WORKER_SECONDS, c.CHILD_SECONDS)
        self.assertLess(c.CHILD_SECONDS, c.FUNCTION_SECONDS)
        self.assertLess(c.FUNCTION_SECONDS+c.STARTUP_SECONDS, c.WATCHDOG_SECONDS)

    def test_offline_preflight_only_reads_committed_payload_and_compiles(self):
        calls = []
        def git_output(args, **kwargs):
            calls.append(args)
            if args == ['git', 'rev-parse', 'HEAD']:
                return self.head+'\n'
            if args[:2] == ['git', 'show']:
                return (c.REPO/args[2].split(':', 1)[1]).read_bytes()
            raise AssertionError(args)
        with patch.object(c.subprocess, 'check_output', side_effect=git_output), \
             patch.object(c, 'credential_environment', side_effect=AssertionError('auth touched')), \
             patch.object(c, 'cli', side_effect=AssertionError('network touched')):
            paths, hashes = c.frozen_payload(self.head)
        self.assertEqual(len(hashes), len(paths)+1)
        self.assertEqual(hashes[(c.PROBE/'candidate.py').relative_to(c.REPO).as_posix()], c.CANDIDATE_SHA)
        self.assertTrue(all(command[:2] in (['git', 'rev-parse'], ['git', 'show']) for command in calls))
        with self.assertRaises(ValueError):
            c.frozen_payload('HEAD')


if __name__ == '__main__':
    unittest.main(verbosity=2)
