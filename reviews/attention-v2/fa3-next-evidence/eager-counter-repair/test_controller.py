from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('eager_counter_controller', HERE/'controller.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 27, 4, 0, tzinfo=timezone.utc)
        self.billing = {'metered_cost': '2.5', 'billed_cost': '0',
            'metered_cost_breakdown': {'apps': '2.5'}}
        self.receipt = {'workspace': 'xuefeng-zhu', 'verified_net_spend_cap_usd': '0',
            'total_spend_usd': '0', 'changes_made': False,
            'observed_utc': self.now.isoformat(), 'verified_credit_balance_usd': '27.5',
            'total_usage_usd': '2.5'}

    def test_payload_is_flat_allowlist_without_auth(self):
        paths = c.payload()
        self.assertEqual(len(paths), 4)
        self.assertTrue(all('/' not in key for key in paths))
        self.assertFalse(any('local-notes' in str(path) for path in paths.values()))
        self.assertEqual(set(paths), {'worker.py', 'counter_helper.py', 'LICENSE', 'THIRD_PARTY_NOTICES.md'})
        self.assertEqual(paths['counter_helper.py'].parent.name, 'baseline-runtime-identity')
        self.assertFalse(any(key in paths for key in ('selected.py', 'reference.py', 'checks.py')))

    def test_remote_entry_has_no_closure_or_auth(self):
        fn = c.make_remote_entry()
        self.assertIsNone(fn.__closure__)
        self.assertFalse(any(name in fn.__code__.co_names for name in ('AUTH', 'cli', 'configure_existing_auth')))

    def test_guardrail_credit_floor_tracks_increment(self):
        self.billing['metered_cost'] = '2.6'
        self.assertEqual(c.validate_budget(self.receipt, self.billing, self.now), Decimal('27.4'))

    def test_guardrail_rejects_paid_stale_and_exhausted(self):
        for field, value in [('verified_net_spend_cap_usd', '1'), ('total_spend_usd', '1'),
                ('changes_made', True), ('verified_credit_balance_usd', '.32'),
                ('observed_utc', (self.now-timedelta(minutes=16)).isoformat())]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    c.validate_budget({**self.receipt, field: value}, self.billing, self.now)
        with self.assertRaises(ValueError):
            c.validate_budget(self.receipt, {**self.billing, 'billed_cost': '.01'}, self.now)

    def test_reserve_uses_live_whole_resource_rate(self):
        rates = {'gpu_hour_cost_h200': '4.548', 'cpu_hour_cost': '.0474', 'mem_gib_hour_cost': '.0081'}
        rate = c.resource_rate(rates)
        self.assertEqual(c.reserve(rate), Decimal('.32'))
        with self.assertRaises(ValueError):
            c.reserve(Decimal('1'))

    def test_source_bound_slot(self):
        head = 'a'*40
        slot = {'state': 'granted', 'experiment': c.EXPERIMENT, 'source_commit': head,
            'gpu': 'H200', 'granted_utc': self.now.isoformat(),
            'expires_utc': (self.now+timedelta(minutes=10)).isoformat()}
        c.validate_slot(slot, head, self.now)
        with self.assertRaises(ValueError):
            c.validate_slot(slot, 'b'*40, self.now)
        with self.assertRaises(ValueError):
            c.validate_slot({**slot, 'expires_utc': self.now.isoformat()}, head, self.now)

    def test_stopped_requires_exact_state_and_zero_tasks(self):
        self.assertTrue(c.stopped([]))
        self.assertTrue(c.stopped([{'state': 'stopped', 'tasks': 0}]))
        self.assertFalse(c.stopped([{'state': 'stopped', 'tasks': 1}]))
        self.assertFalse(c.stopped([{'state': 'running', 'tasks': 0}]))
        self.assertFalse(c.stopped([{'state': 'stopped'}]))

    def test_auth_output_never_forwards_identifier(self):
        out = 'Token: NEVER_EMIT_ME\nWorkspace: xuefeng-zhu (private-id)\nExpires at: 2026-10-27 15:21:50 UTC\n'
        with patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, out, '')) as run:
            result = c.verify_existing_auth(self.now)
        self.assertEqual(result['workspace'], 'xuefeng-zhu')
        self.assertNotIn('NEVER_EMIT_ME', json.dumps(result))
        self.assertEqual(run.call_args.kwargs['env']['TZ'], 'UTC')
        with patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, out, 'SECRET')):
            with self.assertRaisesRegex(RuntimeError, '^Existing Modal authentication status failed; no renewal$'):
                c.verify_existing_auth(self.now)

    def test_auth_rejects_wrong_workspace_unbounded_and_expiring(self):
        for out in ('Workspace: other\nExpires at: 2026-10-27 15:21:50 UTC\n',
                'Workspace: xuefeng-zhu\n',
                'Workspace: xuefeng-zhu\nExpires at: 2026-10-27 04:01:50 UTC\n'):
            with patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, out, '')):
                with self.assertRaises(ValueError):
                    c.verify_existing_auth(self.now)

    def reply(self, report):
        raw = json.dumps(report).encode()
        return {'exit_code': 0, 'report_json': raw.decode(),
            'report_sha256': hashlib.sha256(raw).hexdigest()}

    def test_transport_retains_failure_and_verifies_identity(self):
        reply = self.reply({'passed': False, 'status': 'blocked'})
        reply['exit_code'] = 2
        self.assertEqual(c.verified_result(reply)[1]['status'], 'blocked')
        self.assertEqual(c.verified_result({'exit_code': 124, 'error': 'timeout'}), (None, None))
        with patch.object(c, 'MAX_RESULT', 1):
            with self.assertRaises(ValueError):
                c.verified_result(reply)
        reply['report_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            c.verified_result(reply)

    def test_completion_requires_frozen_bindings_and_no_broadened_scope(self):
        hashes = {'counter_helper.py': 'a'*64, 'worker.py': 'b'*64, 'controller.py': 'c'*64}
        report = {'worker_sha256': hashes['worker.py'],
            'verified_hashes': {'counter_helper.py': 'a'*64},
            'passed': True, 'status': 'complete', 'fa3_kernel_executed': False,
            'selected_candidate_changed': False, 'security_changes': False}
        self.assertTrue(c.validate_report(report, hashes))
        self.assertFalse(c.validate_report({**report, 'passed': False}, hashes))
        with self.assertRaises(ValueError):
            c.validate_report({**report, 'worker_sha256': 'd'*64}, hashes)
        for field in ('fa3_kernel_executed', 'selected_candidate_changed', 'security_changes'):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    c.validate_report({**report, field: True}, hashes)

    def test_timeout_envelope_and_serial_single_app(self):
        self.assertEqual((c.WORKER_SECONDS, c.CHILD_SECONDS), (65, 75))
        self.assertEqual((c.FUNCTION_SECONDS, c.STARTUP_SECONDS, c.WATCHDOG_SECONDS), (85, 20, 110))
        self.assertLess(c.WORKER_SECONDS, c.CHILD_SECONDS)
        self.assertLess(c.CHILD_SECONDS, c.FUNCTION_SECONDS)
        self.assertLess(c.FUNCTION_SECONDS+c.STARTUP_SECONDS, c.WATCHDOG_SECONDS)


if __name__ == '__main__':
    unittest.main()
