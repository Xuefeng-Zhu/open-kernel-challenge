"""Offline CPU inventory admission/transport tests; no service or worker action."""
import ast
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
spec = importlib.util.spec_from_file_location('fa3_image_inventory_controller', HERE/'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CPUInventoryController(unittest.TestCase):
    def setUp(self):
        self.current = datetime(2026, 10, 25, 23, 0, tzinfo=timezone.utc)
        self.receipt = dict(workspace='xuefeng-zhu', verified_net_spend_cap_usd='0', total_spend_usd='0',
            changes_made=False, observed_utc=self.current.isoformat(),
            verified_credit_balance_usd='10', total_usage_usd='2')
        self.billing = dict(billed_cost='0', metered_cost='2.1', metered_cost_breakdown={'cpu': '2.123'})

    def test_no_gpu_keyword_and_fixed_cpu_bounds(self):
        tree = ast.parse((HERE/'controller.py').read_text())
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'function']
        self.assertEqual(len(functions), 1)
        values = {}
        for key in functions[0].keywords:
            if key.arg == 'image':
                continue
            values[key.arg] = getattr(c, key.value.id) if isinstance(key.value, ast.Name) else ast.literal_eval(key.value)
        self.assertNotIn('gpu', values)
        for key, want in dict(cpu=(1, 1), memory=(2048, 2048), min_containers=0, max_containers=1,
                buffer_containers=0, retries=0, timeout=45, startup_timeout=10, serialized=True,
                include_source=False, block_network=True, restrict_modal_access=True).items():
            self.assertEqual(values[key], want)
        self.assertEqual((c.WORKER_SECONDS, c.CHILD_SECONDS, c.FUNCTION_SECONDS,
                          c.STARTUP_SECONDS, c.WATCHDOG_SECONDS), (30, 35, 45, 10, 60))

    def test_cpu_reserve_never_uses_gpu_prices(self):
        rate = c.resource_rate({'cpu_hour_cost': '.0473', 'mem_gib_hour_cost': '.008'})
        self.assertEqual(rate, (Decimal('.0473')+2*Decimal('.008'))/3600)
        self.assertLess(c.reserve(rate), c.RESERVE)
        self.assertEqual(c.RESERVE, Decimal('.02'))
        self.assertEqual(c.reserve(rate), 2*Decimal(60)*rate+Decimal('.005'))
        with self.assertRaises(ValueError):
            c.reserve(rate*100)

    def test_fresh_zero_paid_and_free_credit_guard(self):
        self.assertEqual(c.validate_budget(self.receipt, self.billing, self.current), Decimal('9.877'))
        for changes in [dict(workspace='other'), dict(verified_net_spend_cap_usd='1'),
                        dict(total_spend_usd='.01'), dict(changes_made=True),
                        dict(verified_credit_balance_usd='.01'),
                        dict(observed_utc=(self.current-timedelta(minutes=16)).isoformat()),
                        dict(observed_utc=(self.current+timedelta(seconds=6)).isoformat()),
                        dict(observed_utc='2026-10-25T23:00:00')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_budget({**self.receipt, **changes}, self.billing, self.current)
        with self.assertRaises(ValueError):
            c.validate_budget(self.receipt, {**self.billing, 'billed_cost': '.01'}, self.current)

    def test_payload_allowlist_and_no_remote_local_closure(self):
        paths = c.payload()
        self.assertEqual(len(paths), 3)
        self.assertEqual(set(path.name for path in paths.values()), {'worker.py', 'LICENSE', 'THIRD_PARTY_NOTICES.md'})
        self.assertTrue(all(path.is_relative_to(c.REPO) for path in paths.values()))
        self.assertFalse(any('local-notes' in name or '.local.' in name for name in paths))
        self.assertEqual(c.REPO, HERE.parents[3])
        self.assertIsNone(c.make_remote_entry().__closure__)
        tree = ast.parse((HERE/'controller.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertEqual(sum(node.func.attr == 'remote' for node in calls), 1)
        self.assertFalse(any(node.func.attr in {'add_local_dir', 'pip_install', 'apt_install'} for node in calls))

    def test_transport_requires_exact_worker_and_cpu_only_flags(self):
        report = dict(worker_sha256='f'*64, gpu_requested=False, cuda_context_requested=False, candidate_kernel_launches=0)
        def reply(data):
            raw = json.dumps(data)
            return dict(exit_code=0, report_json=raw, report_sha256=hashlib.sha256(raw.encode()).hexdigest())
        good = reply(report)
        self.assertEqual(c.verify_result(good, 'f'*64)[1], report)
        for change in [dict(worker_sha256='a'*64), dict(gpu_requested=True),
                       dict(cuda_context_requested=True), dict(candidate_kernel_launches=1),
                       dict(candidate_kernel_launches=False)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.verify_result(reply({**report, **change}), 'f'*64)
        for bad in [{**good, 'report_sha256': '0'*64}, {**good, 'exit_code': '0'},
                    {**good, 'report_json': ' '*(c.MAX_RESULT+1)}, {**good, 'error': 'missing'}, reply([])]:
            with self.assertRaises(ValueError):
                c.verify_result(bad, 'f'*64)

    def test_offline_freeze_guard_parses_without_auth_or_service(self):
        head, calls, paths = 'a'*40, [], c.payload()
        controller_key = Path(c.__file__).relative_to(c.REPO).as_posix()
        def git_output(args, **kwargs):
            calls.append(args)
            if args == ['git', 'rev-parse', 'HEAD']:
                return head+'\n'
            if args[:2] == ['git', 'show']:
                name = args[2].split(':', 1)[1]
                return (Path(c.__file__) if name == controller_key else paths[name]).read_bytes()
            raise AssertionError(args)
        with tempfile.TemporaryDirectory() as folder:
            worker = Path(folder)/'worker.py'
            worker.write_text('"""Offline fixture; not the real inventory worker."""\n')
            paths[(c.HERE/'worker.py').relative_to(c.REPO).as_posix()] = worker
            with patch.object(c, 'payload', return_value=paths), \
                 patch.object(c.subprocess, 'check_output', side_effect=git_output), \
                 patch.object(c, 'configure_existing_auth', side_effect=AssertionError('auth touched')), \
                 patch.object(c, 'cli', side_effect=AssertionError('service touched')):
                observed, hashes = c.frozen_payload(head)
        self.assertEqual(len(observed), 3)
        self.assertEqual(len(hashes), 4)
        self.assertTrue(all(args[:2] in (['git', 'rev-parse'], ['git', 'show']) for args in calls))
        with self.assertRaises(ValueError):
            c.frozen_payload('HEAD')

    def test_stop_observation_rejects_stopping_and_live_tasks(self):
        self.assertTrue(c.stopped([dict(state='stopped', tasks='0')]))
        self.assertFalse(c.stopped([dict(state='stopping...', tasks='0')]))
        self.assertFalse(c.stopped([dict(state='stopped', tasks='1')]))

    def test_remote_child_is_bounded_and_hard_killed(self):
        tree = ast.parse((HERE/'controller.py').read_text())
        entry = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'entry')
        text = ast.unparse(entry)
        self.assertIn('timeout=35', text)
        self.assertIn('signal.SIGKILL', text)
        self.assertNotIn('AUTH', text)
        self.assertNotIn('modal', text.lower())
        self.assertEqual(c.MAX_RESULT, 262144)


if __name__ == '__main__':
    unittest.main(verbosity=2)
