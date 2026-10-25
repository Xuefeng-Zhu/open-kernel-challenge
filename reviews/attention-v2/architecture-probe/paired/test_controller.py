"""Local safety guards only: no Modal import, network or GPU invocation."""
import ast
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('architecture_paired_controller', HERE/'controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def test_payload_allowlist_and_no_credential(self):
        paths = controller.payload()
        self.assertEqual(set(paths), {'worker.py', 'candidatecustomhelper.py', 'candidate.py',
            'modal_worker.py', 'official/reference.py', 'official/checks.py',
            'legal/LICENSE', 'legal/THIRD_PARTY_NOTICES.md'})
        self.assertFalse(any('local-notes' in str(path) or 'auth' in str(path) for path in paths.values()))
        self.assertTrue(all(path.is_relative_to(controller.REPO) for path in paths.values()))
        self.assertEqual(hashlib.sha256(paths['candidate.py'].read_bytes()).hexdigest(), controller.SOURCE_SHA256)

    def test_exact_resource_rate_and_reserve(self):
        rate = controller.resource_rate(dict(gpu_hour_cost_h200='4.54', cpu_hour_cost='.0473', mem_gib_hour_cost='.008'))
        self.assertEqual(rate, (Decimal('4.54')+2*Decimal('.0473')+8*Decimal('.008'))/3600)
        self.assertEqual(controller.reserve(rate=rate), Decimal(205)*rate*Decimal('1.1')+Decimal('.01'))
        self.assertLess(controller.reserve(rate=rate), Decimal('.305'))

    def test_single_remote_call_and_bounded_resources(self):
        source = (HERE/'controller.py').read_text()
        tree = ast.parse(source)
        remote = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr in {'remote', 'remote_gen'}]
        self.assertEqual(len(remote), 1)
        self.assertEqual(ast.unparse(remote[0].func), 'fn.remote')
        for fragment in ['timeout=135, startup_timeout=20, retries=0',
                         'min_containers=0, max_containers=1, buffer_containers=0',
                         'block_network=True, restrict_modal_access=True',
                         'threading.Timer(170, watchdog)', "with ledger.open('x')", "with (HERE/'result.json').open('xb')"]:
            self.assertIn(fragment, source)

    def test_remote_entry_has_no_local_state_closure(self):
        entry = controller.make_remote_entry()
        self.assertIsNone(entry.__closure__)
        tree = ast.parse((HERE/'controller.py').read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_remote_entry')
        source = ast.unparse(node)
        self.assertNotIn('MODAL_CONFIG_PATH', source)
        self.assertNotIn('local-notes', source)
        self.assertIn('timeout=120', source)
        self.assertIn("'--deadline', '110'", source)
        self.assertIn('200000', source.replace('_', ''))

    def test_report_integrity_and_size(self):
        raw = b'{"status":"failed","passed":false}'
        reply = dict(exit_code=1, report_json=raw.decode(), report_sha256=hashlib.sha256(raw).hexdigest())
        actual, report = controller.verified_result(reply)
        self.assertEqual(actual, raw)
        self.assertFalse(report['passed'])
        with self.assertRaises(AssertionError):
            controller.verified_result({**reply, 'report_sha256': '0'*64})
        with self.assertRaises(AssertionError):
            controller.verified_result(dict(error='absent', exit_code=1))
        with self.assertRaises(AssertionError):
            controller.verified_result({**reply, 'report_json': 'x'*(controller.MAX_RESULT+1)})

    def test_original_ledger_and_slot_are_read_only(self):
        tree = ast.parse((HERE/'controller.py').read_text())
        write_calls = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr in {'write_text', 'write_bytes', 'open', 'replace'}]
        self.assertFalse(any('ORIGINAL' in value or 'gpu-slot' in value or 'modal-budget' in value for value in write_calls))
        source = (HERE/'controller.py').read_text()
        self.assertIn("CONTROL/'shutdown-observation.json'", source)
        self.assertIn('Decimal(billing[\'billed_cost\']) == 0', source)
        self.assertIn('artificial_stage_ceiling_removed=True', source)
        self.assertNotIn('conservative_stage_spent_usd', source)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControllerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        gpu_launched=False, network_called=False,
        controller_sha256=hashlib.sha256((HERE/'controller.py').read_bytes()).hexdigest())
    (HERE/'controller-local-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
