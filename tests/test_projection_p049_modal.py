"""P049-specific local launch boundaries; no external calls or CUDA."""
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import projection_p049_modal as p


class P049LauncherTests(unittest.TestCase):
    def test_exact_allowlist_and_sources(self):
        self.assertEqual(len(p.PAYLOAD),13)
        self.assertEqual(len(set(p.PAYLOAD)),13)
        self.assertIn('projection_p046_confirmation.py',p.PAYLOAD)
        self.assertIn('projection_p049_screen.py',p.PAYLOAD)
        for source,digest in ((p.SOURCE,p.SOURCE_SHA256),(p.PARENT,p.PARENT_SHA256)):
            self.assertEqual(hashlib.sha256((p.ROOT/source).read_bytes()).hexdigest(),digest)
        self.assertFalse(any('local-notes' in x or 'auth' in x or x.endswith('_modal.py') for x in p.PAYLOAD))

    def test_single_rpc_and_bounded_resources(self):
        tree=ast.parse(Path(p.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        remote=[n for n in calls if n.func.attr in ('remote','remote_gen','spawn')]
        self.assertEqual(len(remote),1);self.assertEqual(remote[0].func.attr,'remote_gen')
        settings={k.arg:ast.literal_eval(k.value) for n in calls if n.func.attr=='function' for k in n.keywords if k.arg!='image'}
        self.assertEqual((settings['gpu'],settings['max_containers'],settings['min_containers']),('H200',1,0))
        self.assertEqual((settings['timeout'],settings['startup_timeout']),(195,45))
        self.assertIsNone(settings['retries'])
        self.assertTrue(settings['block_network'] and settings['restrict_modal_access'])
        self.assertFalse(settings['include_source'])
        self.assertIsNone(p.make_remote_entry().__closure__)

    def test_specific_record_prevents_relaunch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'experiments').mkdir()
            (root/'experiments/projection-p049-screen-modal.json').write_text('{}')
            with patch.object(p,'ROOT',root),patch.object(p,'frozen_payload') as freeze, self.assertRaisesRegex(ValueError,'reconcile'):
                p.launch('0'*40,root)
            freeze.assert_not_called()

    def test_failed_credit_or_paid_guard(self):
        rates=dict(gpu_hour_cost_h200='4.5396',cpu_hour_cost='.04728',mem_gib_hour_cost='.00807')
        billing=dict(billed_cost='0',metered_cost='2.00',metered_cost_breakdown={'app':'2.10'})
        pre=dict(verified_credit_balance_usd='.45',total_usage_usd='2.00')
        with self.assertRaisesRegex(ValueError,'reserve'):p.budget_guard(pre,billing,rates)
        pre['verified_credit_balance_usd']='28'
        self.assertGreater(p.budget_guard(pre,billing,rates)[2],p.reserve())
        with self.assertRaisesRegex(ValueError,'zero'):p.budget_guard(pre,{**billing,'billed_cost':'.01'},rates)

    def test_self_test_has_no_process_or_sdk_action(self):
        with patch.object(p.subprocess,'run',side_effect=AssertionError('external')),patch.object(p.subprocess,'check_output',side_effect=AssertionError('external')):
            self.assertEqual(p.main(['self-test','--control-root','/unused']),0)

    def test_child_has_exact_worker_and_timeouts(self):
        source=Path(p.__file__).read_text()
        self.assertIn('str(root / "projection_p049_screen.py")',source)
        self.assertIn('"--deadline-seconds", "120"',source)
        self.assertIn('child.communicate(timeout=180)',source)
        self.assertIn('threading.Timer(240, watchdog)',source)
        self.assertIn('reserve(rate)',source)
        self.assertNotIn('modal token',source)


if __name__=='__main__':unittest.main()
