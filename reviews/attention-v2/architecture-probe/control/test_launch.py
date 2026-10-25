"""Bounded transport, immutability and one-invocation controller guards; no Modal."""
import ast
import base64
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probe_launch', HERE/'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def transfer(files):
    raw = json.dumps({name: base64.b64encode(data).decode() for name, data in files.items()}, sort_keys=True).encode()
    packed = zlib.compress(raw, 9)
    return dict(chunks=1, packed_bytes=len(packed), raw_bytes=len(raw),
        sha256=hashlib.sha256(packed).hexdigest(),
        artifact_sha256={n: hashlib.sha256(v).hexdigest() for n, v in files.items()}), [packed]


class LaunchGuardTests(unittest.TestCase):
    def test_live_rates_reservation(self):
        rate = launch.resource_rate(dict(gpu_hour_cost_h200='4.54000', cpu_hour_cost='0.04730', mem_gib_hour_cost='0.00800'))
        self.assertEqual(rate, Decimal('4.6986')/3600)
        self.assertEqual(launch.reserve(rate=rate), Decimal(130)*rate*Decimal('1.1')+Decimal('.01'))
        self.assertGreater(launch.reserve(rate=rate), Decimal('.1966'))
        self.assertLess(launch.reserve(rate=rate), Decimal('.1967'))

    def test_remote_entry_has_no_configuration_closure(self):
        fn = launch.make_remote_entry()
        self.assertIsNone(fn.__closure__)
        self.assertEqual(fn.__code__.co_freevars, ())

    def test_single_invocation_and_watchdog_order(self):
        tree = ast.parse((HERE/'launch.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in {'remote', 'remote_gen', 'spawn', 'map'}]
        self.assertEqual([ast.unparse(n.func) for n in calls], ['fn.remote_gen'])
        text = (HERE/'launch.py').read_text()
        self.assertLess(text.index('timer.start()'), text.index('with app.run('))
        self.assertIn("with ledger.open('x')", text)
        self.assertIn('max_containers=1', text)
        self.assertIn('retries=None', text)
        self.assertIn('include_source=False', text)
        self.assertIn('block_network=True', text)
        self.assertIn('restrict_modal_access=True', text)

    def test_valid_artifacts_hash_checked(self):
        files = {'summary.json': b'{"status":"compiled"}', 'bf16.b16.fused.ptx': b'example-native-text'}
        manifest, chunks = transfer(files)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'artifacts'
            self.assertEqual(launch.unpack_artifacts(manifest, chunks, out), {'status': 'compiled'})
            self.assertEqual((out/'bf16.b16.fused.ptx').read_bytes(), files['bf16.b16.fused.ptx'])

    def test_reject_traversal_and_changed_transport(self):
        manifest, chunks = transfer({'summary.json': b'{}', '../escape': b'bad'})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'artifacts'
            with self.assertRaises(AssertionError):
                launch.unpack_artifacts(manifest, chunks, out)
            self.assertFalse(out.exists())
        manifest, chunks = transfer({'summary.json': b'{}'})
        manifest['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                launch.unpack_artifacts(manifest, chunks, Path(tmp)/'artifacts')

    def test_payload_source_hash_and_allowlist(self):
        manifest = json.loads((HERE.parent/'probe/manifest.json').read_text())
        self.assertEqual(manifest['source']['sha256'], launch.SOURCE_SHA256)
        self.assertEqual(set(manifest['payload_sha256']), {'candidate.py', 'worker.py'})
        self.assertIn("PROBE/n for n in ['candidate.py', 'worker.py']", (HERE/'launch.py').read_text())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LaunchGuardTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (HERE/'local-verification.json').write_text(json.dumps(dict(
        status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        gpu_launched=False, network_called=False,
        controller_sha256=hashlib.sha256((HERE/'launch.py').read_bytes()).hexdigest()), indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
