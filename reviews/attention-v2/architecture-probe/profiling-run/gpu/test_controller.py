"""Host-only transport/admission guards. No Modal, network or CUDA calls."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('activity_controller', HERE/'controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def package(files):
    raw = json.dumps({name: base64.b64encode(value).decode() for name,value in files.items()},sort_keys=True).encode()
    packed = zlib.compress(raw)
    chunks = [packed[i:i+controller.CHUNK] for i in range(0,len(packed),controller.CHUNK)]
    return dict(chunks=len(chunks), packed_bytes=len(packed), raw_bytes=len(raw),
        artifact_bytes=sum(map(len,files.values())), sha256=hashlib.sha256(packed).hexdigest(),
        artifact_sha256={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}),chunks


class ControllerTests(unittest.TestCase):
    def test_payload_exact_and_credentials_absent(self):
        paths = controller.payload()
        self.assertEqual(set(paths), {'worker.py','candidate.py','candidatecustomhelper.py','modal_worker.py',
            'counters.py','counter_target.py','inventory.json',
            'official/reference.py','official/checks.py','legal/LICENSE','legal/THIRD_PARTY_NOTICES.md'})
        self.assertTrue(all(path.is_relative_to(controller.REPO) for path in paths.values()))
        self.assertFalse(any('local-notes' in str(path) or 'auth' in str(path) for path in paths.values()))
        self.assertEqual(hashlib.sha256(paths['candidate.py'].read_bytes()).hexdigest(),controller.SOURCE_SHA256)

    def test_transport_roundtrip(self):
        files={'summary.json':b'{"hardware_counter_sessions":0,"status":"complete"}',
               'activity-trace.json':b'{"traceEvents":[]}'}
        manifest,chunks=package(files)
        with tempfile.TemporaryDirectory() as name:
            dest=Path(name)/'artifacts'
            result=controller.unpack_artifacts(manifest,chunks,dest)
            self.assertEqual(result['status'],'complete')
            self.assertEqual((dest/'activity-trace.json').read_bytes(),files['activity-trace.json'])

    def test_transport_rejects_unreviewed_metrics_and_bad_hash(self):
        files={'summary.json':b'{"hardware_counter_sessions":0}', 'metrics.json':b'{}'}
        manifest,chunks=package(files)
        with tempfile.TemporaryDirectory() as name:
            dest=Path(name)/'artifacts'
            with self.assertRaises(AssertionError):controller.unpack_artifacts(manifest,chunks,dest)
            self.assertFalse(dest.exists())
            files.pop('metrics.json');manifest,chunks=package(files)
            manifest['sha256']='0'*64
            with self.assertRaises(AssertionError):controller.unpack_artifacts(manifest,chunks,dest)

    def test_transport_limits_reviewed_counter_sessions(self):
        manifest,chunks=package({'summary.json':b'{"hardware_counter_sessions":3}'})
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(AssertionError):
                controller.unpack_artifacts(manifest,chunks,Path(name)/'artifacts')
        manifest,chunks=package({'summary.json':b'{"hardware_counter_sessions":2}',
                                'fa3.ncu-rep':b'bounded-report'})
        with tempfile.TemporaryDirectory() as name:
            result=controller.unpack_artifacts(manifest,chunks,Path(name)/'artifacts')
            self.assertEqual(result['hardware_counter_sessions'],2)
        manifest,chunks=package({'summary.json':json.dumps(dict(
            hardware_counter_sessions=None,hardware_counter_sessions_upper_bound=2,
            counter_branch=dict(status='unavailable_or_partial'),activity_passed=True)).encode()})
        with tempfile.TemporaryDirectory() as name:
            result=controller.unpack_artifacts(manifest,chunks,Path(name)/'artifacts')
            self.assertIsNone(result['hardware_counter_sessions'])

    def test_inventory_must_be_cpu_and_terminal(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'inventory-run.local.json').write_text(json.dumps(dict(result_retrieved=True,exit_code=0,app_id='test-app')))
            (root/'result.json').write_text(json.dumps(dict(status='complete',gpu_requested=False,
                cuda_context_requested=False,candidate_kernel_launches=0)))
            receipt=dict(apps=[dict(app_id='test-app',state='stopped',tasks='0')])
            (root/'reconciliation.json').write_text(json.dumps(receipt))
            with patch.object(controller,'INVENTORY',root):
                self.assertEqual(controller.inventory_terminal_evidence()[0],'test-app')
                receipt['apps'][0]['state']='stopping...'
                (root/'reconciliation.json').write_text(json.dumps(receipt))
                with self.assertRaises(AssertionError):controller.inventory_terminal_evidence()

    def test_one_remote_call_and_fixed_bounds(self):
        source=(HERE/'controller.py').read_text();tree=ast.parse(source)
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and node.func.attr in {'remote','remote_gen'}]
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].func),'fn.remote_gen')
        self.assertIsNone(controller.make_remote_entry().__closure__)
        for token in ['timeout=190, startup_timeout=20, retries=None',
            'min_containers=0, max_containers=1, buffer_containers=0',
            'threading.Timer(210,watchdog)', 'timeout=175', "'--deadline', '165'",
            'block_network=True, restrict_modal_access=True',"with ledger.open('x')"]:
            self.assertIn(token,source)
        self.assertEqual(controller.MAX_RAW,8*1024*1024)
        self.assertEqual(controller.MAX_PACKED,4*1024*1024)
        self.assertEqual(controller.CHUNK,16*1024)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControllerTests))
    (HERE/'controller-local-verification.json').write_text(json.dumps(dict(
        status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        gpu_launched=False,network_called=False,
        controller_sha256=hashlib.sha256((HERE/'controller.py').read_bytes()).hexdigest()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
