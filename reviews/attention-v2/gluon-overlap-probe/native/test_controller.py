"""Offline guard/transport tests. No Modal SDK import, auth read or GPU launch."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gluon_overlap_compile_controller', HERE/'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def bundle_fixture(files, raw=None, compressed=None):
    if raw is None:
        raw = json.dumps({'files': {name: base64.b64encode(value).decode() for name, value in files.items()}},
                         separators=(',', ':')).encode()
    compressed = gzip.compress(raw, mtime=0) if compressed is None else compressed
    return dict(encoding='gzip+base64', data=base64.b64encode(compressed).decode(),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        uncompressed_sha256=hashlib.sha256(raw).hexdigest(), uncompressed_size=len(raw),
        files={name: dict(size=len(value), sha256=hashlib.sha256(value).hexdigest()) for name, value in files.items()})


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
                        dict(expires_utc=(self.current+timedelta(seconds=c.WATCHDOG_SECONDS-10)).isoformat())]:
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
        self.assertEqual(c.reserve(rate), Decimal('.40'))
        self.assertGreater(c.reserve(rate), Decimal(140)*rate*2+Decimal('.02'))
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
               {**reply, 'report_json': ' '*(c.MAX_RESULT+1)}, {**reply, 'exit_code': '0'}]
        for value in bad:
            with self.assertRaises(ValueError):
                c.verified_result(value)
        array = '[]'
        with self.assertRaises(ValueError):
            c.verified_result(dict(exit_code=0, report_json=array,
                                   report_sha256=hashlib.sha256(array.encode()).hexdigest()))

    def test_payload_and_remote_entry_do_not_capture_credentials_or_config(self):
        paths = c.payload()
        self.assertEqual(len(paths), 6)
        self.assertTrue(all(p.is_relative_to(c.REPO) for p in paths.values()))
        self.assertFalse(any('local-notes' in key or '.local.' in key for key in paths))
        self.assertEqual(paths[(c.PROBE/'kernel.py').relative_to(c.REPO).as_posix()],
                         c.PROBE/'kernel.py')
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
        self.assertEqual(hashes[(c.PROBE/'kernel.py').relative_to(c.REPO).as_posix()], c.KERNEL_SHA)
        self.assertTrue(all(command[:2] in (['git', 'rev-parse'], ['git', 'show']) for command in calls))
        with self.assertRaises(ValueError):
            c.frozen_payload('HEAD')

    def test_claimed_success_requires_exact_frozen_compile_contract(self):
        worker_key = (c.HERE/'worker.py').relative_to(c.REPO).as_posix()
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (c.PROBE/'kernel.py', c.PROBE/'config.py', c.HERE/'resources.py')}
        report = dict(passed=True, status='compiled', compile_complete=True,
            source_sha256=sources, verified_hashes={**sources, **c.LICENSE_HASHES},
            worker_sha256='f'*64, protocol={'compile_only': True}, kernel_launches=0,
            correctness_tested=False, performance_tested=False, execution_admitted=False,
            kernels=[dict(arm=arm, unit=unit, kernel_launched=False) for arm in ('serial', 'overlap') for unit in ('core',)])
        reply, hashes = {'exit_code': 0}, {worker_key: 'f'*64, **{path.relative_to(c.REPO).as_posix(): sources[path.name] for path in (c.PROBE/'kernel.py', c.PROBE/'config.py', c.HERE/'resources.py')}}
        artifacts = {name: b'x' for name in c.ARTIFACT_NAMES}
        c.validate_success(reply, report, hashes, artifacts)
        for changes in [dict(status='partial'), dict(compile_complete=False),
                        dict(source_sha256={}), dict(verified_hashes={}), dict(worker_sha256='a'*64),
                        dict(protocol={'compile_only': False}), dict(kernel_launches=1),
                        dict(kernel_launches=False), dict(kernels=[]), dict(correctness_complete=True),
                        dict(timings=[{'invented': True}]), dict(official_check_admitted=True),
                        dict(correctness_tested=True), dict(performance_tested=True), dict(execution_admitted=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_success(reply, {**report, **changes}, hashes, artifacts)
        with self.assertRaises(ValueError):
            c.validate_success({'exit_code': 1}, report, hashes, artifacts)
        with self.assertRaises(ValueError):
            c.validate_success(reply, report, hashes, {})
        c.validate_success({'exit_code': 124}, {'passed': False, 'status': 'partial'}, {}, {})

    def test_artifact_roundtrip_and_exclusive_flat_persistence(self):
        files = {'serial.core.ttgir': b'module {}\n',
                 'overlap.core.ptx': b'// native PTX\n',
                 'serial.core.cubin': b'\x00\xff\x01'}
        bundle = bundle_fixture(files)
        self.assertEqual(c.decode_bundle(bundle), files)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)/'artifacts'
            c.persist_artifacts(files, bundle, destination)
            for name, value in files.items():
                self.assertEqual((destination/name).read_bytes(), value)
            self.assertEqual(json.loads((destination/'manifest.json').read_text())['files'], bundle['files'])
            with self.assertRaises(ValueError):
                c.persist_artifacts(files, bundle, destination)
            with self.assertRaises(ValueError):
                c.persist_artifacts({'../escape': b'bad'}, bundle, Path(folder)/'other')
            self.assertFalse((Path(folder)/'escape').exists())

    def test_artifact_hash_size_base64_and_contents_tampering_rejected(self):
        files = {'overlap.core.ptx': b'PTX'}
        good = bundle_fixture(files)
        mutations = [dict(encoding='tar'), dict(data='??'), dict(compressed_sha256='0'*64),
                     dict(compressed_size=False), dict(compressed_size=1),
                     dict(uncompressed_sha256='0'*64), dict(uncompressed_size=True),
                     dict(uncompressed_size=good['uncompressed_size']+1), dict(files={}),
                     dict(files={'overlap.core.ptx': dict(size=3, sha256='0'*64)}),
                     dict(files={'overlap.core.ptx': dict(size=False, sha256=hashlib.sha256(b'PTX').hexdigest())})]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.decode_bundle({**good, **change})
        invalid_data = b'{"files":{"overlap.core.ptx":"??"}}'
        with self.assertRaises(ValueError):
            c.decode_bundle(bundle_fixture(files, raw=invalid_data))
        with self.assertRaises(ValueError):
            c.decode_bundle(bundle_fixture(files, raw=b'{"files":[]}'))
        with patch.object(c, 'MAX_ARTIFACT_FILE', 2), self.assertRaises(ValueError):
            c.decode_bundle(good)

    def test_artifact_paths_duplicate_keys_and_file_count_rejected(self):
        for name in ['../overlap.core.ptx', '/overlap.core.ptx',
                     'nested/overlap.core.ptx', 'overlap.core.exe',
                     'candidate\\producer.ptx', 'overlap.core.ptx\x00']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.decode_bundle(bundle_fixture({name: b'x'}))
        with self.assertRaises(ValueError):
            c.decode_bundle(bundle_fixture({}, raw=b'{"files":{},"files":{}}'))
        with self.assertRaises(ValueError):
            c.decode_bundle(bundle_fixture({'overlap.core.ptx': b'x'},
                raw=b'{"files":{"overlap.core.ptx":"eA==","overlap.core.ptx":"eA=="}}'))
        names = {f'fake{i}.ptx': b'x' for i in range(c.MAX_ARTIFACT_FILES+1)}
        with patch.object(c, 'ARTIFACT_NAMES', set(names)), self.assertRaises(ValueError):
            c.decode_bundle(bundle_fixture(names))

    def test_artifact_gzip_bomb_truncation_and_concatenation_rejected(self):
        raw = b'{"files":{}}'
        compressed = gzip.compress(raw, mtime=0)
        for wrong in [compressed[:-2], compressed+compressed, compressed+b'trailer']:
            with self.subTest(size=len(wrong)), self.assertRaises(ValueError):
                c.decode_bundle(bundle_fixture({}, raw=raw, compressed=wrong))
        bomb = bundle_fixture({}, raw=b' '*2048)
        bomb['uncompressed_size'] = 128
        with patch.object(c, 'MAX_RAW_BUNDLE', 128), self.assertRaises(ValueError):
            c.decode_bundle(bomb)


if __name__ == '__main__':
    unittest.main(verbosity=2)
