import base64
import csv
import io
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
import inspect
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('fa3_b32_capture_executor_controller', HERE/'controller.py')
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
        self.assertEqual(len(paths), 14)
        self.assertTrue(all('/' not in key for key in paths))
        self.assertFalse(any('local-notes' in str(path) for path in paths.values()))
        self.assertEqual(paths['selected.py'].name, 'candidate.py')
        self.assertEqual(paths['cpu_identity.py'].parent.name, 'dispatch-identity-cpu')
        self.assertFalse(any('m64n8' in str(path).lower() for path in paths.values()))
        self.assertEqual(paths['cli_contract.py'].parent.name, 'eager-counter-repair')

    def test_upload_and_entry_roots_match(self):
        source = Path(c.__file__).read_text()
        entry = inspect.getsource(c.make_remote_entry)
        self.assertIn("Path('/opt/attention-fa3-b32-capture-executor')", entry)
        self.assertIn("'/opt/attention-fa3-b32-capture-executor/'+name", source)
        self.assertNotIn('/opt/attention-baseline-identity', source)
        self.assertNotIn('/opt/attention-eager-counter/', source)

    def test_remote_entry_has_no_closure_or_auth(self):
        fn = c.make_remote_entry()
        self.assertIsNone(fn.__closure__)
        self.assertFalse(any(name in fn.__code__.co_names for name in ('AUTH', 'cli', 'configure_existing_auth')))

    def test_guardrail_credit_floor_tracks_increment(self):
        self.billing['metered_cost'] = '2.6'
        self.assertEqual(c.validate_budget(self.receipt, self.billing, self.now), Decimal('27.4'))

    def test_guardrail_rejects_paid_stale_and_exhausted(self):
        for field, value in [('verified_net_spend_cap_usd', '1'), ('total_spend_usd', '1'),
                ('changes_made', True), ('verified_credit_balance_usd', '.70'),
                ('observed_utc', (self.now-timedelta(minutes=16)).isoformat())]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    c.validate_budget({**self.receipt, field: value}, self.billing, self.now)
        with self.assertRaises(ValueError):
            c.validate_budget(self.receipt, {**self.billing, 'billed_cost': '.01'}, self.now)

    def test_reserve_uses_live_whole_resource_rate(self):
        rates = {'gpu_hour_cost_h200': '4.548', 'cpu_hour_cost': '.0474', 'mem_gib_hour_cost': '.0081'}
        rate = c.resource_rate(rates)
        self.assertEqual(c.reserve(rate), Decimal('.70'))
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

    def reply(self, report, trace=None, artifacts=None):
        raw = json.dumps(report).encode()
        reply = {'exit_code': 0, 'report_json': raw.decode(), 'report_sha256': hashlib.sha256(raw).hexdigest()}
        if trace is not None:
            packed = gzip.compress(trace)
            reply['trace'] = {'raw_bytes': len(trace), 'raw_sha256': hashlib.sha256(trace).hexdigest(),
                'gzip_bytes': len(packed), 'gzip_sha256': hashlib.sha256(packed).hexdigest(),
                'gzip_base64': base64.b64encode(packed).decode()}
        if artifacts is not None:
            reply['artifacts'] = artifacts
        return reply

    def encoded(self, raw):
        packed = gzip.compress(raw, mtime=0)
        encoding = base64.b64encode(packed).decode()
        return {'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'gzip_bytes': len(packed), 'gzip_sha256': hashlib.sha256(packed).hexdigest(),
            'compression': 'gzip', 'stable_file_snapshot': True,
            'chunks_base64': [encoding[i:i+c.CHUNK_ENCODING] for i in range(0, len(encoding), c.CHUNK_ENCODING)]}

    def csv(self, first='11.47', kernel=None):
        expected = json.loads((HERE/'expected-native.json').read_text())
        kernel = expected['kernel_name'] if kernel is None else kernel
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(['ID', 'Kernel Name', 'Grid Size', 'Block Size', *c.METRICS])
        writer.writerow(['', '', '', '', '%', '%', '%', '%'])
        writer.writerow(['0', kernel, '(132, 1, 1)', '(256, 1, 1)', first, '36.77', '43.40', '12.49'])
        return out.getvalue().encode()

    def completion(self, imported=None):
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in c.payload().items()}
        hashes['controller.py'] = hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()
        expected = json.loads((HERE/'expected-native.json').read_text())
        values = {'baseline.ncu-rep': self.encoded(b'NCU-REPORT-BINARY'),
            'profile-original.csv': self.encoded(self.csv()),
            'profile-imported.csv': self.encoded(imported if imported is not None else self.csv())}
        reply = self.reply({}, b'{"traceEvents":[]}', values)
        _, _, trace, durable = c.verified_result(reply)
        metadata = {name: {'bytes': row['raw_bytes'], 'sha256': row['raw_sha256']}
            for name, row in values.items()}
        m = trace['manifest']
        absolute = '/tmp/attention-fa3-b32-capture-executor/baseline.ncu-rep'
        report = {'worker_sha256': hashes['worker.py'],
            'verified_hashes': {key: value for key, value in hashes.items()
                if key not in ('worker.py', 'controller.py')},
            'passed': True, 'status': 'complete', 'case': c.CASE,
            'selected_candidate_changed': False, 'official_check': False,
            'formal_entry': False, 'security_changes': False, 'performance_variant': False,
            'counter_sessions_attempted': 1, 'activity_sessions': 1,
            'trace_artifacts': {
                'activity-trace.json': {'bytes': m['raw_bytes'], 'sha256': m['raw_sha256']},
                'activity-trace.json.gz': {'bytes': m['gzip_bytes'], 'sha256': m['gzip_sha256']}},
            'artifact_manifest': metadata,
            'ncu_report_artifact': {'filename': 'baseline.ncu-rep', **metadata['baseline.ncu-rep']},
            'report_roundtrip': {'passed': True, 'import_exit_code': 0,
                'signature_equal': True, 'launch_equal': True, 'metric_values_units_equal': True,
                'report_unchanged': True, 'report_before_import': metadata['baseline.ncu-rep'],
                'report_after_import': metadata['baseline.ncu-rep']},
            'commands': [{'argv': ['ncu', '--export', absolute], 'state': 'complete', 'returncode': 0},
                {'argv': ['ncu', '--import', absolute], 'state': 'complete', 'returncode': 0}]}
        bindings = expected['bindings']
        report['native_capture_bindings'] = bindings
        report['cpu_file_identity'] = {'files': [
            {'path': '/opt/sglang/sgl_kernel/flash_attn.py', 'sha256': bindings['wrapper_sha256']}],
            'binary_files': [{'path': '/opt/sglang/sgl_kernel/flash_ops.abi3.so', 'sha256': bindings['binary_sha256']}]}
        report['fixture'] = {'tensors': {key[len('fixture.'):]: {'sha256': value}
            for key, value in bindings.items() if key.startswith('fixture.')}}
        parser = c.local_csv_parser(hashes)
        plan = c.local_capture_plan(hashes)
        state = lambda name: {'returncode': 0, 'stdout': durable[name]['raw'].decode(),
            'report_path': absolute, 'report': metadata['baseline.ncu-rep'],
            'bindings': bindings, 'row': parser(durable[name]['raw'].decode(), c.METRICS)}
        report['native_capture_identity'] = plan.capture_import_gate(
            state('profile-original.csv'), state('profile-imported.csv'), expected)
        report['graph_proxy_identity'] = {'raw_graph_ncu_symbol_equal': False,
            'graph_abi_equivalence': 'UNRESOLVED', 'scheduler_state': 'UNKNOWN'}
        return report, hashes, trace, durable

    def test_transport_verifies_raw_and_compressed_identity(self):
        reply = self.reply({'passed': True}, b'{"traceEvents":[]}')
        raw, report, artifact, durable = c.verified_result(reply)
        self.assertEqual(report, {'passed': True})
        self.assertEqual(artifact['raw'], b'{"traceEvents":[]}')
        self.assertEqual(durable, {})
        for field in ('raw_sha256', 'gzip_sha256'):
            modified = self.reply({}, b'{}')
            modified['trace'][field] = '0'*64
            with self.assertRaises(ValueError): c.verified_result(modified)

    def test_transport_retains_failure_summary_and_rejects_bound(self):
        reply = self.reply({'passed': False, 'status': 'blocked'}, artifacts={
            'baseline.ncu-rep': self.encoded(b'PARTIAL-REPORT')})
        reply['exit_code'] = 2
        self.assertEqual(c.verified_result(reply)[1]['status'], 'blocked')
        self.assertEqual(c.verified_result(reply)[3]['baseline.ncu-rep']['raw'], b'PARTIAL-REPORT')
        self.assertEqual(c.verified_result({'exit_code': 124, 'error': 'timeout'}), (None, None, None, {}))
        with patch.object(c, 'MAX_RESULT', 1):
            with self.assertRaises(ValueError): c.verified_result(reply)
        with patch.object(c, 'MAX_TRACE', 1):
            with self.assertRaises(ValueError): c.verified_result(self.reply({}, b'{}'))

    def test_lossless_report_chunks_and_corrupt_transport_rejection(self):
        raw = bytes(range(256))*10
        with patch.object(c, 'CHUNK_ENCODING', 128):
            row = self.encoded(raw)
            decoded = c.decode_artifacts({'baseline.ncu-rep': row})
            self.assertEqual(decoded['baseline.ncu-rep']['raw'], raw)
        for field in ('raw_sha256', 'gzip_sha256'):
            row = self.encoded(raw); row[field] = '0'*64
            with self.assertRaises(ValueError): c.decode_artifacts({'baseline.ncu-rep': row})
        with self.assertRaises(ValueError): c.decode_artifacts({'baseline.ncu-rep': self.encoded(b'')})
        with patch.dict(c.REQUIRED_ARTIFACTS, {'baseline.ncu-rep': 1}):
            with self.assertRaises(ValueError): c.decode_artifacts({'baseline.ncu-rep': self.encoded(b'ab')})
        with patch.object(c, 'MAX_ARTIFACT_ENCODING', 1):
            with self.assertRaises(ValueError): c.decode_artifacts({'baseline.ncu-rep': self.encoded(raw)})

    def test_completion_requires_lossless_report_and_successful_import(self):
        report, hashes, trace, durable = self.completion()
        self.assertTrue(c.validate_report(report, hashes, trace, durable))
        self.assertFalse(c.validate_report({**report, 'passed': False}, hashes, trace, {}))
        for missing in c.REQUIRED_ARTIFACTS:
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    c.validate_report(report, hashes, trace, {k: v for k, v in durable.items() if k != missing})
        for modification in (
                {'report_roundtrip': {**report['report_roundtrip'], 'import_exit_code': 1}},
                {'report_roundtrip': {**report['report_roundtrip'], 'report_after_import': {'bytes': 1, 'sha256': 'f'*64}}},
                {'artifact_manifest': {}}, {'ncu_report_artifact': {}},
                {'trace_artifacts': {}}, {'case': {**c.CASE, 'q_len': 1}},
                {'worker_sha256': 'd'*64}, {'counter_sessions_attempted': 2}, {'formal_entry': True}):
            with self.subTest(modification=modification):
                with self.assertRaises(ValueError): c.validate_report({**report, **modification}, hashes, trace, durable)
        with self.assertRaises(ValueError): c.validate_report(report, hashes, None, durable)
        failed = json.loads(json.dumps(report)); failed['commands'][1]['returncode'] = 1
        with self.assertRaises(ValueError): c.validate_report(failed, hashes, trace, durable)

    def test_local_reparse_rejects_false_csv_equivalence_claim(self):
        report, hashes, trace, durable = self.completion(imported=self.csv(first='12.47'))
        with self.assertRaisesRegex(ValueError, 'CSV metric values'):
            c.validate_report(report, hashes, trace, durable)
        report, hashes, trace, durable = self.completion(imported=self.csv(kernel='void other_kernel<int>'))
        with self.assertRaisesRegex(ValueError, 'CSV signatures'):
            c.validate_report(report, hashes, trace, durable)

    def test_native_bindings_are_recomputed_and_unknown_graph_names_do_not_block(self):
        report, hashes, trace, durable = self.completion()
        self.assertTrue(c.validate_report(report, hashes, trace, durable))
        changed = json.loads(json.dumps(report))
        changed['cpu_file_identity']['binary_files'][0]['sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'runtime evidence'):
            c.validate_report(changed, hashes, trace, durable)

    def test_shared_materializer_saves_exact_files_and_partial_raw_before_rejection(self):
        report, hashes, trace, durable = self.completion()
        reply = self.reply(report, trace['raw'], {name: self.encoded(row['raw']) for name, row in durable.items()})
        with tempfile.TemporaryDirectory() as td:
            outcome = c.materialize_result(reply, Path(td)/'pass', hashes)
            self.assertTrue(outcome['completion_verified'])
            self.assertEqual((Path(td)/'pass/artifacts/baseline.ncu-rep').read_bytes(), b'NCU-REPORT-BINARY')
            changed = json.loads(json.dumps(reply))
            modified_report = json.loads(changed['report_json'])
            modified_report['report_roundtrip']['import_exit_code'] = 1
            raw = json.dumps(modified_report).encode()
            changed['report_json'] = raw.decode()
            changed['report_sha256'] = hashlib.sha256(raw).hexdigest()
            with self.assertRaises(ValueError):
                c.materialize_result(changed, Path(td)/'fail', hashes)
            self.assertTrue((Path(td)/'fail/result.json').is_file())
            self.assertEqual((Path(td)/'fail/artifacts/baseline.ncu-rep').read_bytes(), b'NCU-REPORT-BINARY')


if __name__ == '__main__':
    unittest.main()
