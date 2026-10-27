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
spec = importlib.util.spec_from_file_location('vstage_runtime_controller', HERE/'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
worker_spec = importlib.util.spec_from_file_location('vstage_real_worker_contract', HERE/'worker.py')
w = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(w)


def actual_worker_samples(fa3=110., after=100., before=96.):
    samples = []
    for index in range(w.WARMUP_SAMPLES, w.WARMUP_SAMPLES+w.MEASURED_SAMPLES):
        order, version = w.order_and_version(index)
        samples.append(dict(index=index, order=order, input_version=version,
            raw_us={'fa3': fa3, 'v_after_wait': after, 'v_before_wait': before}))
    return samples


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
               {**reply, 'report_json': ' '*500001}, {**reply, 'exit_code': '0'}, {**reply, 'exit_code': False}]
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
        self.assertEqual(paths[(c.PROBE/'kernel.py').relative_to(c.REPO).as_posix()],
                         c.ATTENTION/'gluon-vstage-probe/kernel.py')
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

    def completion_fixture(self):
        worker_key = (c.HERE/'worker.py').relative_to(c.REPO).as_posix()
        sources = {'kernel.py': c.KERNEL_SHA, 'config.py': c.CONFIG_SHA,
                   'fusion-lookahead-probe/candidate.py': c.WORKING_SHA}
        report = dict(passed=True, status='complete', correctness_complete=True, timing_complete=True,
            source_sha256=sources, support_sha256=c.SUPPORT_HASHES,
            verified_hashes={**sources, **c.SUPPORT_HASHES, **c.LICENSE_HASHES},
            worker_sha256='f'*64, counts=c.EXPECTED_COUNTS, timings=[{'case': 'three-arm-target'}],
            benefit=w.benefit_gate(actual_worker_samples()))
        return {'exit_code': 0}, report, {worker_key: 'f'*64}

    def test_explicit_complete_success_and_negative_benefit(self):
        reply, report, hashes = self.completion_fixture()
        c.validate_success(reply, report, hashes)
        for times, expected in [((110.,100.,96.), (True,True,True)),
                                ((110.,100.,105.), (False,True,False)),
                                ((90.,100.,96.), (True,False,False)),
                                ((90.,100.,110.), (False,False,False))]:
            benefit = w.benefit_gate(actual_worker_samples(*times))
            self.assertEqual(tuple(benefit[key] for key in
                ['mechanism_passed','actual_fa3_passed','prospective_followup_supported']), expected)
            c.validate_success(reply, {**report, 'benefit': benefit}, hashes)
        for changes in [dict(status='partial'), dict(correctness_complete=False), dict(timing_complete=False),
                        dict(source_sha256={}), dict(support_sha256={}), dict(verified_hashes={}),
                        dict(worker_sha256='a'*64), dict(counts={}), dict(timings=[]),
                        dict(benefit={'admitted': True}), dict(passed='true'), dict(passed=1),
                        dict(official_check_admitted=True), dict(production_replacement_admitted=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_success(reply, {**report, **changes}, hashes)
        with self.assertRaises(ValueError):
            c.validate_success({'exit_code': 1}, report, hashes)
        c.validate_success({'exit_code': 124}, {'passed': False, 'status': 'partial'}, {})

    def test_expected_counts_match_actual_worker_fields_and_workload_arithmetic(self):
        cases, arms = len(w.CASES), len(w.NAMES)
        rounds = w.WARMUP_SAMPLES+w.MEASURED_SAMPLES
        expected = dict(original_oracle_calls=cases*arms*3+1+2, exact_return_A_checks=cases*arms,
            cross_arm_cache_byte_comparisons=cases*3*(arms-1)*2, concurrent_original_oracle_calls=2,
            independent_setup_oracle_calls=1, post_concurrent_isolation_checks=4, extra_candidate_replays=2,
            prototype_output_newkv_ABA_comparisons=cases*3, cross_arm_newkv_ABA_comparisons=cases*3*(arms-1),
            checked_warmup_measured_replays=rounds*arms, prototype_output_newkv_timing_comparisons=rounds,
            cross_arm_newkv_timing_comparisons=rounds*(arms-1))
        self.assertEqual(c.EXPECTED_COUNTS, expected)
        import ast
        tree = ast.parse((HERE/'worker.py').read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Subscript) and ast.unparse(target)=="report['counts']" for target in node.targets))
        self.assertEqual({keyword.arg for keyword in assignment.value.keywords}, set(expected))

    def test_performance_flag_requires_actual_raw_times_not_planned_rows(self):
        self.assertFalse(c.has_actual_timing({'timings': []}))
        self.assertFalse(c.has_actual_timing({'timings': [{'warmups': [{'raw_us': {}}], 'samples': []}]}))
        self.assertTrue(c.has_actual_timing({'timings': [{'warmups': [{'raw_us': {'fa3': 123.}}], 'samples': []}]}))

    def test_local_native_gate_binds_source_compile_reviews_and_all_pass_verdicts(self):
        with tempfile.TemporaryDirectory() as folder:
            probe = Path(folder); (probe/'native').mkdir()
            for name in ['kernel.py', 'config.py']:
                (probe/name).write_bytes((c.PROBE/name).read_bytes())
            compiled = dict(passed=True, compile_complete=True, kernel_launches=0,
                source_sha256={'kernel.py': c.KERNEL_SHA, 'config.py': c.CONFIG_SHA}, resource_gate={'status': 'pass'})
            raw = json.dumps(compiled).encode(); (probe/'native/result.json').write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            for name in ['SOURCE_REVIEW.md', 'native/NATIVE_REVIEW.md']:
                (probe/name).write_text('PASS explicit local review fixture\n')
            gate = dict(state='passed', experiment=c.EXPERIMENT, execution_admitted=True,
                kernel_sha256=c.KERNEL_SHA, config_sha256=c.CONFIG_SHA, raw_compile_sha256=sha,
                gate_verdicts={name: 'PASS' for name in ['mechanism','resource','lifetime','matched_work']},
                reviews={key: dict(path=name, sha256=hashlib.sha256((probe/name).read_bytes()).hexdigest(), verdict='PASS')
                    for key,name in [('source','SOURCE_REVIEW.md'),('native','native/NATIVE_REVIEW.md')]})
            path = probe/'gate.local.json'
            with patch.object(c, 'RAW_COMPILE_SHA', sha):
                path.write_text(json.dumps(gate)); self.assertEqual(c.validate_native_gate(path, probe), gate)
                for change in [dict(state='pending'), dict(execution_admitted=False), dict(raw_compile_sha256='0'*64),
                               dict(kernel_sha256='0'*64), dict(gate_verdicts={}), dict(reviews={}),
                               dict(reviews={**gate['reviews'], 'native': {'path':'../escape.md','sha256':'0'*64,'verdict':'PASS'}})]:
                    path.write_text(json.dumps({**gate, **change}))
                    with self.subTest(change=change), self.assertRaises(ValueError): c.validate_native_gate(path, probe)


if __name__ == '__main__':
    unittest.main(verbosity=2)
