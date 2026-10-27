import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('locality_probe_controller', HERE/'controller.py')
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
        self.assertEqual(len(paths), 8)
        self.assertTrue(all('/' not in key for key in paths))
        self.assertFalse(any('local-notes' in str(path) for path in paths.values()))
        self.assertEqual(paths['selected.py'].name, 'candidate.py')
        self.assertEqual(set(paths), {'worker.py','fixture.py','selected.py','reference.py','checks.py',
            'trace_helpers.py','LICENSE','THIRD_PARTY_NOTICES.md'})
        self.assertEqual(c.REPO, HERE.parents[2])
        self.assertEqual(c.ATTENTION, HERE.parent)
        for name, digest in c.EXPECTED.items():
            self.assertEqual(hashlib.sha256(paths[name].read_bytes()).hexdigest(), digest)

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

    def reply(self, report, trace=b'{"traceEvents":[]}'):
        raw = json.dumps(report).encode()
        reply = {'exit_code': 0, 'report_json': raw.decode(), 'report_sha256': hashlib.sha256(raw).hexdigest()}
        if trace is not None:
            packed = gzip.compress(trace, mtime=0)
            reply['trace'] = {'raw_bytes': len(trace), 'raw_sha256': hashlib.sha256(trace).hexdigest(),
                'gzip_bytes': len(packed), 'gzip_sha256': hashlib.sha256(packed).hexdigest(),
                'gzip_base64': base64.b64encode(packed).decode()}
        return reply

    def valid_report(self, keep=False):
        w = c.worker_protocol()
        hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in c.payload().items()}
        hashes['controller.py'] = hashlib.sha256((HERE/'controller.py').read_bytes()).hexdigest()
        trace = self.reply({})['trace']
        metadata = {k:v for k,v in trace.items() if k != 'gzip_base64'}
        fixtures = []
        for seed in w.SEEDS:
            fixtures.append(dict(seed=seed, plan={'full_capacity_permutation':True}, oracle=[dict(arm=a,passed=True,ownership=True,
                stable_addresses=True) for a in w.ARMS], warmup=[{}]*4,
                measured=[dict(index=i,order=w.measured_order(i),raw_us={
                    'random_pages':100.,'contiguous_pages':80. if keep else 99.}) for i in range(12)],
                post_call_full_cache_inverse_exact=True,cross_layout_output_exact=True,
                locality_contrast={'adjacent_attended_pairs':10,'moved_attended_history_pages':5,
                    'canonical_layout_contiguous_adjacent_pairs':10,'random_layout_contiguous_adjacent_pairs':0}))
        kernels=[{'name':'void FlashAttnFwdSm90<64, 128, 64>(Params)','launch':{'grid':[132,1,1],'block':[256,1,1],
            'registers per thread':112,'shared memory':127488,'graph id':1,'graph node id':2}}]
        identity=w.native_identity_comparison(kernels,kernels)
        counts=w.expected_counts(True)
        for name in ('fixtures','capture_warmup_function_calls_excluded','capture_function_calls_excluded'):
            del counts[name]
        protocol={'cases':list(w.CASES),'seeds':list(w.SEEDS),'arms':list(w.ARMS),
            'expected_counts':w.expected_counts(True),'warmup_rounds_per_fixture':4,
            'reverse_pairs_per_fixture':6,'same_graph_per_case':True,
            'private_default_capture_pools':True,'cold_l2_eviction_bytes':256*1024*1024,
            'raw_whole_call':True,'replay_cost_subtracted':False,'candidate_source_changed':False,
            'page_table_always_present':True,'provided_decode_metadata_preserved':True,'diagnostic_only':True}
        report={'worker_sha256':hashes['worker.py'],
            'verified_hashes':{k:v for k,v in hashes.items() if k not in ('worker.py','controller.py')},
            'status':'complete','passed':True,'selected_candidate_changed':False,'official_check':False,
            'formal_entry':False,'hardware_counter_sessions':0,'correctness_complete':True,
            'timing_complete':True,'default_graph_pool_handles_distinct':True,'protocol':protocol,
            'counts':counts,'cases':[{'case':w.CASES[0],'graph_captures':1,'fixtures':fixtures}],
            'decision':w.threshold_decision(fixtures,True),
            'activity':{'status':'complete','seed':w.SEEDS[-1],'sessions':1,
                'native_identity_fully_observed':True,
                'artifact':dict(name='activity-trace.json',gzip_name='activity-trace.json.gz',**metadata),
                'cross_layout_native_identity':[dict(case='full_verify_b32',**identity)],
                'summary':{'groups':{'PROFILE_SELECTED_FULL_VERIFY_B32_'+arm.upper():
                    {'kernels':copy.deepcopy(kernels)} for arm in w.ARMS}}}}
        reply=self.reply(report)
        return hashes,reply

    def test_completed_keep_and_completed_drop_are_both_valid(self):
        for keep in (True,False):
            hashes,reply=self.valid_report(keep)
            _,report,artifact=c.verified_result(reply)
            self.assertEqual(report['decision']['decision'],'KEEP' if keep else 'DROP')
            self.assertTrue(c.validate_report(report,hashes,artifact))
            with tempfile.TemporaryDirectory() as tmp:
                outcome=c.materialize_result(reply,Path(tmp),hashes)
                self.assertTrue(outcome['completion_verified'])
                self.assertEqual((Path(tmp)/'result.json').read_text(),reply['report_json'])
                self.assertTrue((Path(tmp)/'activity-trace.json.gz').is_file())

    def test_rejects_unknown_or_different_native_identity(self):
        hashes,reply=self.valid_report()
        _,report,artifact=c.verified_result(reply)
        altered=copy.deepcopy(report)
        altered['activity']['native_identity_fully_observed']=False
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):
            c.validate_report(altered,hashes,artifact)
        for key,value in [('grid',[1,1,1]),('graph node id',None)]:
            altered=copy.deepcopy(report)
            altered['activity']['summary']['groups']['PROFILE_SELECTED_FULL_VERIFY_B32_CONTIGUOUS_PAGES']['kernels'][0]['launch'][key]=value
            with self.assertRaisesRegex(ValueError,'differ or are UNKNOWN'):
                c.validate_report(altered,hashes,artifact)

    def test_rejects_wrong_sources_counts_case_or_semantics(self):
        hashes,reply=self.valid_report()
        _,report,artifact=c.verified_result(reply)
        mutations=[lambda r:r.update(worker_sha256='0'*64),
            lambda r:r['counts'].update(total_checked_graph_replays=69),
            lambda r:r['cases'][0]['case'].update(q_len=1),
            lambda r:r['protocol'].update(replay_cost_subtracted=True),
            lambda r:r.update(official_check=True),
            lambda r:r.update(correctness_complete=False),
            lambda r:r['cases'][0]['fixtures'][0].update(post_call_full_cache_inverse_exact=False),
            lambda r:r['cases'][0]['fixtures'][0]['plan'].update(full_capacity_permutation=False),
            lambda r:r['activity'].update(artifact={}),
            lambda r:r['decision'].update(decision='KEEP')]
        for mutation in mutations:
            altered=copy.deepcopy(report);mutation(altered)
            with self.assertRaises(ValueError):
                c.validate_report(altered,hashes,artifact)
        with self.assertRaises(ValueError):c.validate_report(report,hashes,None)

    def test_rejects_non_fa3_activity_groups(self):
        hashes,reply=self.valid_report()
        _,report,artifact=c.verified_result(reply)
        for group in report['activity']['summary']['groups'].values():
            group['kernels'][0]['name']='unrelated_kernel'
        with self.assertRaisesRegex(ValueError,'exactly one observed selected FA3'):
            c.validate_report(report,hashes,artifact)

    def test_local_protocol_import_requires_frozen_regular_worker(self):
        hashes,reply=self.valid_report()
        _,report,artifact=c.verified_result(reply)
        report['worker_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'differs from frozen'):
            c.validate_report(report,{**hashes,'worker.py':'0'*64},artifact)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'worker.py').symlink_to(HERE/'worker.py')
            with patch.object(c,'HERE',Path(tmp)):
                with self.assertRaisesRegex(ValueError,'regular nonsymlink'):
                    c.worker_protocol(hashes['worker.py'])

    def test_partial_failure_summary_is_saved_and_never_complete(self):
        hashes,reply=self.valid_report()
        report=json.loads(reply['report_json']);report.update(status='failed',passed=False)
        reply=self.reply(report);reply['exit_code']=124
        with tempfile.TemporaryDirectory() as tmp:
            result=c.materialize_result(reply,Path(tmp),hashes)
            self.assertFalse(result['completion_verified'])
            self.assertTrue((Path(tmp)/'result.json').exists())
            self.assertTrue((Path(tmp)/'activity-trace.json.gz').exists())

    def test_nonzero_exit_cannot_pass_even_with_complete_summary(self):
        hashes,reply=self.valid_report();reply['exit_code']=1
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(c.materialize_result(reply,Path(tmp),hashes)['completion_verified'])

    def test_transport_corruption_is_preserved_before_rejection(self):
        hashes,reply=self.valid_report()
        reply['trace']['gzip_sha256']='0'*64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Compressed trace digest'):
                c.materialize_result(reply,Path(tmp),hashes)
            self.assertTrue((Path(tmp)/'result.json').exists())
            self.assertTrue((Path(tmp)/'activity-trace.json.gz').exists())
            self.assertTrue((Path(tmp)/'trace-manifest.json').exists())
        hashes,reply=self.valid_report();reply['report_sha256']='0'*64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):c.materialize_result(reply,Path(tmp),hashes)
            self.assertTrue((Path(tmp)/'result.json').exists())
            self.assertTrue((Path(tmp)/'activity-trace.json.gz').exists())

    def test_transport_size_and_digest_bounds(self):
        for field in ('raw_sha256','gzip_sha256'):
            reply=self.reply({},b'{}');reply['trace'][field]='0'*64
            with self.assertRaises(ValueError):c.verified_result(reply)
        for bound in ('MAX_RESULT','MAX_TRACE','MAX_GZIP','MAX_TRACE_ENCODING'):
            with patch.object(c,bound,1):
                with self.assertRaises(ValueError):c.verified_result(self.reply({},b'{}'))
        self.assertEqual(c.verified_result({'exit_code':124}), (None,None,None))
        with self.assertRaises(ValueError):c.verified_result({'exit_code':True})

    def test_cleanup_polls_past_stopped_one_task_race(self):
        states=[[{'app_id':'ap-test','state':'stopped','tasks':1}],
            [{'app_id':'ap-test','state':'stopped','tasks':0}]]
        with patch.object(c,'cli',side_effect=states) as cli,patch.object(c.time,'sleep') as sleep:
            _,exact,observations=c.observe_shutdown('ap-test',attempts=3)
        self.assertTrue(c.stopped([exact]));self.assertEqual(len(observations),2)
        self.assertEqual(cli.call_count,2);sleep.assert_called_once_with(2)
        with patch.object(c,'cli',return_value=[]),patch.object(c.time,'sleep'):
            _,exact,observations=c.observe_shutdown('missing',attempts=2)
        self.assertIsNone(exact);self.assertEqual(len(observations),2)

    def test_remote_command_and_transport_use_fixed_flat_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);payload=tmp/'payload';payload.mkdir();output=tmp/'output';output.mkdir()
            hashes,reply=self.valid_report()
            (output/'summary.json').write_text(reply['report_json'])
            (output/'activity-trace.json').write_bytes(b'{"traceEvents":[]}')
            packed=base64.b64decode(reply['trace']['gzip_base64'])
            (output/'activity-trace.json.gz').write_bytes(packed)
            def remap(value):
                return {'/opt/attention-locality-b32l8':payload,
                    '/tmp/attention-locality-b32l8':output,
                    '/tmp/attention-locality-b32l8-worker.log':tmp/'worker.log'}.get(str(value),Path(value))
            with patch('pathlib.Path',side_effect=remap),patch.object(c.subprocess,'Popen') as popen:
                popen.return_value.wait.return_value=0
                result=c.make_remote_entry()()
            command=popen.call_args.args[0]
            self.assertEqual(command[2:],[ '--execute','--activity','--deadline','160',
                '--payload-root',str(payload),'--output',str(output/'summary.json')])
            self.assertEqual(popen.return_value.wait.call_args.kwargs['timeout'],175)
            self.assertEqual(result['trace']['gzip_sha256'],reply['trace']['gzip_sha256'])
            self.assertTrue(c.verified_result(result)[2]['raw'])


if __name__ == '__main__':
    unittest.main()
