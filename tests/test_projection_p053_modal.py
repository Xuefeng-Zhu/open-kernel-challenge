"""P053 launcher fixture/static tests only; no SDK, CUDA or external calls."""
import ast
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib

import projection_p053_modal as p


def artifact_fixture():
    report = dict(mode="timing", complete=True, oracle_checks=580,
        sources_sha256={n: p.FIXED.get(n, '1'*64) for n in p.SOURCE_NAMES},
        inherited_p052=dict(sha256=p.FIXED[p.RECEIPT]), cases=[], assessment=dict(screen_passed=False))
    for m in p.CASES:
        report['cases'].append(dict(m=m, passed=True, correctness=[dict(passed=True) for _ in range(50)],
            series={name: dict(samples=[dict(correctness=dict(parent=True,candidate=True)) for _ in range(24)],
                               warmup=[dict(correctness=dict(parent=True,candidate=True)) for _ in range(6)]) for name in p.SERIES}))
    content = {'timing.json':json.dumps(report).encode()}
    activity = dict(mode='activity', complete=True, oracle_checks=18,
        sources_sha256=report['sources_sha256'], inherited_p052=report['inherited_p052'],
        timing_receipt=dict(sha256=hashlib.sha256(content['timing.json']).hexdigest()), profiles=[])
    for m in p.CASES:
        for runner in p.RUNNERS:
            for repeat in range(3):
                name = f'activity.pooler_m{m}.{runner}.repeat{repeat}.trace.json'
                raw = json.dumps(dict(traceEvents=[], fixture=name)).encode()
                content[name]=raw
                activity['profiles'].append(dict(m=m,runner=runner,repeat=repeat,passed=True,
                    trace_file=name,trace_sha256=hashlib.sha256(raw).hexdigest(),kernel_intervals=[]))
    content['activity.json']=json.dumps(activity).encode()
    content['summary.json'] = json.dumps(dict(complete=True,
        stages=[dict(mode='timing',exit_code=0),dict(mode='activity',exit_code=0)])).encode()
    content['timing.log']=b'fixture timing'
    content['activity.log']=b'fixture activity'
    return content


def encode(content):
    raw = json.dumps({n: base64.b64encode(v).decode() for n, v in content.items()}, sort_keys=True).encode()
    packed = zlib.compress(raw)
    chunks = [packed[i:i+p.CHUNK] for i in range(0, len(packed), p.CHUNK)]
    return dict(chunks=len(chunks), packed_bytes=len(packed), encoded_bytes=len(raw),
        artifact_bytes=sum(map(len, content.values())), sha256=hashlib.sha256(packed).hexdigest(),
        artifact_sha256={n: hashlib.sha256(v).hexdigest() for n, v in content.items()}), chunks


class P053LauncherTests(unittest.TestCase):
    def test_allowlist_exact_no_credentials_or_controller(self):
        self.assertEqual(len(p.PAYLOAD), 16)
        self.assertEqual(len(set(p.PAYLOAD)), 16)
        self.assertIn('official/reference.py', p.PAYLOAD)
        self.assertIn(p.PARENT, p.PAYLOAD)
        self.assertIn(p.CORRECTNESS_WORKER, p.PAYLOAD)
        self.assertFalse(any('local-notes' in n or 'auth' in n or n.endswith('_modal.py') for n in p.PAYLOAD))
        self.assertEqual(len(p.ARTIFACTS), 23)
        self.assertEqual(sum(n.endswith('.trace.json') for n in p.ARTIFACTS), 18)
        self.assertFalse(any('/' in n or '..' in n for n in p.ARTIFACTS))

    def test_one_invocation_resources_and_timeout_bounds(self):
        source = Path(p.__file__).read_text(); tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual([n.func.attr for n in calls if n.func.attr in ('remote', 'remote_gen', 'spawn')], ['remote_gen'])
        settings = {k.arg: ast.literal_eval(k.value) for n in calls if n.func.attr == 'function'
                    for k in n.keywords if k.arg != 'image'}
        self.assertEqual((settings['gpu'], settings['cpu'], settings['memory']), ('H200', (2,2), (8192,8192)))
        self.assertEqual((settings['min_containers'], settings['max_containers'], settings['buffer_containers']), (0,1,0))
        self.assertEqual((settings['timeout'], settings['startup_timeout'], settings['scaledown_window']), (230,45,2))
        self.assertIsNone(settings['retries'])
        self.assertTrue(settings['block_network'] and settings['restrict_modal_access'])
        self.assertFalse(settings['include_source'])
        self.assertIsNone(p.make_remote_entry().__closure__)
        for token in ('child.communicate(timeout=105)', 'threading.Timer(275, watchdog)', 'start_new_session=True',
                      'os.killpg(child.pid, signal.SIGKILL)', 'app.run(detach=False'):
            self.assertIn(token, source)
        self.assertNotIn('"--compile-only"', source)

    def test_timing_gate_requires_sources_complete_coverage_and_correctness(self):
        good=json.loads(artifact_fixture()['timing.json'])
        self.assertTrue(p.validate_timing(good))
        mutations=(lambda r:r.update(mode='activity'),lambda r:r.update(complete=False),
            lambda r:r.update(oracle_checks=579),lambda r:r.update(failure={'type':'fixture'}),
            lambda r:r['sources_sha256'].update({p.CORRECTNESS_WORKER:'0'*64}),
            lambda r:r['inherited_p052'].update(sha256='0'*64),lambda r:r['cases'].pop(),
            lambda r:r['cases'][0].update(passed=False),lambda r:r['cases'][0]['series'].pop('partial'),
            lambda r:r['cases'][0]['series']['whole']['samples'].pop(),
            lambda r:r['cases'][1]['series']['selected_partial']['warmup'].pop(),
            lambda r:r['cases'][0]['correctness'].pop(),
            lambda r:r['cases'][0]['correctness'][0].update(passed=False),
            lambda r:r['cases'][0]['series']['whole']['samples'][0]['correctness'].update(parent=False),
            lambda r:r['cases'][1]['series']['partial']['warmup'][0]['correctness'].update(candidate=False))
        for mutate in mutations:
            report=json.loads(json.dumps(good));mutate(report)
            with self.assertRaises(ValueError):p.validate_timing(report)

    def test_committed_payload_exact_and_fixed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); head='a'*40
            blobs = {n: ('fixture '+n).encode() for n in (*p.PAYLOAD, 'projection_p053_modal.py', 'projection_p044_modal.py')}
            for name, raw in blobs.items():
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            def git(cmd, **kwargs):
                return head+'\n' if cmd[1] == 'rev-parse' else blobs[cmd[2].split(':',1)[1]]
            fixed = {n: hashlib.sha256(blobs[n]).hexdigest() for n in p.FIXED}
            with patch.object(p.subprocess, 'check_output', side_effect=git):
                with self.assertRaisesRegex(ValueError, 'Frozen'): p.frozen_payload(head, root)
                with patch.object(p, 'FIXED', fixed):
                    paths, hashes = p.frozen_payload(head, root)
                    self.assertEqual(set(paths), set(p.PAYLOAD))
                    self.assertEqual(hashes[p.WORKER], hashlib.sha256(blobs[p.WORKER]).hexdigest())
                    (root/p.WORKER).write_text('changed')
                    with self.assertRaisesRegex(ValueError, 'Uncommitted'): p.frozen_payload(head, root)

    def test_wrong_head_and_symlink_refused(self):
        with patch.object(p.subprocess, 'check_output', return_value='b'*40+'\n'):
            with self.assertRaisesRegex(ValueError, 'HEAD differs'): p.frozen_payload('a'*40)
        with patch.object(p.subprocess, 'check_output') as call:
            with self.assertRaisesRegex(ValueError, 'reviewed commit'): p.frozen_payload('HEAD')
            call.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'target').write_text('fixture'); (root/p.WORKER).symlink_to(root/'target')
            with patch.object(p.subprocess, 'check_output', return_value='a'*40+'\n'):
                with self.assertRaisesRegex(ValueError, 'symlink'): p.frozen_payload('a'*40, root)

    def test_existing_record_no_auth_or_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'experiments').mkdir()
            (root/'experiments/projection-p053-phase-modal.json').write_text('{}')
            with patch.object(p,'ROOT',root), patch.object(p,'configure_existing_auth') as auth, patch.object(p,'frozen_payload') as freeze:
                with self.assertRaisesRegex(ValueError,'reconcile'): p.launch('a'*40,root)
                auth.assert_not_called(); freeze.assert_not_called()
        self.assertIn('with record_path.open("x")', Path(p.__file__).read_text())

    def test_budget_paid_credits_lag_and_rate_floor(self):
        self.assertEqual(p.reserve(), Decimal(305)*p.RATE_FLOOR*Decimal('1.1')+Decimal('.02'))
        self.assertLess(p.reserve(), Decimal('.459'))
        rates=dict(gpu_hour_cost_h200='4.5396',cpu_hour_cost='.04728',mem_gib_hour_cost='.00807')
        billing=dict(billed_cost='0',metered_cost='2',metered_cost_breakdown={'app':'2.10'})
        pre=dict(verified_credit_balance_usd='.55',total_usage_usd='2')
        with self.assertRaisesRegex(ValueError,'reserve'): p.budget_guard(pre,billing,rates)
        pre['verified_credit_balance_usd']='28'
        rate,reserve,floor=p.budget_guard(pre,billing,rates)
        self.assertEqual(floor,Decimal('27.90')); self.assertGreaterEqual(rate,p.RATE_FLOOR)
        with self.assertRaisesRegex(ValueError,'zero'): p.budget_guard(pre,dict(billing,billed_cost='.01'),rates)

    def test_fresh_paid_cap_and_explicit_slot_required(self):
        stamp=datetime.now(timezone.utc)
        good=dict(observed_utc=stamp.isoformat(),workspace='xuefeng-zhu',gpu_slot_granted=True,
                  verified_net_spend_cap_usd='0',total_spend_usd='0',verified_credit_balance_usd='28',total_usage_usd='2')
        p.validate_preflight(good,stamp)
        for change in (dict(observed_utc=(stamp-timedelta(minutes=16)).isoformat()),dict(gpu_slot_granted=False),dict(verified_net_spend_cap_usd='1')):
            with self.assertRaises(ValueError): p.validate_preflight(dict(good,**change),stamp)

    def test_complete_transport_traces_and_receipt_hash_bound(self):
        content=artifact_fixture()
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'out'
            self.assertTrue(p.unpack_artifacts(*encode(content),dest)['complete'])
            for name,raw in content.items(): self.assertEqual((dest/name).read_bytes(),raw)
            with self.assertRaisesRegex(ValueError,'destination exists'): p.unpack_artifacts(*encode(content),dest)

    def test_transport_rejects_unsafe_corrupt_truncated_or_wrong_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'out'
            content=artifact_fixture(); content['../bad.py']=b'bad'
            with self.assertRaisesRegex(ValueError,'Unsafe artifact path'): p.unpack_artifacts(*encode(content),dest)
            manifest,chunks=encode(artifact_fixture())
            with self.assertRaisesRegex(ValueError,'Packed hash'): p.unpack_artifacts(dict(manifest,sha256='0'*64),chunks,dest)
            with self.assertRaisesRegex(ValueError,'chunk count'): p.unpack_artifacts(manifest,[],dest)
            for field in ('complete','timing_receipt','sources_sha256','oracle_checks'):
                content=artifact_fixture();report=json.loads(content['activity.json']);report[field]={} if field in ('timing_receipt','sources_sha256') else False
                content['activity.json']=json.dumps(report).encode()
                with self.assertRaises(ValueError):p.unpack_artifacts(*encode(content),dest)
            self.assertFalse(dest.exists())

    def test_transport_binds_dynamic_worker_hash_and_retains_negative_screen(self):
        content=artifact_fixture()
        hashes=json.loads(content['timing.json'])['sources_sha256']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'admitted payload'):
                p.unpack_artifacts(*encode(content),root/'bad',dict(hashes,**{p.WORKER:'0'*64}))
            self.assertFalse((root/'bad').exists())
            self.assertTrue(p.unpack_artifacts(*encode(content),root/'good',hashes)['complete'])
            self.assertFalse(json.loads((root/'good'/'timing.json').read_text())['assessment']['screen_passed'])

    def test_success_requires_all_trace_hashes_profiles_and_stage_order(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'out'
            content=artifact_fixture();content.pop('activity.pooler_m7.p046.repeat2.trace.json')
            with self.assertRaisesRegex(ValueError,'activity trace'):p.unpack_artifacts(*encode(content),dest)
            content=artifact_fixture();content['activity.pooler_m8.control.repeat0.trace.json']=b'wrong'
            with self.assertRaisesRegex(ValueError,'activity trace'):p.unpack_artifacts(*encode(content),dest)
            for mutation in ('duplicate_profile','wrong_stage_order','failed_oracle'):
                content=artifact_fixture()
                if mutation=='wrong_stage_order':
                    report=json.loads(content['summary.json']);report['stages'].reverse();content['summary.json']=json.dumps(report).encode()
                else:
                    report=json.loads(content['activity.json'])
                    if mutation=='duplicate_profile':report['profiles'][-1]=dict(report['profiles'][0])
                    else:report['profiles'][0]['passed']=False
                    content['activity.json']=json.dumps(report).encode()
                with self.assertRaises(ValueError):p.unpack_artifacts(*encode(content),dest)
            self.assertFalse(dest.exists())

    def _remote_fixture(self, behavior):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';dest=root/'remote';source.mkdir();original_type=type(root)
            source_hashes={}
            for name in (*p.SOURCE_NAMES,p.RECEIPT):
                raw=('fixture '+name).encode();path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                source_hashes[name]=hashlib.sha256(raw).hexdigest()
            mappings={'/opt/projection-p053-phase':str(source),'/tmp/projection-p053-phase':str(dest)}
            def mapped_path(path):return original_type(mappings.get(str(path),path))
            commands=[]
            def popen(command,**kwargs):
                commands.append(command);self.assertNotIn('--compile-only',command)
                self.assertEqual(command[-4:] if len(commands)==1 else command[-6:-2],['--deadline-seconds','90','--outer-timeout-seconds','105'])
                if len(commands)==1:
                    self.assertEqual(command[command.index('--mode')+1],'timing')
                    report=json.loads(artifact_fixture()['timing.json'])
                    report['sources_sha256']={n:source_hashes[n] for n in p.SOURCE_NAMES}
                    report['inherited_p052']['sha256']=source_hashes[p.RECEIPT]
                    if behavior=='incomplete':report['complete']=False
                    if behavior=='missing_sample':report['cases'][0]['series']['partial']['samples'].pop()
                    if behavior=='wrong_source':report['sources_sha256'][p.WORKER]='0'*64
                    if behavior=='failed_replay':report['cases'][0]['series']['whole']['samples'][0]['correctness']['candidate']=False
                    if behavior=='failed_probe':report['cases'][1]['correctness'][0]['passed']=False
                    (dest/'timing.json').write_text(json.dumps(report))
                    if behavior=='timeout':
                        (dest/'timing.json.tmp').write_text('incomplete')
                        return Mock(pid=1234567,communicate=Mock(side_effect=[subprocess.TimeoutExpired('fixture',105),(b'timeout',None)]))
                    if behavior=='symlink_temp':(dest/'timing.json.tmp').symlink_to(dest/'timing.json')
                    if behavior=='unknown':(dest/'unknown.txt').write_text('reject')
                    if behavior=='failure':return Mock(returncode=1,communicate=Mock(return_value=(b'failed',None)))
                else:
                    self.assertEqual(command[command.index('--mode')+1],'activity');self.assertIn('--timing-receipt',command)
                    (dest/'activity.json').write_text('{"complete":true}')
                    (dest/'activity.json.tmp').write_text('incomplete temporary')
                return Mock(returncode=0,communicate=Mock(return_value=(b'fixture',None)))
            with patch('pathlib.Path',side_effect=mapped_path),patch('subprocess.Popen',side_effect=popen),patch('os.killpg') as kill:
                items=list(p.make_remote_entry()())
            if behavior in ('symlink_temp','unknown'):
                self.assertEqual(items,[dict(kind='failure',reason='artifact_path_guard')]);return commands,None
            manifest=items[0];chunks=[item['data'] for item in items if item['kind']=='chunk']
            self.assertEqual(items[-1]['kind'],'complete');self.assertFalse(any(n.endswith('.tmp') for n in manifest['artifact_sha256']))
            content=json.loads(zlib.decompress(b''.join(chunks)));summary=json.loads(base64.b64decode(content['summary.json']))
            if behavior=='timeout':
                kill.assert_called_once();self.assertEqual(summary['stages'][0]['exit_code'],124)
                out=root/'retrieved';self.assertFalse(p.unpack_artifacts(manifest,chunks,out)['complete'])
                self.assertEqual((out/'timing.log').read_bytes(),b'timeout')
            return commands,summary

    def test_remote_admits_activity_after_complete_timing_even_negative_performance(self):
        commands,summary=self._remote_fixture('valid')
        self.assertEqual(len(commands),2);self.assertTrue(summary['complete'])
        for behavior in ('incomplete','missing_sample','wrong_source','failed_replay','failed_probe'):
            commands,summary=self._remote_fixture(behavior)
            self.assertEqual(len(commands),1);self.assertFalse(summary['complete'])
            self.assertEqual(summary['gate_error'],'timing_receipt_rejected')

    def test_remote_failure_and_timeout_preserved_without_activity_run(self):
        for behavior in ('failure','timeout'):
            commands,summary=self._remote_fixture(behavior)
            self.assertEqual(len(commands),1);self.assertFalse(summary['complete'])

    def test_remote_rejects_symlink_temp_and_unknown_files(self):
        self._remote_fixture('symlink_temp');self._remote_fixture('unknown')

    def test_shutdown_exact_tasks_paid_zero_and_no_relaunch(self):
        source=Path(p.__file__).read_text()
        for value in ('stop()\n        timer.cancel()', 'exact and all_stopped([exact])',
                      'record["all_apps_stopped_tasks_zero"] = all_stopped(apps)',
                      'record["paid_after_zero"] = Decimal(after["billed_cost"]) == 0',
                      'record["state"] = "terminal" if terminal else "needs-reconciliation"'):
            self.assertIn(value,source)
        self.assertTrue(p.all_stopped([dict(state='stopped',tasks='0')]))
        self.assertFalse(p.all_stopped([dict(state='stopped',tasks='1')]))

    def test_self_test_does_not_import_sdk_or_call_external(self):
        with patch.object(p.subprocess,'run',side_effect=AssertionError('external')), \
             patch.object(p.subprocess,'check_output',side_effect=AssertionError('external')), \
             patch.object(p.Path,'is_file',return_value=True),patch.object(p.Path,'is_symlink',return_value=False),patch('builtins.print'):
            self.assertEqual(p.main(['self-test','--control-root','/unused']),0)


if __name__=='__main__': unittest.main()
