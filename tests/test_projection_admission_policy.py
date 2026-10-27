"""Local fake-CLI admissions only: no subprocess, network, real ledger or GPU calls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cycle2_check as coordinator


class ProjectionAdmissionPolicy(unittest.TestCase):
    def setUp(self):
        # Keep all fixture writes in this worktree, without touching real state.
        self.temp = tempfile.TemporaryDirectory(prefix='policy-test-', dir=coordinator.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['candidates', 'results', 'experiments']:
            (self.root/name).mkdir()
        self.relative = 'candidates/projection_policy_fixture.py'
        self.path = self.root/self.relative
        self.source = b'# okc: problem=muse_bf16_proj\ndef prepare(w): return w\ndef kernel_fn(d, s): return d["x"]\n'
        self.path.write_bytes(self.source)
        self.sha = hashlib.sha256(self.source).hexdigest()
        self.commit = 'a'*40
        self.ledger_path = self.root/'experiments/ledger.json'
        self.slot_path = self.root/'slot.json'
        self.slot_path.write_text(json.dumps({'state':'idle'}))
        self.cpu_path = self.root/'results/cpu.json'
        self.cpu_path.write_text(json.dumps({'passed':True,'environment':{'submission_sha256':self.sha}}))
        self.ledger = {
            'maximum_new_experiments':10, 'remaining_authorized':5,
            'max_active_jobs':1, 'rules_version':'test-version',
            'current_authorization':{'personal_check_count_ceiling_removed':True},
            'plans':{'next':{'candidate':self.relative,'cpu_report':'results/cpu.json',
                             'hypothesis':'fixture only','control_job_id':'old-control'}},
            'requests':[{'experiment':'old-'+str(i),'state':'terminal',
                         'status':'done','job_id':'old-job-'+str(i)} for i in range(11)],
        }
        self.calls = []
        self.quota_left = 1
        self.rule_version = 'test-version'
        self.check_reply = {'job_id':'fake-accepted'}
        self.timeout_on_check = False
        self.cli_path = self.root/'fake-okc'
        self.patches = [patch.object(coordinator,'ROOT',self.root),
                        patch.object(coordinator,'LEDGER',self.ledger_path),
                        patch.object(coordinator,'GPU_SLOT',self.slot_path),
                        patch.object(coordinator,'CLI',self.cli_path),
                        patch.object(coordinator.subprocess,'run',side_effect=self.fake_run),
                        patch.object(coordinator.subprocess,'check_output',side_effect=self.fake_git)]
        for p in self.patches:
            p.start();self.addCleanup(p.stop)

    def fake_git(self, command, **kwargs):
        if command == ['git','log','-1','--format=%H','--',self.relative]:
            return (self.commit+'\n').encode()
        if command == ['git','show',self.commit+':'+self.relative]:
            return self.source
        raise AssertionError('Unexpected git request: '+repr(command))

    def fake_run(self, command, **kwargs):
        # Never invoke an actual command; unexpected calls fail closed.
        self.calls.append(command)
        if command == ['curl','-fsSL','https://openkernelchallenge.dev/api/v1/rules']:
            reply = {'version':self.rule_version}
        elif command == [str(self.cli_path),'problems','--json']:
            reply = {'problems':[{'name':'muse_bf16_proj','limits':{
                'checks':{'left':self.quota_left},'scored':{'left':0}}}]}
        elif command == [str(self.cli_path),'check',self.relative,'--no-wait','--json']:
            # Reservation and shared-slot ownership must precede the sole fake POST.
            saved = self.read_ledger()
            self.assertEqual(saved['requests'][-1]['state'],'sending')
            self.assertEqual(saved['requests'][-1]['submission_sha256'],self.sha)
            self.assertEqual(json.loads(self.slot_path.read_text())['state'],'sending')
            if self.timeout_on_check:
                raise subprocess.TimeoutExpired(command,45)
            reply = self.check_reply
        else:
            raise AssertionError('Unexpected external command: '+repr(command))
        return subprocess.CompletedProcess(command,0,json.dumps(reply),'')

    def run_send(self):
        self.ledger_path.write_text(json.dumps(self.ledger))
        with patch('sys.argv',['cycle2_check.py','send','next',str(self.path)]),contextlib.redirect_stdout(io.StringIO()):
            return coordinator.main()

    def read_ledger(self):
        return json.loads(self.ledger_path.read_text())

    def fake_posts(self):
        return [c for c in self.calls if len(c)>1 and c[0]==str(self.cli_path) and c[1]=='check']

    def test_explicit_uncapped_policy_admits_over_ten_and_preserves_history(self):
        self.assertEqual(self.run_send(),0)
        saved = self.read_ledger()
        self.assertEqual(len(saved['requests']),12)
        self.assertEqual(saved['maximum_new_experiments'],10)
        self.assertEqual(saved['remaining_authorized'],5)
        self.assertEqual(saved['requests'][:-1],self.ledger['requests'])
        self.assertEqual(saved['requests'][-1]['job_id'],'fake-accepted')
        self.assertEqual(saved['requests'][-1]['source_commit'],self.commit)
        self.assertEqual(len(self.fake_posts()),1)

    def test_missing_false_or_malformed_authorization_keeps_legacy_cap(self):
        for value in ['missing',None,{},False,{'personal_check_count_ceiling_removed':False},
                      {'personal_check_count_ceiling_removed':'true'},
                      {'personal_check_count_ceiling_removed':1}]:
            with self.subTest(policy=value):
                self.ledger.pop('current_authorization',None)
                if value!='missing':self.ledger['current_authorization']=value
                with self.assertRaisesRegex(RuntimeError,'authorization exhausted'):
                    self.run_send()
                self.assertEqual(self.calls,[])
                self.assertEqual(self.read_ledger()['requests'],self.ledger['requests'])

    def test_legacy_policy_below_cap_admits_and_updates_remaining(self):
        self.ledger.pop('current_authorization')
        self.ledger['requests']=self.ledger['requests'][:8]
        self.assertEqual(self.run_send(),0)
        self.assertEqual(len(self.read_ledger()['requests']),9)
        self.assertEqual(self.read_ledger()['remaining_authorized'],1)
        self.assertEqual(self.read_ledger()['maximum_new_experiments'],10)
        self.assertEqual(len(self.fake_posts()),1)

    def test_daily_zero_still_blocks_without_reservation(self):
        self.quota_left=0
        with self.assertRaisesRegex(RuntimeError,'quota exhausted'):self.run_send()
        self.assertEqual(self.fake_posts(),[])
        self.assertEqual(self.read_ledger()['requests'],self.ledger['requests'])
        self.assertEqual(json.loads(self.slot_path.read_text())['state'],'idle')

    def test_active_projection_still_blocks(self):
        self.ledger['requests'][0].update(state='accepted',status='running')
        with self.assertRaisesRegex(RuntimeError,'active-job limit'):self.run_send()
        self.assertEqual(self.calls,[])

    def test_shared_attention_slot_still_blocks(self):
        raw=json.dumps({'state':'accepted','problem':'muse_attention_decode','experiment':'active'})
        self.slot_path.write_text(raw)
        with self.assertRaisesRegex(RuntimeError,'GPU slot not free'):self.run_send()
        self.assertEqual(self.calls,[])
        self.assertEqual(self.slot_path.read_text(),raw)

    def test_sending_and_uncertain_reservations_still_block(self):
        for state in ['sending','uncertain']:
            with self.subTest(state=state):
                self.ledger['requests'][0].update(state=state,status='unknown',job_id=None)
                with self.assertRaisesRegex(RuntimeError,'Uncertain request reserved'):self.run_send()
                self.assertEqual(self.calls,[])

    def test_post_timeout_retains_one_uncertain_reservation_and_no_retry(self):
        self.timeout_on_check=True
        with self.assertRaises(subprocess.TimeoutExpired):self.run_send()
        saved=self.read_ledger()
        self.assertEqual(len(saved['requests']),12)
        self.assertEqual(saved['requests'][-1]['state'],'uncertain')
        self.assertEqual(saved['remaining_authorized'],5)
        self.assertEqual(json.loads(self.slot_path.read_text())['state'],'uncertain')
        self.assertEqual(len(self.fake_posts()),1)

    def test_unacknowledged_post_stays_uncertain(self):
        self.check_reply={}
        self.assertEqual(self.run_send(),2)
        self.assertEqual(self.read_ledger()['requests'][-1]['state'],'uncertain')
        self.assertEqual(json.loads(self.slot_path.read_text())['state'],'uncertain')
        self.assertEqual(len(self.fake_posts()),1)

    def test_rule_version_guard_still_blocks(self):
        self.rule_version='changed'
        with self.assertRaisesRegex(RuntimeError,'Rules version changed'):self.run_send()
        self.assertEqual(self.fake_posts(),[])
        self.assertEqual(self.read_ledger()['requests'],self.ledger['requests'])

    def test_changed_source_and_cpu_hash_still_block(self):
        self.path.write_bytes(self.source+b'# uncommitted\n')
        with self.assertRaisesRegex(RuntimeError,'differs from its source commit'):self.run_send()
        self.assertEqual(self.calls,[])
        self.path.write_bytes(self.source)
        self.cpu_path.write_text(json.dumps({'passed':True,'environment':{'submission_sha256':'wrong'}}))
        with self.assertRaisesRegex(RuntimeError,'Missing passing CPU gate'):self.run_send()
        self.assertEqual(self.calls,[])

    def test_pause_and_duplicate_guards_still_block(self):
        self.ledger['admission_paused']=True
        with self.assertRaisesRegex(RuntimeError,'Official dispatch remains paused'):self.run_send()
        self.ledger['admission_paused']=False
        self.ledger['requests'][0]['experiment']='next'
        with self.assertRaisesRegex(RuntimeError,'already journaled'):self.run_send()
        self.assertEqual(self.calls,[])
