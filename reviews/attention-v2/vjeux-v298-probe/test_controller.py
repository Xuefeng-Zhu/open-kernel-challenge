"""Offline safety/source/lifecycle checks. No service or untrusted-source import."""
import base64
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import controller as c

class ControllerTests(unittest.TestCase):
    def artifact(self,name,raw):
        z=gzip.compress(raw,mtime=0)
        return dict(name=name,raw_bytes=len(raw),raw_sha256=c.digest(raw),gzip_bytes=len(z),gzip_sha256=c.digest(z),gzip_base64=base64.b64encode(z).decode())
    def test_exact_source_and_license(self):
        for name,want in c.EXPECTED.items(): self.assertEqual(c.digest(c.regular_bytes(c.payload()[name])),want)
        self.assertEqual(c.digest(c.regular_bytes(c.HERE/'upstream/LICENSE')),'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4')
    def test_payload_has_no_credentials_or_directories(self):
        self.assertEqual(set(c.payload()),{'worker.py','selected.py','v298.py','reference.py','checks.py','LICENSE','THIRD_PARTY_NOTICES.md','UPSTREAM_LICENSE'})
        self.assertTrue(all('/' not in name and 'local-notes' not in str(path) for name,path in c.payload().items()))
    def test_remote_has_no_local_closure(self):
        f=c.make_remote_entry();self.assertIsNone(f.__closure__)
        self.assertFalse({'AUTH','cli','ORIGINAL','PRIOR_RECEIPT'} & set(f.__code__.co_names))
    def test_bounded_network_blocked_one_gpu_no_retries(self):
        s=(c.HERE/'controller.py').read_text()
        for x in ['block_network=True','restrict_modal_access=True','max_containers=1','retries=0','gpu="H200"']: self.assertIn(x,s)
        self.assertLess(c.WORKER_SECONDS,c.CHILD_SECONDS);self.assertLess(c.CHILD_SECONDS,c.FUNCTION_SECONDS);self.assertLess(c.FUNCTION_SECONDS,c.WATCHDOG_SECONDS)
    def test_accounting_reuses_prior_evidence(self):
        x=c.historical_accounting();self.assertEqual(x['paid_limit_usd'],'0');self.assertFalse(x['fresh_billing_queried']);self.assertIsNone(x['actual_charge_usd']);self.assertGreater(float(x['modeled_balance_after_allowances_usd']),0)
    def test_shutdown_needs_exact_observed_zero_tasks(self):
        self.assertTrue(c.stopped([dict(state='stopped',tasks='0')]))
        for task in [1,'1',True,None]: self.assertFalse(c.stopped([dict(state='stopped',tasks=task)]))
        seq=[[dict(app_id='ap-test',state='stopped',tasks=1)],[dict(app_id='ap-test',state='stopped',tasks='0')]]
        with patch.object(c,'cli',side_effect=seq),patch.object(c.time,'sleep'):
            _,exact,observations=c.observe_shutdown('ap-test')
        self.assertEqual(len(observations),2);self.assertTrue(c.stopped([exact]))
    def test_artifact_roundtrip_and_traversal_rejected(self):
        raw=b'{"status":"failed"}\n'
        with tempfile.TemporaryDirectory() as d:
            m=c.unpack_artifacts({'exit_code':0,'artifacts':[self.artifact('summary.json',raw)]},Path(d)/'artifacts')
            self.assertEqual((Path(d)/'artifacts/summary.json').read_bytes(),raw)
            self.assertEqual(m['summary.json']['raw_sha256'],c.digest(raw))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):c.unpack_artifacts({'exit_code':0,'artifacts':[self.artifact('../escape',raw)]},Path(d)/'artifacts')
    def test_failed_worker_cannot_claim_completion(self):
        hashes={name:'a'*64 for name in c.payload()}
        r={'worker_sha256':hashes['worker.py'],'verified_hashes':{k:hashes[k] for k in ('selected.py','v298.py','reference.py','checks.py')},'status':'failed'}
        self.assertFalse(c.validate_report(r,hashes,{}))
    def complete_report(self,hashes):
        import copy
        checked={'passed':True,'ownership':True,'fixed_addresses':True}
        return dict(worker_sha256=hashes['worker.py'],verified_hashes={k:hashes[k] for k in ('selected.py','v298.py','reference.py','checks.py')},status='complete',candidate_changed=False,official_check=False,ranked_submission=False,correctness_complete=True,timing_complete=True,prerequisites_complete=True,
            prerequisite_cases=[{'arms':{arm:copy.deepcopy(checked) for arm in ('champion','v298')}} for _ in range(16)],
            paired_cases=[dict(seed=seed,case=dict(batch=b,q_len=l,layer=k),oracle={arm:copy.deepcopy(checked) for arm in ('champion','v298')},post_oracle={arm:copy.deepcopy(checked) for arm in ('champion','v298')},mutation_oracle={arm:dict(timed=False,**checked) for arm in ('champion','v298')},samples=[dict(champion=100.,v298=50.) for _ in range(12)]) for seed in (135901,135902) for b in (1,16,32) for l in (1,8) for k in ('sliding','full')],whole_suite=[dict(seed=seed) for seed in (135901,135902)])
    def test_complete_scope_and_missing_oracles(self):
        import copy
        h={name:'a'*64 for name in c.payload()};r=self.complete_report(h)
        self.assertTrue(c.validate_report(r,h,{}))
        for mutate in (lambda x:x['prerequisite_cases'].pop(),lambda x:x['paired_cases'].pop(),lambda x:x['paired_cases'][0]['mutation_oracle']['v298'].update(passed=False),lambda x:x.update(official_check=True)):
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):c.validate_report(bad,h,{})
    def test_original_files_never_written_by_controller(self):
        source=inspect.getsource(c.payload);self.assertIn('upstream/muse_attention_decode.py',source)
        self.assertFalse(c.HERE.joinpath('run.local.json').exists())

if __name__=='__main__':unittest.main()
