"""Offline analytical/error-path tests; synthetic receipts are never measurements."""
import copy
import json
import math
import unittest
from assess_p062 import ROOT,RAW_SHA,BASELINES,assess,normalized_latency


class RawControlAssessment(unittest.TestCase):
    def setUp(self):
        self.parent=json.loads((ROOT/'results/remote-P2-09-pooler-column-batch2-hybrid.json').read_text())
        self.synthetic=copy.deepcopy(self.parent)
        self.synthetic['admission'].update(submission_sha256=RAW_SHA,mode='check',job_id='synthetic-only')
        self.compressed=copy.deepcopy(self.parent)
        # Synthetic arithmetic fixture only; it is never written as a receipt.
        self.compressed['admission'].update(job_id=BASELINES[0][0],submission_sha256=BASELINES[0][1])
        for r in self.synthetic['result']['records']:
            if r.get('event')=='profile' and r['case']['layer']=='pooler' and r['case']['m'] in (8,16,32):
                r['report']='_pooler_codec_partial\n_reduce\n'

    def test_ratio_direction_and_invalid_values(self):
        self.assertEqual(normalized_latency(2,1),.5)
        self.assertEqual(normalized_latency(.5,1),2)
        for x in (0,-1,math.inf,math.nan):
            with self.assertRaises(ValueError):normalized_latency(x,1)

    def test_identical_reference_ratios_mean_no_gain(self):
        r=assess(self.synthetic,self.compressed,self.parent)
        self.assertTrue(r['custom_path_execution_verified'])
        self.assertEqual(r['control_count'],27)
        self.assertEqual(r['controls_over5pct'],[])
        self.assertTrue(all(x['raw_over_selected']==1 for x in r['targets']))
        self.assertTrue(r['selected_preserved'])

    def test_large_target_deficit_closes_only_this_consumer(self):
        for r in self.synthetic['result']['records']:
            if r.get('event')=='benchmark' and r['case']['layer']=='pooler' and r['case']['m'] in (8,16,32):
                r['paired_ratio']/=2
        out=assess(self.synthetic,self.compressed,self.parent)
        self.assertIn('deprioritize this custom consumer',out['decision'])
        self.assertTrue(all(x['raw_over_selected']==2 for x in out['targets']))

    def test_wrong_source_pending_or_missing_case_rejected(self):
        for mutation in ('source','pending','case'):
            r=copy.deepcopy(self.synthetic)
            if mutation=='source':r['admission']['submission_sha256']='bad'
            elif mutation=='pending':r['status']='queued'
            else:r['result']['records']=[x for x in r['result']['records'] if not (x.get('event')=='benchmark' and x['case']['m']==8 and x['case']['layer']=='pooler')]
            with self.assertRaises(ValueError):assess(r,self.compressed,self.parent)

    def test_missing_profile_prevents_execution_claim(self):
        self.synthetic['result']['records']=[r for r in self.synthetic['result']['records'] if r.get('event')!='profile']
        out=assess(self.synthetic,self.compressed,self.parent)
        self.assertFalse(out['custom_path_execution_verified'])
        self.assertIn('unresolved',out['decision'])

    def test_failure_produces_no_timing_claim(self):
        self.synthetic['result']['passed']=False
        out=assess(self.synthetic,self.compressed,self.parent)
        self.assertFalse(out['passed'])
        self.assertNotIn('score',out)

    def test_wrong_historical_identity_rejected(self):
        with self.assertRaises(ValueError):assess(self.synthetic,self.parent,self.parent)
        self.parent['admission']['job_id']='wrong'
        with self.assertRaises(ValueError):assess(self.synthetic,self.compressed,self.parent)

    def test_changed_evaluator_limits_attribution(self):
        self.synthetic['result']['evaluator']['code_sha256']='changed'
        out=assess(self.synthetic,self.compressed,self.parent)
        self.assertFalse(out['same_evaluator'])
        self.assertIn('limit attribution',out['decision'])

    def test_control_drift_limits_attribution(self):
        row=next(r for r in self.synthetic['result']['records'] if r.get('event')=='benchmark' and r['case']['layer']=='o')
        row['paired_ratio']/=1.06
        out=assess(self.synthetic,self.compressed,self.parent)
        self.assertEqual(len(out['controls_over5pct']),1)
        self.assertIn('limit attribution',out['decision'])
