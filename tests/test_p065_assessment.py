"""Synthetic arithmetic/error cases only; never saved as measured receipts."""
import copy,json,unittest
from assess_p065 import ROOT,SHA,assess
class Assessment(unittest.TestCase):
 def setUp(self):
  self.s=json.loads((ROOT/'results/remote-P2-09-pooler-column-batch2-hybrid.json').read_text())
  self.e=json.loads((ROOT/'results/remote-P2-07-pooler-lossless-codec.json').read_text())
  self.c=copy.deepcopy(self.s);self.c['admission'].update(submission_sha256=SHA,job_id='synthetic')
  for r in self.c['result']['records']:
   if r.get('event')=='profile' and r['case']['layer']=='pooler' and r['case']['m'] in (8,16,32):r['report']='_pooler_codec_partial\n_reduce'
 def test_identical_ratios_no_gain(self):
  out=assess(self.c,self.s,self.e)
  self.assertTrue(out['codec_execution_verified']);self.assertFalse(out['promotion_screen'])
  self.assertEqual(out['target_only_modeled_score_ratio'],1)
  self.assertTrue(all(r['candidate_vs_selected_normalized_latency']==1 for r in out['targets']))
 def test_target_regression_and_control_drift_not_promoted(self):
  for r in self.c['result']['records']:
   if r.get('event')=='benchmark' and r['case']['m']==8:r['paired_ratio']/=2
  out=assess(self.c,self.s,self.e)
  self.assertIn('Reject P065',out['decision']);self.assertFalse(out['promotion_screen']);self.assertTrue(out['controls_over5pct'])
  self.assertLess(out['target_only_modeled_score_ratio'],1)
 def test_wrong_identity_or_pending_rejected(self):
  for key,value in [('status','queued'),('source','bad')]:
   c=copy.deepcopy(self.c)
   if key=='source':c['admission']['submission_sha256']=value
   else:c[key]=value
   with self.assertRaises(ValueError):assess(c,self.s,self.e)
 def test_native_failure_has_no_score_claim(self):
  self.c['result']['passed']=False
  out=assess(self.c,self.s,self.e);self.assertFalse(out['passed']);self.assertNotIn('score',out)
 def test_missing_profile_is_unresolved(self):
  self.c['result']['records']=[x for x in self.c['result']['records'] if x.get('event')!='profile']
  self.assertFalse(assess(self.c,self.s,self.e)['codec_execution_verified'])
if __name__=='__main__':unittest.main()
