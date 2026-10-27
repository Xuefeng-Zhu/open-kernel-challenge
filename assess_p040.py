"""Offline P040 receipt comparison, preserving the predeclared15%target goal."""
from datetime import datetime,timezone
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 a=json.loads((ROOT/'results/remote-P2-04-router-m1-fp32-gemv.json').read_text())
 b=json.loads((ROOT/'results/remote-P2-05-o-m1-group8.json').read_text())
 assert b['admission']['job_id']=='d57d89967d1a408f' and b['result']['passed']
 old={(r['case']['load'],r['case']['layer']):r for r in a['result']['records'] if r['event']=='benchmark'}
 rows=[]
 for r in b['result']['records']:
  if r['event']!='benchmark':continue
  c=r['case'];p=old[c['load'],c['layer']];relative=r['paired_ratio']/p['paired_ratio']
  rows.append(dict(load=c['load'],layer=c['layer'],m=c['m'],target=c['layer']=='o' and c['m']==1,
   parent_us=p['median_ns']/1000,candidate_us=r['median_ns']/1000,parent_sglang_us=p['baseline_ns']/1000,sglang_us=r['baseline_ns']/1000,
   parent_raw_us=p['median_raw_ns']/1000,candidate_raw_us=r['median_raw_ns']/1000,
   parent_paired_ratio=p['paired_ratio'],paired_ratio=r['paired_ratio'],relative_speedup=relative,normalized_latency_cut=1-1/relative))
 assert len(rows)==30
 t=next(r for r in rows if r['target']);controls=[r for r in rows if not r['target']]
 load=a['weighted_loads']['decode 1']['candidate_load_us'];equivalent=t['parent_sglang_us']/t['paired_ratio']
 modeled=a['result']['speedup']*(load/(load-62*(t['parent_us']-equivalent)))**(1/6)
 oldci=a['result']['speedup_ci'];ci=b['result']['speedup_ci']
 out=dict(observed_utc=datetime.now(timezone.utc).isoformat(),parent_job=a['admission']['job_id'],job=b['admission']['job_id'],source_commit=b['admission']['source_commit'],source_sha256=b['admission']['submission_sha256'],passed=True,
  same_evaluator=a['result']['evaluator']==b['result']['evaluator'],parent_score=a['result']['speedup'],score=b['result']['speedup'],score_ratio=b['result']['speedup']/a['result']['speedup'],ci=ci,parent_ci=oldci,
  ci_overlap=max(ci[0],oldci[0])<=min(ci[1],oldci[1]),target=t,all30shapes=rows,predeclared_minimum_latency_cut=.15,target_goal_met=t['normalized_latency_cut']>=.15,
  control_max_abs_latency_shift=max(abs(r['normalized_latency_cut']) for r in controls),controls_over5pct_latency_regression=[r for r in controls if r['normalized_latency_cut']<-.05],
  target_only_frozen_parent_model_score=modeled,
  decision='Reject promotion: native correct but6.5%target gain is below15%goal and overall0.22%is indistinguishable from noise/inconclusive. RetainP039; no grouping/tile sweep.',
  evidence_limits=['Separate jobs not directly paired candidates','Separate one-off profile is not median','No register/shared/spill/DRAM/occupancy/stall measurements',
   'New native tails/strides/outliers/concurrent stress not separately run; CPUmodels not GPU proof'])
 (ROOT/'results/router-p040-official-assessment.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k in ['score','score_ratio','ci_overlap','target_goal_met','control_max_abs_latency_shift','target_only_frozen_parent_model_score','decision']}))

if __name__=='__main__':main()
