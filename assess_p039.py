"""Saved official P039 comparison only; no GPU, CLI or credentials."""
from datetime import datetime,timezone
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 a=json.loads((ROOT/'results/remote-P2-02-full-composition.json').read_text())
 b=json.loads((ROOT/'results/remote-P2-04-router-m1-fp32-gemv.json').read_text())
 assert b['admission']['job_id']=='d5636ac8d1404abd' and b['result']['passed']
 before={(r['case']['load'],r['case']['layer']):r for r in a['result']['records'] if r['event']=='benchmark'}
 rows=[]
 for r in b['result']['records']:
  if r['event']!='benchmark':continue
  c=r['case'];p=before[c['load'],c['layer']];relative=r['paired_ratio']/p['paired_ratio']
  rows.append(dict(load=c['load'],layer=c['layer'],m=c['m'],target=c['layer']=='router' and c['m']==1,
   parent_us=p['median_ns']/1000,candidate_us=r['median_ns']/1000,parent_sglang_us=p['baseline_ns']/1000,sglang_us=r['baseline_ns']/1000,
   parent_raw_us=p['median_raw_ns']/1000,candidate_raw_us=r['median_raw_ns']/1000,
   parent_paired_ratio=p['paired_ratio'],paired_ratio=r['paired_ratio'],relative_speedup=relative,normalized_latency_cut=1-1/relative))
 assert len(rows)==30
 target=next(r for r in rows if r['target']);controls=[r for r in rows if not r['target']]
 load=a['weighted_loads']['decode 1']['candidate_load_us']
 equivalent=target['parent_sglang_us']/target['paired_ratio']
 modeled=a['result']['speedup']*(load/(load-62*(target['parent_us']-equivalent)))**(1/6)
 oldci=a['result']['speedup_ci'];newci=b['result']['speedup_ci']
 out=dict(observed_utc=datetime.now(timezone.utc).isoformat(),parent_job=a['admission']['job_id'],job=b['admission']['job_id'],source_commit=b['admission']['source_commit'],source_sha256=b['admission']['submission_sha256'],
  passed=True,same_evaluator=a['result']['evaluator']==b['result']['evaluator'],parent_score=a['result']['speedup'],score=b['result']['speedup'],score_ratio=b['result']['speedup']/a['result']['speedup'],parent_ci=oldci,ci=newci,
  ci_overlap=max(oldci[0],newci[0])<=min(oldci[1],newci[1]),target=target,all30shapes=rows,
  unchanged_controls_over5pct_speedup_shift=[r for r in controls if abs(r['relative_speedup']-1)>.05],
  controls_over5pct_latency_regression=[r for r in controls if r['normalized_latency_cut']<-.05],
  target_only_frozen_parent_model_score=modeled,
  decision='Retain P039 for correct native30%target latency reduction; highest verified point score, but aggregate gain over previous best is unresolved by overlapping CIs and shifted controls',
  limitations=['Separate jobs, no directly paired candidates','Positive unchanged-control shifts cannot be assigned to P039',
   'No compiled registers/shared/spill/counter capture for this candidate','CPU tail/outlier/stride/probe coverage is not native execution',
   'Native target scoredm1 canonical FP32 correctness/fresh graph verified; no explicit native tail/stride/concurrent stress'])
 (ROOT/'results/router-p039-official-assessment.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k in ['score','score_ratio','ci_overlap','same_evaluator','target_only_frozen_parent_model_score','unchanged_controls_over5pct_speedup_shift']}))

if __name__=='__main__':main()
