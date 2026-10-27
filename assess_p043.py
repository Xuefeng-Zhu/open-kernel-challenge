"""Offline P043 official receipt comparison; never issues GPU or network requests."""
import json,math
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPERIMENT='P2-07-pooler-lossless-codec'

def assess(parent,candidate):
 result=candidate['result'];passed=result.get('passed') is True
 base=dict(observed_utc=datetime.now(timezone.utc).isoformat(),
  parent_job=parent['admission']['job_id'],job=candidate['admission']['job_id'],
  source_commit=candidate['admission']['source_commit'],source_sha256=candidate['admission']['submission_sha256'],
  status=candidate['status'],passed=passed,selected_parent_unchanged=True,
  prerequisite_tests=candidate['prerequisite_tests'],shape_checks=candidate['shape_checks'])
 if not passed:
  return dict(**base,decision='Reject promotion: official correctness/compilation did not pass; preserve P039.',
   error=result.get('error'),records=result.get('records',[]),
   limitations=['No valid complete performance comparison from a failed check; no automatic retry.'])
 old={(r['case']['load'],r['case']['layer']):r for r in parent['result']['records'] if r.get('event')=='benchmark'}
 rows=[];equivalent_loads={k:v['candidate_load_us'] for k,v in parent['weighted_loads'].items()}
 for r in result['records']:
  if r.get('event')!='benchmark':continue
  c=r['case'];p=old[c['load'],c['layer']];relative=r['paired_ratio']/p['paired_ratio']
  target=c['layer']=='pooler' and c['m'] in (8,16,32)
  row=dict(load=c['load'],layer=c['layer'],m=c['m'],target=target,
   parent_us=p['median_ns']/1000,candidate_us=r['median_ns']/1000,
   parent_reference_us=p['baseline_ns']/1000,reference_us=r['baseline_ns']/1000,
   parent_raw_us=p['median_raw_ns']/1000,candidate_raw_us=r['median_raw_ns']/1000,
   parent_paired_ratio=p['paired_ratio'],paired_ratio=r['paired_ratio'],relative_speedup=relative,
   normalized_latency_cut=1-1/relative)
  rows.append(row)
  if target:equivalent_loads[c['load']]+=c['weight']*(row['parent_reference_us']/r['paired_ratio']-row['parent_us'])
 assert len(rows)==30
 targets=[r for r in rows if r['target']];controls=[r for r in rows if not r['target']]
 assert len(targets)==3
 model_factor=math.prod(parent['weighted_loads'][k]['candidate_load_us']/v for k,v in equivalent_loads.items())**(1/6)
 profiles=[dict(case=r['case'],report=r['report']) for r in result['records'] if r.get('event')=='profile' and r['case']['layer']=='pooler' and r['case']['m'] in (8,16,32)]
 codec_profiles=[p['case']['m'] for p in profiles if '_pooler_codec_partial' in p['report'] and '_reduce' in p['report']]
 gates=dict(each_target_normalized_cut_at_least_10pct=all(r['normalized_latency_cut']>=.10 for r in targets),
  no_target_regression_over5pct=all(r['normalized_latency_cut']>=-.05 for r in targets),
  target_only_modeled_aggregate_gain_at_least1p5pct=model_factor>=1.015,
  execution_profiles_show_all_three_codec_paths=sorted(codec_profiles)==[8,16,32])
 qualified=all(gates.values())
 oldci=parent['result']['speedup_ci'];ci=result['speedup_ci']
 return dict(**base,parent_score=parent['result']['speedup'],score=result['speedup'],score_ratio=result['speedup']/parent['result']['speedup'],
  parent_ci=oldci,ci=ci,ci_overlap=max(ci[0],oldci[0])<=min(ci[1],oldci[1]),
  same_evaluator=parent['result']['evaluator']==result['evaluator'],
  targets=targets,all30shapes=rows,codec_profiles=profiles,gates=gates,gates_passed=qualified,
  target_only_frozen_parent_model_factor=model_factor,
  target_only_frozen_parent_model_score=parent['result']['speedup']*model_factor,
  unchanged_controls_over5pct_latency_movement=[r for r in controls if abs(r['normalized_latency_cut'])>.05],
  controls_over5pct_latency_regression=[r for r in controls if r['normalized_latency_cut']<-.05],
  decision=('Passes predeclared candidate gates; retain as promising pending independent paired confirmation, preserve selected P039.' if qualified else 'Reject promotion: predeclared target/score/execution gates missed; preserve P039 and stop this fixed codec mechanism without a tile sweep.'),
  limitations=['Separate jobs, reference-normalized comparisons; no direct paired P039/candidate repeat',
   'Matched raw-control source was not GPU measured; elapsed time cannot identify compression or integer decoding causality',
   'One-off instrumented profiles verify path execution; not the calibrated latency or DRAM/register/stall counters',
   'No generated PTX/SASS or register/shared/spill metadata for this candidate',
   'Dedicated GPU all-bit decoder, graph A-B-A, changed weights/strides and concurrency stress not separately run; CPU tests do not prove these'])

if __name__=='__main__':
 parent=json.loads((ROOT/'results/remote-P2-04-router-m1-fp32-gemv.json').read_text())
 candidate=json.loads((ROOT/('results/remote-'+EXPERIMENT+'.json')).read_text())
 out=assess(parent,candidate)
 (ROOT/'results/router-p043-official-assessment.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ('job','passed','score','score_ratio','targets','gates','target_only_frozen_parent_model_factor','decision') if k in out}))
