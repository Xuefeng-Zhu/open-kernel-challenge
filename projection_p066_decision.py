"""Read-only selected-receipt arithmetic, never a new performance measurement."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'candidates/projection_pooler_column_batch2_hybrid.py'
RECEIPT=ROOT/'results/remote-P2-09-pooler-column-batch2-hybrid.json'
SHA='001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e'

def run():
 d=json.loads(RECEIPT.read_text());assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
 assert d['admission']['submission_sha256']==SHA and d['admission']['job_id']=='41ba778763a5433b'
 rows=[r for r in d['result']['records'] if r.get('event')=='benchmark']
 assert len({(r['case']['load'],r['case']['layer']) for r in rows})==30
 assert len(rows)==30 and all(r['status']=='pass' and r['case']['weight']==62 for r in rows)
 loads={}
 for r in rows:loads.setdefault(r['case']['load'],[]).append(r)
 assert len(loads)==6 and all(len(x)==5 for x in loads.values())
 recomputed=math.prod(sum(r['baseline_ns']*62 for r in rs)/sum(r['median_ns']*62 for r in rs) for rs in loads.values())**(1/6)
 assert math.isclose(recomputed,d['result']['speedup'],rel_tol=1e-10)
 table=[];shares={k:[] for k in ['pooler','qkv_gate','unpooler','o','router']}
 for load,rs in loads.items():
  times={r['case']['layer']:r['median_ns']/1000 for r in rs};total=sum(times.values())
  table.append(dict(load=load,total_one_of_each_us=total,weighted_load_us=total*62,calibrated_us=times))
  for k,v in times.items():shares[k].append(v/total)
 # Scope confirmed by selected receipt's actual visible split-K+reducer phases.
 observed={(r['case']['layer'],r['case']['m']) for r in d['result']['records'] if r.get('event')=='profile' and r['case']['layer'] in ['pooler','qkv_gate','unpooler'] and 'splitKreduce_kernel' in r['report']}
 assert observed=={('qkv_gate',8),('qkv_gate',16),('qkv_gate',32),('qkv_gate',128),('pooler',128),('unpooler',8)}
 scopes={'three_families_all':lambda c:c['layer'] in ['pooler','qkv_gate','unpooler'],
         'three_families_m_gt1':lambda c:c['layer'] in ['pooler','qkv_gate','unpooler'] and c['m']>1,
         'observed_vendor_split_paths':lambda c:(c['layer'],c['m']) in observed,
         'qkv128_only':lambda c:c['layer']=='qkv_gate' and c['m']==128}
 models={}
 for name,predicate in scopes.items():
  fractions={load:sum(r['median_ns'] for r in rs if predicate(r['case']))/sum(r['median_ns'] for r in rs) for load,rs in loads.items()}
  def gain(cut):return math.prod(1/(1-cut*f) for f in fractions.values())**(1/6)-1
  assert gain(0)==0 and gain(.2)>=gain(.1)
  lo,hi=0,.999999
  for _ in range(60):
   mid=(lo+hi)/2
   if gain(mid)<.05:lo=mid
   else:hi=mid
  models[name]=dict(case_count=sum(predicate(r['case']) for r in rows),load_fractions=fractions,mean_share=sum(fractions.values())/6,
                   conditional_score_gain={str(c):gain(c) for c in [.1,.2,.3,.5]},uniform_cut_for_5pct_score=hi if gain(.999999)>=.05 else None)
 return dict(kind='selected-receipt recalculation and conditional models, not new timing',source_sha256=SHA,
             receipt_sha256=hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),job_id=d['admission']['job_id'],score=recomputed,
             measured_loads=table,mean_within_load_family_share={k:sum(v)/6 for k,v in shares.items()},models=models,
             no_gpu_or_official_check=True)

if __name__=='__main__':
 out=run();(ROOT/'results/projection-p066-decision.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'family_share':out['mean_within_load_family_share'],'models':out['models']},indent=2))
