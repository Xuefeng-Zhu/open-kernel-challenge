import ast, hashlib, json, math
from pathlib import Path
root=Path(__file__).resolve().parents[3]
out=root/'reviews/attention-v2/fa3-next-evidence'
source=root/'reviews/attention-v2/fusion-lookahead-probe/candidate.py'
profile=root/'reviews/attention-v2/fusion-lookahead-probe/official-check/artifacts/candidate.5090b83f1d63407b.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
assert sha(profile)=='3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300'
x=json.loads(profile.read_text())
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
node.returns=None
for a in node.args.args:a.annotation=None
ns={}
for name in ('_fa3_full_verify_fused','_fa3_sliding_verify_fused','_fa3_full_verify','_fa3_sliding_verify','_e2_kernel_fn'):
 ns[name]=lambda data,name=name:name
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'selected-dispatch-only','exec'),ns)
profiles={r['index']:r for r in x['profiles']}
rows=[]
for b in x['benchmarks']:
 c=b['case']; B=c['batch']; L=c.get('q_len',1); window=2048 if c['layer']=='sliding' else 0
 cfg={'batch':B,'q_len':L,'window':window,'rope_theta':500000 if window else 0}
 route=ns['kernel_fn']({'config':cfg})
 same_load=[q for q in x['benchmarks'] if q['case']['load']==c['load']]
 total=sum(q['case']['weight']*q['median_ns']/1000 for q in same_load)
 frac=c['weight']*b['median_ns']/1000/total
 grids=None
 if route=='_e2_kernel_fn':
  assert L==1
  S=(86001+8191)//8192 if window==0 else min((264+2*B-1)//(2*B),2048//128)
  grids={'producer':[B,2,S],'producer_CTA':B*2*S,'merge':[B,16],'merge_CTA':B*16,'splits':S,'query_rows_allocated':16,'query_rows_valid':8}
 rows.append({'load':c['load'],'layer':c['layer'],'batch':B,'q_len':L,'current_route':route,'whole_call_adjusted_us':b['median_ns']/1000,'same_run_reference_us':b['baseline_ns']/1000,'within_load_weight':c['weight'],'fraction_of_own_weighted_load':frac,'local_log_score_sensitivity':frac/6,'score_factor_if_this_path_10pct_faster':(1/(1-.1*frac))**(1/6),'source_derived_custom_grid':grids,'instrumented_profile_report':profiles[b['index']]['report']})
assert len(rows)==12
assert sum(r['current_route']=='_e2_kernel_fn' for r in rows)==4
res={'status':'LOCAL_SOURCE_AND_SAVED_RESULTS_ONLY; no GPU/native compile/service launch', 'selected_sha256':sha(source),'official_profile_sha256':sha(profile),'official_job_id':'5090b83f1d63407b','score':x['summary']['speedup'],'ci':x['summary']['speedup_ci'],'method':'Dispatch evaluated from isolated selected kernel_fn AST with pure return-name stubs; no candidate imports, Torch, Triton or GPU. Custom grids use scored max page-table shape86001 for agentic decode32, not observed CTA residency. Sensitivity freezes all other times and references. Activities are instrumented reports, not critical-path decomposition.','rows':rows}
(out/'baseline_matrix.json').write_text(json.dumps(res,indent=2)+'\n')
print('12 rows,4 custom/8 fused FA3; source and profile hashes verified')
for r in rows: print(r['load'],r['layer'],r['current_route'],round(r['whole_call_adjusted_us'],3),round(100*r['fraction_of_own_weighted_load'],2),r['source_derived_custom_grid'])
