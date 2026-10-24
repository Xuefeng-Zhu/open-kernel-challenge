"""Authorized bounded attention tests; explicit allowlist, no OKC or credential creation."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEDGER=HERE/'modal-budget.json'
FIXED=['submission.py','modal_worker.py','fixtures.json','official/reference.py',
       'official/checks.py','official/baseline.py','official/TASK.md','official/AGENTS.md','official/template.py']
LEGAL={'legal/LICENSE':'LICENSE','legal/THIRD_PARTY_NOTICES.md':'THIRD_PARTY_NOTICES.md',
       'legal/June-9-Researcher-Reciprocity.txt':'licenses/June-9-Researcher-Reciprocity.txt'}


def now():return datetime.now(timezone.utc).isoformat()
def save(d):LEDGER.write_text(json.dumps(d,indent=2)+'\n')
def read_cli(*args):
    p=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','modal',*args],capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError('Supported Modal read failed, no launch')
    return json.loads(p.stdout)


def bundle(mode,destination,experiment=None):
    files={p:HERE/p for p in FIXED}
    files.update({p:ROOT/src for p,src in LEGAL.items()})
    if mode=='merge':files['separate_merge.py']=HERE/'separate_merge.py'
    probe=None
    if mode=='replay':
        probe=json.loads(LEDGER.read_text())['attention_probe_plans'][experiment]
        for name in [probe['parent'],probe['candidate'],'replay_probe.py']:files[name]=HERE/name
        for name,sha in probe['expected_sha256'].items():
            if hashlib.sha256(files[name].read_bytes()).hexdigest()!=sha:raise ValueError('Planned probe source changed')
    for name,p in files.items():
        if p.is_symlink() or p.read_bytes()!=subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT):
            raise ValueError('Commit reviewed exact file before bundling: '+name)
    destination=destination.resolve()
    if (ROOT/'local-notes').resolve() not in destination.parents:raise ValueError('Use ignored local-notes bundle')
    destination.mkdir(exist_ok=False)
    for name,p in files.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    manifest=dict(kind='Explicit attention source/license allowlist; no repository/auth contents',mode=mode,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()})
    if probe:manifest['probe']=probe
    (destination/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built',len(files),'allowlisted files for',mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','run','finalize','status'])
    parser.add_argument('--mode',choices=['baseline','merge','replay'],default='baseline')
    parser.add_argument('--experiment',choices=['A03-baseline','A05-merge','A07-replay','A09-pipeline-probe'],default='A03-baseline')
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args()
    if args.action=='build':bundle(args.mode,args.bundle,args.experiment);return
    d=json.loads(LEDGER.read_text())
    auth=ROOT/'local-notes/modal-auth.toml'
    if not auth.exists() or auth.stat().st_mode&0o077:raise ValueError('Existing approved owner-only auth required')
    os.environ['MODAL_CONFIG_PATH']=str(auth)
    if args.action=='status':
        print(read_cli('app','list','--json'));print(read_cli('billing','summary','--json'));return
    if args.action=='finalize':
        e=next(e for e in d['attempts'] if e['experiment']==args.experiment)
        app=next(a for a in read_cli('app','list','--json') if a['app_id']==e['app_id'])
        if app['state']!='stopped' or int(app['tasks']):raise RuntimeError('Exact app is not stopped')
        raw=json.loads((ROOT/f'local-notes/attention-{args.experiment}-result.local.json').read_text())
        billing=read_cli('billing','summary','--json')
        elapsed=(datetime.fromisoformat(app['stopped_at'])-datetime.fromisoformat(app['created_at'])).total_seconds()
        upper=Decimal(math.ceil(elapsed+5))*(Decimal('.001261')+2*Decimal('.0000131')+8*Decimal('.00000222'))*Decimal('1.1')+Decimal('.01')
        e.update(state='terminal',app_lifetime_seconds=elapsed,conservative_cost_upper_usd=str(upper),
            platform_state='stopped',tasks=0,termination_verified_utc=now(),billing_after=billing,
            correctness_and_graph_passed=raw['exit_code']==0 and bool((raw['report'] or {}).get('passed')))
        calculated=Decimal('.12054281')+sum((Decimal(a.get('conservative_cost_upper_usd','0')) for a in d['attempts']),Decimal(0))
        d['conservative_stage_spent_usd']=str(max(Decimal(d['conservative_stage_spent_usd']),calculated))
        observed=max(Decimal(billing['metered_cost']),sum((Decimal(v) for v in billing['metered_cost_breakdown'].values()),Decimal(0)))
        d['observed_gross_high_water_usd']=str(max(Decimal(d['observed_gross_high_water_usd']),observed))
        d.update(final_app_states=read_cli('app','list','--json'),final_billing_snapshot=billing,latest_billing_observed_utc=now());save(d)
        (ROOT/'local-notes/modal-budget-latest.local.json').write_text(json.dumps(d,indent=2)+'\n')
        evidence=dict(kind='Attention Modal diagnostic, not OKC result',attempt=e,app=app,
            result={k:v for k,v in raw.items() if k!='log'},failure_log=raw['log'] if raw['exit_code'] else None,
            billing_snapshot=billing,conservative_stage_spent_usd=d['conservative_stage_spent_usd'])
        encoded=json.dumps(evidence,indent=2)+'\n'
        if '/Users/' in encoded or re.search(r'\b(?:ak|as)-[A-Za-z0-9]{8,}',encoded):raise ValueError('Unexpected private path/credential identifier')
        (HERE/(args.experiment+'-result.json')).write_text(encoded)
        print(args.experiment,'passed',e['correctness_and_graph_passed'],'stopped',e['app_id'],
              'guard',d['conservative_stage_spent_usd'],'net',billing['billed_cost'])
        if Decimal(billing['billed_cost'])!=0 or Decimal(d['conservative_stage_spent_usd'])>2:raise RuntimeError('Cost gate, stop')
        return
    expected_mode={'A03-baseline':'baseline','A05-merge':'merge','A07-replay':'replay','A09-pipeline-probe':'replay'}[args.experiment]
    if args.mode!=expected_mode:raise ValueError('Experiment/mode mismatch')
    if any(e['experiment']==args.experiment for e in d['attempts']):raise RuntimeError('Already reserved, inspect existing app, never duplicate')
    if any(e['state']!='terminal' for e in d['attempts']):raise RuntimeError('Prior attempt not terminal')
    if args.mode=='merge' and not any(e['experiment']=='A03-baseline' and e.get('correctness_and_graph_passed') for e in d['attempts']):
        raise RuntimeError('Establish corrected baseline before new algorithm')
    if (datetime.now(timezone.utc)-datetime.fromisoformat(d['billing_ui_verified_utc'])).total_seconds()>600:
        raise RuntimeError('Fresh Mac UI credit/cap verification needed')
    if d['verified_net_spend_cap_usd']!=0 or d['verified_credit_balance_usd']<2:raise RuntimeError('Credits/cap gate')
    if datetime.now(timezone.utc)>=datetime(2026,10,25,6,26,47,tzinfo=timezone.utc):raise RuntimeError('Existing 12h auth expired, do not renew')
    billing=read_cli('billing','summary','--json');apps=read_cli('app','list','--json')
    if Decimal(billing['billed_cost'])!=0 or any(a['state']!='stopped' or int(a['tasks']) for a in apps):raise RuntimeError('Billing/active app gate')
    observed=max(Decimal(billing['metered_cost']),sum((Decimal(v) for v in billing['metered_cost_breakdown'].values()),Decimal(0)))
    d['observed_gross_high_water_usd']=str(max(Decimal(d['observed_gross_high_water_usd']),observed))
    guard=max(Decimal(d['conservative_stage_spent_usd']),Decimal(d['observed_gross_high_water_usd']))
    reserve=Decimal('.25') if args.mode=='replay' else Decimal('.50')
    timeout,startup,deadline=(90,45,70) if args.mode=='replay' else (240,60,210)
    if guard+reserve>2:raise RuntimeError('Remaining conservative budget cannot reserve this run')
    path=args.bundle.resolve()
    if (ROOT/'local-notes').resolve() not in path.parents:raise ValueError('Use explicit ignored bundle')
    manifest=json.loads((path/'bundle-manifest.json').read_text())
    expected_count=(len(set(FIXED)|set(LEGAL)|{manifest['probe']['parent'],manifest['probe']['candidate'],'replay_probe.py'}) if args.mode=='replay' else (12 if args.mode=='baseline' else 13))
    if manifest['mode']!=args.mode or len(manifest['sha256'])!=expected_count:raise ValueError('Wrong reviewed file count/mode')
    if {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}!=set(manifest['sha256'])|{'bundle-manifest.json'}:raise ValueError('Extra upload files')
    for name,sha in manifest['sha256'].items():
        if (path/name).is_symlink() or hashlib.sha256((path/name).read_bytes()).hexdigest()!=sha:raise ValueError('Bundle hash changed')
    if (path/'modal_worker.py').read_bytes()!=(HERE/'modal_worker.py').read_bytes():raise ValueError('Worker bundle differs')
    import modal
    spec=importlib.util.spec_from_file_location('attention_modal_worker',HERE/('replay_probe.py' if args.mode=='replay' else 'modal_worker.py'));worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    e=dict(experiment=args.experiment,mode=args.mode,state='reserved',app_id=None,created_utc=now(),
        source_commit=manifest['source_commit'],submission_sha256=manifest['sha256']['submission.py'],
        manifest=manifest,billing_before=billing,conservative_reserve_usd=str(reserve),
        resources=dict(gpu='H200',cpu=[2,2],memory_mib=[8192,8192],max_containers=1,min_containers=0,
            buffer_containers=0,scaledown_window=2,timeout=timeout,startup_timeout=startup,internal_deadline=deadline,retries=0))
    d['attempts'].append(e);save(d);started=time.monotonic()
    image=modal.Image.from_id(d['reused_image_id']).add_local_dir(path,'/opt/attention-bundle/reviews/attention-v2',copy=False)
    app=modal.App('okc-attention-'+args.mode,include_source=False)
    fn=app.function(image=image,gpu='H200',cpu=(2,2),memory=(8192,8192),max_containers=1,
        min_containers=0,buffer_containers=0,scaledown_window=2,timeout=timeout,startup_timeout=startup,
        retries=0,serialized=True,include_source=False,block_network=True,restrict_modal_access=True)(worker.make_entry())
    try:
        with app.run(detach=False,environment_name='main'):
            e.update(state='accepted',app_id=app.app_id,accepted_utc=now());save(d)
            print('Accepted',args.experiment,app.app_id,flush=True)
            result=fn.remote(args.mode,deadline)
            (ROOT/f'local-notes/attention-{args.experiment}-result.local.json').write_text(json.dumps(result,indent=2)+'\n')
            e.update(state='function-terminal',exit_code=result['exit_code'],worker_wall_seconds=result['worker_wall_seconds'])
    except Exception as error:
        e.update(state='needs-status-reconciliation',app_id=app.app_id,error_type=type(error).__name__);raise
    finally:
        e.update(client_finished_utc=now(),client_wall_seconds=time.monotonic()-started);save(d)
        print('Client exited. Verify exact app stopped and costs before next run.',flush=True)


if __name__=='__main__':main()
