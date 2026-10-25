"""Explicit attention-cycle ledgers; reserve final, never retry an uncertain POST."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
CLI=PROJECT/'bin/okc'
LEDGER=HERE/'attention-cycle-budget.json'
GPU_SLOT=PROJECT/'local-notes/gpu-slot.local.json'
TERMINAL={'done','failed','rejected','duplicate'}


def now():return datetime.now(timezone.utc).isoformat()
def save(d):
    maximum=d['maximum_new_checks']
    if maximum is None and d.get('user_check_ceiling_removed') is not True:
        raise RuntimeError('Missing explicit uncapped free-check authorization')
    d['remaining']=None if maximum is None else maximum-len(d['requests'])
    d['all_requests_terminal']=all(e['state']=='terminal' for e in d['requests'])
    p=LEDGER.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(LEDGER)
def gpu_slot(state,e):
    current=json.loads(GPU_SLOT.read_text())
    if current['state'] not in ['idle','terminal'] and (
            current.get('problem')!='muse_attention_decode' or current.get('experiment')!=e['experiment']):
        raise RuntimeError('Cannot overwrite another active GPU experiment')
    value=dict(state=state,problem='muse_attention_decode',experiment=e['experiment'],
        job_id=e.get('job_id'),source_sha256=e['source_sha256'],updated_utc=now())
    p=GPU_SLOT.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(GPU_SLOT)
def cli(*args):
    p=subprocess.run([str(CLI),*args],cwd=PROJECT,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=120)
    try:r=json.loads(p.stdout)
    except ValueError:r=None
    return p,r
def quota():
    p,r=cli('problems','--json')
    if p.returncode or not r:raise RuntimeError('Read-only quota failed')
    return next(x['limits'] for x in r['problems'] if x['name']=='muse_attention_decode')
def sanitized(value):
    if isinstance(value,dict):return {k:('[redacted]' if any(s in k.lower() for s in ['token','authorization','cookie','email']) else sanitized(v)) for k,v in value.items()}
    if isinstance(value,list):return [sanitized(x) for x in value]
    if isinstance(value,str):
        value=value.replace(str(PROJECT),'<project>')
        return re.sub(r'\b(?:ak|as)-[A-Za-z0-9]{8,}','[redacted identifier]',value)
    return value


def main():
    global LEDGER
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['send','poll','quota'])
    parser.add_argument('experiment',nargs='?')
    parser.add_argument('--cycle2',action='store_true',help='Use the separately authorized ten-request ledger')
    args=parser.parse_args()
    if args.cycle2:LEDGER=HERE/'attention-cycle2-budget.json'
    d=json.loads(LEDGER.read_text())
    if args.action=='quota':
        d.update(last_platform_quota=quota(),quota_observed_utc=now());save(d);print(d['last_platform_quota']);return
    if args.action=='send':
        if not GPU_SLOT.exists() or json.loads(GPU_SLOT.read_text())['state'] not in ['idle','terminal']:
            raise RuntimeError('Coordinator GPU slot not free; no competing GPU measurement')
        if any(e['experiment']==args.experiment for e in d['requests']):raise RuntimeError('Already reserved; inspect existing job, never duplicate')
        if any(e['state'] in ['sending','uncertain'] for e in d['requests']):raise RuntimeError('Uncertain request reserved, reconcile before POST')
        maximum=d['maximum_new_checks']
        if maximum is None and d.get('user_check_ceiling_removed') is not True:
            raise RuntimeError('Missing explicit uncapped free-check authorization')
        if maximum is not None and len(d['requests'])>=maximum:raise RuntimeError('This cycle authorization exhausted')
        plan=d['plans'][args.experiment]
        if maximum is not None and maximum-len(d['requests'])<=d['reserved_final_checks'] and not plan['final']:raise RuntimeError('Last check reserved for final complete candidate')
        if sum(e.get('status') not in TERMINAL for e in d['requests'])>=d['max_active_jobs']:raise RuntimeError('Active job limit')
        path=(HERE/plan['candidate']).resolve()
        if path.parent!=HERE or path.is_symlink():raise ValueError('Only intended standalone attention source')
        raw=path.read_bytes();relative=path.relative_to(PROJECT).as_posix()
        commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=PROJECT,text=True).strip()
        if raw!=subprocess.check_output(['git','show',commit+':'+relative],cwd=PROJECT):raise ValueError('Commit exact source before check')
        sha=hashlib.sha256(raw).hexdigest()
        if sha!=plan['expected_sha256'] or not raw.startswith(b'# okc: problem=muse_attention_decode\n'):raise ValueError('Wrong exact planned source')
        compile(raw,str(path),'exec')
        rules=subprocess.run(['curl','-fsSL','https://openkernelchallenge.dev/api/v1/rules'],capture_output=True,text=True,timeout=30)
        if rules.returncode or json.loads(rules.stdout)['version']!=d['rules_version']:raise RuntimeError('Rules changed/unavailable; no agreement or upload')
        before=quota()
        if before['checks']['left']<1:raise RuntimeError('Daily attention check quota exhausted')
        e=dict(experiment=args.experiment,mode='check',state='sending',status='unknown',job_id=None,
            candidate=relative,source_commit=commit,source_sha256=sha,hypothesis=plan['hypothesis'],final=plan['final'],
            created_utc=now(),quota_before=before,command=['bin/okc','check',relative,'--no-wait','--json'])
        d['requests'].append(e);save(d);gpu_slot('sending',e)
        try:p,r=cli('check',relative,'--no-wait','--json')
        except subprocess.TimeoutExpired:
            e.update(state='uncertain',reason='POST CLI timed out; inspect own jobs before any retry');save(d);gpu_slot('uncertain',e);raise
        (HERE/(args.experiment+'-accept.local.json')).write_text(p.stdout)
        (HERE/(args.experiment+'-accept.local.txt')).write_text(p.stderr)
        job=r.get('job_id') if isinstance(r,dict) else None
        e.update(cli_exit_code=p.returncode)
        if job:e.update(state='accepted',status='accepted',job_id=job,accepted_utc=now())
        else:e.update(state='uncertain',reason='No accepted ID; inspect own jobs, reservation remains counted')
        save(d);gpu_slot(e['state'],e);print(json.dumps(e));return
    e=next(e for e in d['requests'] if e['experiment']==args.experiment)
    if not e.get('job_id'):raise RuntimeError('No accepted ID; reconcile first')
    p,r=cli('status',e['job_id'],'--json')
    if not isinstance(r,dict):raise RuntimeError('Read-only status failed, reservation unchanged')
    if r.get('job_id')==e['job_id'] and r.get('status') in ['received','queued','running'] and 'result' not in r:
        e.update(status=r['status'],last_checked_utc=now());save(d);gpu_slot(e['state'],e);print(json.dumps(e));return
    if r.get('id')!=e['job_id'] or r.get('mode')!='check':raise RuntimeError('Job identity/mode mismatch')
    (HERE/(args.experiment+'-job.local.json')).write_text(json.dumps(r,indent=2)+'\n')
    (HERE/(args.experiment+'-official-result.json')).write_text(json.dumps(sanitized(r),indent=2)+'\n')
    e.update(status=r['status'],last_checked_utc=now())
    if r['status'] in TERMINAL:
        result=r.get('result') or {};e.update(state='terminal',passed=result.get('passed'),speedup=result.get('speedup'),
            speedup_ci=result.get('speedup_ci'),quota_after=quota())
    save(d)
    gpu_slot(e['state'],e)
    print(json.dumps(e))


if __name__=='__main__':main()
