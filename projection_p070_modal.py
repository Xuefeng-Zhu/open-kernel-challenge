"""Bounded P070 execution using existing auth; no billing or credential actions."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import zlib

import projection_p070_quack_preflight as p
from projection_p044_modal import cli

ROOT = Path(__file__).resolve().parent
CONTROL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
PAYLOAD = tuple(p.FROZEN) + ('LICENSE','THIRD_PARTY_NOTICES.md',
    'licenses/June-9-Researcher-Reciprocity.txt','licenses/Triton-MIT.txt','projection_p070_quack_preflight.py')


def remote_entry(stage, prior):
    import base64, hashlib, json, os, shutil, subprocess, sys, traceback, zlib
    from pathlib import Path
    root = Path('/opt/p070'); out = Path('/tmp/p070-compile')
    sys.path.insert(0, str(root))
    import projection_p070_quack_preflight as worker
    result = {'stage': stage, 'success': False}
    files = {}
    try:
        if stage == 'compile':
            cmd = [sys.executable, '-B', str(root / 'projection_p070_quack_preflight.py'),
                   'compile', '--quack-root', '/opt/p070-quack', '--output', str(out)]
        else:
            out.mkdir()
            for name, raw in prior.items():
                if name != 'receipt.json' and name not in {arm+'.'+ext for arm in ('serial','separate','separate_reduce') for ext in ('o','cubin','ptx','mlir')}:
                    raise ValueError('Unexpected prior artifact')
                (out / name).write_bytes(raw)
            cmd = [sys.executable, '-B', str(root / 'projection_p070_quack_preflight.py'),
                   'run', '--quack-root', '/opt/p070-quack', '--compile-review', str(out/'receipt.json'),
                   '--output', str(out/'native.json')]
        child = subprocess.run(cmd, cwd=root, capture_output=True, timeout=210)
        files['worker.stdout.txt'] = child.stdout
        files['worker.stderr.txt'] = child.stderr
        result['exit_code'] = child.returncode
        if stage == 'compile' and child.returncode == 0:
            tool = shutil.which('cuobjdump')
            if not tool:
                candidate=Path('/usr/local/cuda/bin/cuobjdump')
                tool=str(candidate) if candidate.is_file() else None
            result['cuobjdump'] = tool
            if tool:
                for name in ('serial', 'separate', 'separate_reduce'):
                    for flag, suffix in (('--dump-resource-usage','resources'),('--dump-sass','sass'),('--dump-ptx','ptx')):
                        r = subprocess.run([tool,flag,str(out/(name+'.cubin'))],capture_output=True,timeout=20)
                        files[name+'.'+suffix+'.txt'] = r.stdout + r.stderr
            result['success'] = bool(tool) and all(b'REG:' in files.get(name+'.resources.txt',b'') for name in ('serial','separate','separate_reduce'))
        else:
            result['success'] = child.returncode == 0
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)[:500]
        files['traceback.txt'] = traceback.format_exc().encode()
    if out.exists():
        for name in ('receipt.json','native.json',*(arm+'.'+ext for arm in ('serial','separate','separate_reduce') for ext in ('o','cubin','ptx','mlir'))):
            if (out/name).is_file(): files[name]=(out/name).read_bytes()
    files['result.json']=json.dumps(result).encode()
    raw=json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode()
    packed=zlib.compress(raw,9)
    if len(packed)>2*1024*1024:
        raise RuntimeError('Artifact transport bound exceeded')
    yield {'kind':'manifest','sha256':hashlib.sha256(packed).hexdigest(),'bytes':len(packed)}
    for i in range(0,len(packed),16384):yield {'kind':'chunk','data':packed[i:i+16384]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('compile','native'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--prior',type=Path)
    args=ap.parse_args();assert not args.output.exists()
    p.source_hashes()
    auth=CONTROL/'local-notes/modal-auth.toml'
    assert auth.is_file() and not auth.is_symlink() and auth.stat().st_mode & 0o077 == 0
    assert subprocess.run(['git','check-ignore','-q',str(auth)],cwd=CONTROL).returncode==0
    assert not subprocess.check_output(['git','ls-files',str(auth)],cwd=CONTROL)
    os.environ.update(MODAL_CONFIG_PATH=str(auth),MODAL_PROFILE='xuefeng-zhu')
    apps=cli(CONTROL,'app','list','--env','main','--json')
    assert all(a.get('state')=='stopped' and int(a.get('tasks',0))==0 for a in apps),'Active app; do not overlap'
    prior={}
    if args.stage=='native':
        assert args.prior
        prior={n:(args.prior/n).read_bytes() for n in ('receipt.json',*(arm+'.'+ext for arm in ('serial','separate','separate_reduce') for ext in ('o','cubin','ptx','mlir')))}
        receipt=json.loads(prior['receipt.json'])
        assert receipt['quack_commit']==p.PIN and receipt['source_hashes']==p.source_hashes()
        assert receipt.get('review',{}).get('ready_for_execution') is True
        assert all(hashlib.sha256(prior[n]).hexdigest()==digest for n,digest in receipt['objects'].items())
    import warnings
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore',message=r'Your Modal token .* expires',category=ServerWarning)
    import modal
    image=modal.Image.from_id(IMAGE).run_commands(
        'git clone --filter=blob:none --no-checkout https://github.com/Dao-AILab/quack.git /opt/p070-quack',
        'git -C /opt/p070-quack checkout --detach '+p.PIN)
    for name in PAYLOAD:image=image.add_local_file(ROOT/name,'/opt/p070/'+name,copy=False)
    app=modal.App('okc-p070-'+args.stage,include_source=False)
    options=dict(image=image,cpu=(4,4),memory=(8192,8192),min_containers=0,max_containers=1,
        buffer_containers=0,scaledown_window=2,timeout=300,startup_timeout=60,
        retries=None,serialized=True,include_source=False,block_network=True,restrict_modal_access=True)
    if args.stage=='native':options['gpu']='H200'
    fn=app.function(**options)(remote_entry)
    args.output.mkdir(parents=True)
    record={'stage':args.stage,'app_id':None,'source_sha256':hashlib.sha256((ROOT/PAYLOAD[-1]).read_bytes()).hexdigest(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'payload_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in PAYLOAD},
        'hard_seconds':300,'paid_spending_limit_usd':0,'billing_inspected':False,
        'conservative_free_credit_reserve_usd':.70 if args.stage=='native' else .15,'success':False}
    def save(): (args.output/'controller.json').write_text(json.dumps(record,indent=2)+'\n')
    lock=threading.Lock();stop_done=False
    def stop():
        nonlocal stop_done
        with lock:
            if app.app_id and not stop_done:
                cli(CONTROL,'app','stop',app.app_id,'--yes','--env','main');stop_done=True
    expired=threading.Event()
    def watchdog():
        expired.set()
        stop()
    timer=threading.Timer(420,watchdog);timer.daemon=True
    try:
        timer.start()
        with app.run(detach=False,environment_name='main'):
            record['app_id']=app.app_id;save();print('Accepted '+args.stage+' app '+app.app_id,flush=True)
            assert not expired.is_set(),'Deadline elapsed before invocation'
            chunks=[];manifest=None
            for item in fn.remote_gen(args.stage,prior):
                if item['kind']=='manifest':manifest=item
                else:chunks.append(item['data'])
            packed=b''.join(chunks)
            assert manifest and len(packed)==manifest['bytes'] and hashlib.sha256(packed).hexdigest()==manifest['sha256']
            for name,value in json.loads(zlib.decompress(packed)).items():
                assert '/' not in name and '..' not in name
                (args.output/name).write_bytes(base64.b64decode(value))
            record['success']=json.loads((args.output/'result.json').read_text())['success']
            stop()
    except Exception as error:record['error']=type(error).__name__+': '+str(error)[:300]
    finally:
        try:stop()
        except Exception as error:record['stop_error']=type(error).__name__
        timer.cancel()
        try:
            for _ in range(3):
                apps=cli(CONTROL,'app','list','--env','main','--json')
                exact=next((a for a in apps if a.get('app_id')==app.app_id),None)
                record['shutdown']={k:exact.get(k) for k in ('app_id','state','tasks')} if exact else None
                if exact and exact.get('state')=='stopped' and int(exact.get('tasks',0))==0:break
                time.sleep(2)
        except Exception as error:record['shutdown_error']=type(error).__name__
        save();print(json.dumps(record),flush=True)
    return 0 if record['success'] else 1


if __name__=='__main__':raise SystemExit(main())
