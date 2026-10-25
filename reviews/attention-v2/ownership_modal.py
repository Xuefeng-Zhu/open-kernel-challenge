"""One approved, bounded exact-source H200 correctness diagnostic; never OKC."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER = HERE / 'A13-ownership-modal.json'
BASE = Decimal('1.708917914')
RESERVE = Decimal('.25')
RATE = Decimal('.001261') + 2 * Decimal('.0000131') + 8 * Decimal('.00000222')
SOURCE = 'mixed_fa3_decode_parallel_merge.py'
SHA = 'c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0'
FILES = [SOURCE, 'native_ownership_probe.py', 'modal_worker.py',
         'official/reference.py', 'official/checks.py', 'official/baseline.py',
         'official/TASK.md', 'official/AGENTS.md']
LEGAL = {'legal/LICENSE': 'LICENSE', 'legal/THIRD_PARTY_NOTICES.md': 'THIRD_PARTY_NOTICES.md',
         'legal/June-9-Researcher-Reciprocity.txt': 'licenses/June-9-Researcher-Reciprocity.txt'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def configure():
    auth = ROOT / 'local-notes/modal-auth.toml'
    assert auth.is_file() and not auth.is_symlink()
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600
    os.environ['MODAL_CONFIG_PATH'] = str(auth)


def cli(*args):
    p = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'modal', *args],
                       capture_output=True, text=True, timeout=25)
    if p.returncode:
        # Do not print raw authentication errors or credential identifiers.
        raise RuntimeError('Supported Modal operation failed: ' + ' '.join(args[:2]))
    return json.loads(p.stdout)


def cost_bound(seconds):
    return Decimal(math.ceil(seconds + 5)) * RATE * Decimal('1.1') + Decimal('.01')


def make_entry():
    # Serialize only this small, self-contained function, never local config/state.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path('/opt/ownership-bundle')
        out = Path('/tmp/ownership-report.json')
        start = time.monotonic()
        manifest = json.loads((root/'bundle-manifest.json').read_text())
        for name, sha in manifest['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != sha:
                return dict(exit_code=2, error='Bundle hash mismatch', worker_wall_seconds=0)
        with Path('/tmp/ownership-worker.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(root/'native_ownership_probe.py'),
                '--worker', '--bounded-modal', '--candidate', str(root/manifest['candidate']),
                '--expected-sha', manifest['candidate_sha256'], '--output', str(out),
                '--rounds', '1', '--deadline', '75'], cwd=root,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=83)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code = 124
        report = json.loads(out.read_text()) if out.exists() else None
        value = dict(exit_code=code, worker_wall_seconds=time.monotonic()-start,
                     report=report)
        # Keep payload below blob threshold; preserve numerical/coverage rows,
        # no PTX, traces, tensors or captured stdout in the remote result.
        encoded = json.dumps(value)
        if 16384 < len(encoded.encode()) <= 262144:
            packed = base64.b64encode(zlib.compress(encoded.encode(),9)).decode()
            if len(packed) < 22000:
                return dict(transport='zlib-json',data=packed,
                    uncompressed_sha256=hashlib.sha256(encoded.encode()).hexdigest())
        if len(encoded.encode()) > 24576:
            value = dict(exit_code=code, worker_wall_seconds=time.monotonic()-start,
                report=None, report_status=str((report or {}).get('status','unknown'))[:100],
                report_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                completed_records=len((report or {}).get('records', [])),
                retrieval_limited=True, error='Report exceeded 24 KiB; no false coverage pass')
        assert len(json.dumps(value).encode()) <= 24576
        return value
    return entry


def build():
    if LEDGER.exists():
        raise RuntimeError('Existing attempt/plan; inspect, do not overwrite')
    files = {n: HERE/n for n in FILES}
    files.update({n: ROOT/src for n, src in LEGAL.items()})
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen={}
    for name, p in files.items():
        raw=p.read_bytes()
        if p.is_symlink() or raw != subprocess.check_output(
                ['git', 'show', head+':'+p.relative_to(ROOT).as_posix()], cwd=ROOT):
            raise ValueError('Commit exact reviewed file before bundling: '+name)
        frozen[name]=raw
    assert hashlib.sha256(frozen[SOURCE]).hexdigest() == SHA
    destination = ROOT/'local-notes/A13-ownership-bundle'
    destination.mkdir(exist_ok=False)
    for name,raw in frozen.items():
        dest=destination/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
    manifest=dict(candidate=SOURCE,candidate_sha256=SHA,
        source_commit=head,
        sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in frozen.items()})
    write(destination/'bundle-manifest.json',manifest)
    write(LEDGER,dict(experiment='A13-ownership',state='prepared',created_utc=now(),
        kind='One exact-source correctness/overlap diagnostic, no performance score',
        prior_conservative_cumulative_usd=str(BASE),incremental_free_credit_ceiling_usd=str(RESERVE),
        paid_ceiling_usd='0',manifest=manifest,app_id=None))
    print(json.dumps(dict(prepared_files=len(files),candidate_sha256=SHA)))


def run():
    d=json.loads(LEDGER.read_text())
    if d['state']!='prepared' or d['app_id']:
        raise RuntimeError('Already reserved/accepted; reconcile, never relaunch')
    pre=json.loads((ROOT/'local-notes/ownership-modal-preflight.local.json').read_text())
    assert pre['workspace']=='xuefeng-zhu'
    assert Decimal(pre['verified_credit_balance_usd']) >= RESERVE
    assert Decimal(pre['verified_net_spend_cap_usd']) == 0
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(pre['billing_ui_verified_utc'])).total_seconds()<900
    if (datetime(2026,10,25,6,26,47,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds()<180:
        raise RuntimeError('Insufficient existing credential lifetime for bounded run/cleanup; do not renew')
    billing=cli('billing','summary','--json')
    apps=cli('app','list','--env','main','--json')
    assert Decimal(billing['billed_cost'])==0
    assert all(a['state']=='stopped' and int(a['tasks'])==0 for a in apps)
    slot=json.loads((ROOT/'local-notes/gpu-slot.local.json').read_text())
    assert slot['state'] in ['terminal','idle'], 'Other GPU job active'
    assert cost_bound(120+15+2)<RESERVE
    path=ROOT/'local-notes/A13-ownership-bundle'
    manifest=d['manifest']
    assert not (path/'bundle-manifest.json').is_symlink()
    assert json.loads((path/'bundle-manifest.json').read_text())==manifest
    assert {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}==set(manifest['sha256'])|{'bundle-manifest.json'}
    for name,sha in manifest['sha256'].items():
        assert not (path/name).is_symlink()
        assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha
    import modal
    image=modal.Image.from_id('im-cPTbGpNiX3RarKResPgX7h').add_local_dir(path,'/opt/ownership-bundle',copy=False)
    app=modal.App('okc-attention-ownership-A13',include_source=False)
    fn=app.function(image=image,gpu='H200',cpu=(2,2),memory=(8192,8192),
        max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
        timeout=90,startup_timeout=30,retries=0,serialized=True,include_source=False,
        block_network=True,restrict_modal_access=True)(make_entry())
    d.update(state='reserved',reserved_utc=now(),billing_before=billing,
        resources=dict(gpu='H200',cpu=[2,2],memory_mib=[8192,8192],max_containers=1,
            min_containers=0,buffer_containers=0,scaledown_window=2,timeout=90,startup_timeout=30,
            internal_deadline=75,retries=0,client_stop_watchdog_seconds=120),
        conservative_incremental_reservation_usd='.25')
    write(LEDGER,d)
    slot_path=ROOT/'local-notes/gpu-slot.local.json'
    write(slot_path,dict(state='reserved',provider='modal',experiment='A13-ownership',
        app_id=None,updated_utc=now()))
    start=time.monotonic()
    timer=None
    stop_lock=threading.Lock()
    stop_outcome={}
    def stop_watchdog():
        with stop_lock:
            if not app.app_id or stop_outcome:
                return
            stop_outcome.update(requested_utc=now(),app_id=app.app_id)
            try:
                stopped=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','modal','app','stop',
                    app.app_id,'--yes','--env','main'],capture_output=True,text=True,timeout=15)
                stop_outcome['returncode']=stopped.returncode
            except Exception as error:
                stop_outcome['error_type']=type(error).__name__
            finally:
                print('Bounded app stop watchdog fired; inspect exact app.',flush=True)
    try:
        with app.run(detach=False,environment_name='main'):
            d.update(state='accepted',app_id=app.app_id,accepted_utc=now())
            write(LEDGER,d)
            write(slot_path,dict(state='accepted',provider='modal',experiment='A13-ownership',
                app_id=app.app_id,updated_utc=now()))
            print('Accepted A13',app.app_id,flush=True)
            timer=threading.Timer(120,stop_watchdog);timer.daemon=True;timer.start()
            result=fn.remote()
            if result.get('transport')=='zlib-json':
                import base64,zlib
                decoder=zlib.decompressobj()
                raw=decoder.decompress(base64.b64decode(result['data']),262145)
                assert len(raw)<=262144 and not decoder.unconsumed_tail and decoder.eof
                assert hashlib.sha256(raw).hexdigest()==result['uncompressed_sha256']
                result=json.loads(raw)
            encoded=json.dumps(result,indent=2)+'\n'
            if re.search(r'\b(?:ak|as)-[A-Za-z0-9]{8,}',encoded):
                raise RuntimeError('Unexpected credential identifier in result; do not persist')
            (HERE/'A13-ownership-result.json').write_text(encoded)
            d.update(state='function-terminal',exit_code=result['exit_code'])
    except Exception as error:
        d.update(state='needs-status-reconciliation',app_id=app.app_id,error_type=type(error).__name__)
        # Exact stage/type is sufficient; avoid unsanitized SDK credential details.
        print('A13 operation failed; inspect saved app:',type(error).__name__,flush=True)
    finally:
        try:
            states=cli('app','list','--env','main','--json')
            exact=next((a for a in states if a['app_id']==app.app_id),None)
            if app.app_id and (not exact or exact['state']!='stopped' or int(exact['tasks'])):
                stop_watchdog()
                states=cli('app','list','--env','main','--json')
                exact=next((a for a in states if a['app_id']==app.app_id),None)
            d['shutdown_observation']=exact
            d['shutdown_verified']=bool(exact and exact['state']=='stopped' and int(exact['tasks'])==0)
        except Exception as error:
            d.update(shutdown_verified=False,shutdown_error_type=type(error).__name__)
            stop_watchdog()
        if timer:timer.cancel()
        d['explicit_stop_outcome']=stop_outcome
        d.update(client_finished_utc=now(),client_wall_seconds=time.monotonic()-start)
        write(LEDGER,d)
        write(slot_path,dict(state='terminal' if d.get('shutdown_verified') else 'uncertain',
            provider='modal',experiment='A13-ownership',app_id=app.app_id,updated_utc=now()))
        print('A13 client returned; verify exact app stopped and billing.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','run'])
    args=p.parse_args()
    if args.action=='build':build()
    else:
        configure()
        run()


if __name__=='__main__':main()
