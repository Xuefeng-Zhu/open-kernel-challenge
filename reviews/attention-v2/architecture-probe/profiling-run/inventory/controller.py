"""One CPU-only existing-tool inventory using the existing protected config.

Uploads worker.py only. Never installs, requests a GPU, creates credentials or
alters billing/permissions. Parent must review/freeze the exact source first.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import warnings

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
RESERVE = Decimal('.02')
MAX_RESULT = 128_000


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def cli(*args):
    # Expiry warnings contain a credential identifier; capture and discard raw
    # diagnostics, reporting only a generic operation name on failure.
    code = ('import warnings; from modal.exception import ServerWarning; '
        'warnings.filterwarnings("ignore",message=r"Your Modal token .* expires",category=ServerWarning); '
        'from modal.__main__ import main; main()')
    child = subprocess.run([str(PYTHON), '-c', code, *args],
                           capture_output=True, text=True, timeout=20)
    if child.returncode:
        raise RuntimeError('Supported Modal operation failed: '+' '.join(args[:2]))
    return json.loads(child.stdout) if '--json' in args else {'exit_code': child.returncode}


def gross(billing):
    return max(Decimal(billing['metered_cost']),
        sum((Decimal(v) for v in billing['metered_cost_breakdown'].values()), Decimal(0)))


def configure_existing_auth():
    assert AUTH.is_file() and not AUTH.is_symlink()
    assert AUTH.stat().st_uid == os.getuid() and AUTH.stat().st_mode & 0o777 == 0o600
    assert not any(os.environ.get(k) for k in ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
        'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET'])
    os.environ['MODAL_CONFIG_PATH'] = str(AUTH)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'


def frozen_payload(head):
    assert re.fullmatch(r'[0-9a-f]{40}', head)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
    hashes = {}
    for path in [HERE/'worker.py', Path(__file__).resolve()]:
        assert path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        assert data == subprocess.check_output(['git', 'show', head+':'+path.relative_to(REPO).as_posix()], cwd=REPO)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    return hashes


def make_remote_entry():
    # No local state/config/auth object is closed over or uploaded.
    def entry():
        import hashlib, os, signal, subprocess, sys, time
        from pathlib import Path
        root = Path('/opt/profile-inventory')
        output = Path('/tmp/profile-inventory.json')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'worker.py'), '--output', str(output)],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            code = 124
        if not output.is_file() or output.is_symlink() or output.stat().st_size > 128_000:
            return dict(exit_code=code, error='Inventory result missing or outside size guard',
                        worker_wall_seconds=time.monotonic()-started)
        raw = output.read_bytes()
        return dict(exit_code=code, report_json=raw.decode(),
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                    worker_wall_seconds=time.monotonic()-started)
    return entry


def verify_result(reply):
    assert isinstance(reply, dict) and not reply.get('error')
    raw = reply['report_json'].encode()
    assert len(raw) <= MAX_RESULT and hashlib.sha256(raw).hexdigest() == reply['report_sha256']
    report = json.loads(raw)
    assert report['gpu_requested'] is False and report['cuda_context_requested'] is False
    assert report['candidate_kernel_launches'] == 0
    return raw, report


def launch(head, receipt_path):
    ledger = HERE/'inventory-run.local.json'
    assert not ledger.exists() and not (HERE/'result.json').exists(), 'Existing request: reconcile, never duplicate'
    hashes = frozen_payload(head)
    configure_existing_auth()
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    assert Path(modal.config.user_config_path).resolve() == AUTH.resolve()
    subprocess.run(['git', 'check-ignore', '-q', 'local-notes/modal-auth.toml'], cwd=ORIGINAL, check=True)
    assert not subprocess.check_output(['git', 'ls-files', 'local-notes/modal-auth.toml'], cwd=ORIGINAL)
    login = json.loads((ORIGINAL/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(login['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    assert login['state']=='verified' and (expiry-datetime.now(timezone.utc)).total_seconds() > 150
    assert receipt_path.is_file() and not receipt_path.is_symlink()
    receipt_path = receipt_path.resolve()
    assert receipt_path.parent == (HERE.parent/'control').resolve()
    assert receipt_path.is_file() and not receipt_path.is_symlink()
    pre = json.loads(receipt_path.read_text())
    assert pre['workspace']=='xuefeng-zhu' and Decimal(pre['verified_net_spend_cap_usd'])==0
    assert Decimal(pre['total_spend_usd'])==0 and not pre['changes_made']
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(pre['observed_utc'].replace('Z','+00:00'))).total_seconds()
    assert -5 <= age < 900
    billing = cli('billing','summary','--json')
    assert Decimal(billing['billed_cost'])==0
    apps = cli('app','list','--env','main','--json')
    assert all(a['state']=='stopped' and int(a['tasks'])==0 for a in apps)
    rates = cli('billing','rates','--json')
    rate = (2*Decimal(rates['cpu_hour_cost'])+8*Decimal(rates['mem_gib_hour_cost']))/3600
    # 50s watchdog +20s stop allowance, double model +one cent slack.
    modeled_reserve = 70*rate*2+Decimal('.01')
    assert modeled_reserve <= RESERVE
    floor = Decimal(pre['verified_credit_balance_usd'])-max(Decimal(0),gross(billing)-Decimal(pre['total_usage_usd']))
    assert floor > RESERVE
    image = modal.Image.from_id(IMAGE).add_local_file(HERE/'worker.py','/opt/profile-inventory/worker.py',copy=False)
    app = modal.App('okc-profile-cpu-inventory',include_source=False)
    fn = app.function(image=image,gpu=None,cpu=(2,2),memory=(8192,8192),
        min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=2,
        timeout=30,startup_timeout=15,retries=0,serialized=True,
        include_source=False,block_network=True,restrict_modal_access=True)(make_remote_entry())
    record = dict(experiment='CPU-only existing profiling tools inventory',state='reserved',app_id=None,
        created_utc=now(),source_commit=head,payload_sha256={'worker.py':hashes['worker.py']},
        controller_sha256=hashes['controller.py'],base_image=IMAGE,gpu_requested=False,
        billing_before=billing,apps_before=apps,cost_reserve_usd=str(RESERVE),
        modeled_reserve_usd=str(modeled_reserve),resource_rate_usd_per_second=str(rate),
        verified_free_floor_usd=str(floor),actual_paid_limit_usd='0',
        billing_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        artificial_stage_ceiling_removed=True,resources=dict(gpu=None,cpu=[2,2],memory_mib=[8192,8192],
            max_containers=1,worker_alarm=20,child_hard_timeout=25,function_timeout=30,
            startup_timeout=15,app_watchdog=50,retries=0,explicit_invocations=1))
    with ledger.open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n')
    lock = threading.Lock()
    stopped = {}
    expired = threading.Event()
    started = time.monotonic()

    def stop():
        with lock:
            if stopped or not app.app_id:
                return
            stopped.update(requested_utc=now(),app_id=app.app_id)
            try:
                stopped.update(cli('app','stop',app.app_id,'--yes','--env','main'))
            except Exception as exc:
                stopped['error_type'] = type(exc).__name__

    def watchdog():
        expired.set()
        stop()

    timer = threading.Timer(50,watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False,environment_name='main'):
            record.update(state='accepted',app_id=app.app_id,accepted_utc=now())
            save(ledger,record)
            print('Accepted CPU-only inventory '+app.app_id,flush=True)
            if expired.is_set():
                raise TimeoutError('Inventory watchdog expired before invocation')
            reply = fn.remote()
            raw,report = verify_result(reply)
            assert report['worker_sha256']==hashes['worker.py']
            with (HERE/'result.json').open('xb') as stream:
                stream.write(raw)
            record.update(state='function-terminal',result_retrieved=True,worker_status=report['status'],
                exit_code=reply['exit_code'],report_sha256=reply['report_sha256'])
            save(ledger,record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation',app_id=app.app_id,error_type=type(exc).__name__)
        print('Inventory needs reconciliation; do not relaunch.',flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            observations = []
            for attempt in range(3):
                states = cli('app','list','--env','main','--json')
                exact = next((a for a in states if a['app_id']==app.app_id),None)
                observations.append(dict(observed_utc=now(),app=exact))
                if exact and exact['state']=='stopped' and int(exact['tasks'])==0:
                    break
                if attempt < 2:
                    time.sleep(1)
            record['shutdown_observations'] = observations
            record['shutdown_verified'] = bool(exact and exact['state']=='stopped' and int(exact['tasks'])==0)
            record['apps_after'] = states
            record['all_apps_stopped_zero_tasks'] = all(a['state']=='stopped' and int(a['tasks'])==0 for a in states)
            after = cli('billing','summary','--json')
            record['billing_after'] = after
            record['observed_gross_delta_usd'] = str(max(Decimal(0),gross(after)-gross(billing)))
            assert Decimal(after['billed_cost'])==0
        except Exception as exc:
            record.update(shutdown_verified=False,shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stopped,client_finished_utc=now(),client_wall_seconds=time.monotonic()-started)
        save(ledger,record)
        print(json.dumps({k:record.get(k) for k in ['app_id','state','worker_status','exit_code','shutdown_verified','all_apps_stopped_zero_tasks','observed_gross_delta_usd']}),flush=True)
    return 0 if record.get('result_retrieved') and record.get('shutdown_verified') else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen-head',required=True)
    parser.add_argument('--billing-receipt',required=True,type=Path)
    args=parser.parse_args()
    raise SystemExit(launch(args.frozen_head,args.billing_receipt))
