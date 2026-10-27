"""One bounded representative correctness/replay/paired-timing Modal call.

Only payload() files are uploaded. Existing protected credentials are used in
place through MODAL_CONFIG_PATH, never copied, read, renewed or uploaded.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import warnings

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ATTENTION = HERE.parents[1]
CONTROL = HERE.parent/'control'
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
SOURCE_SHA256 = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
RATE = (Decimal('4.54')+2*Decimal('.0473')+8*Decimal('.008'))/3600
MAX_RESULT = 200_000


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def reserve(seconds=200, rate=RATE):
    return Decimal(math.ceil(seconds+5))*rate*Decimal('1.1')+Decimal('.01')


def resource_rate(rates):
    return (Decimal(rates['gpu_hour_cost_h200'])+2*Decimal(rates['cpu_hour_cost'])
        +8*Decimal(rates['mem_gib_hour_cost']))/3600


def billed_gross(billing):
    return max(Decimal(billing['metered_cost']),
        sum((Decimal(v) for v in billing['metered_cost_breakdown'].values()), Decimal(0)))


def payload():
    return {'worker.py': HERE/'worker.py',
        'candidatecustomhelper.py': HERE/'candidatecustomhelper.py',
        'candidate.py': HERE.parent/'probe/candidate.py',
        'modal_worker.py': ATTENTION/'modal_worker.py',
        'official/reference.py': ATTENTION/'official/reference.py',
        'official/checks.py': ATTENTION/'official/checks.py',
        'legal/LICENSE': REPO/'LICENSE',
        'legal/THIRD_PARTY_NOTICES.md': REPO/'THIRD_PARTY_NOTICES.md'}


def frozen_payload(head):
    assert re.fullmatch(r'[0-9a-f]{40}', head), 'Exact reviewed source commit required'
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert actual == head, 'HEAD differs from reviewed freeze'
    paths = payload()
    hashes = {}
    for name, path in [*paths.items(), ('controller.py', Path(__file__).resolve())]:
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        committed = subprocess.check_output(['git', 'show', head+':'+path.relative_to(REPO).as_posix()], cwd=REPO)
        assert raw == committed, 'Reviewed bytes changed: '+name
        hashes[name] = hashlib.sha256(raw).hexdigest()
    assert hashes['candidate.py'] == SOURCE_SHA256
    return paths, hashes


def credential_environment():
    assert AUTH.is_file() and not AUTH.is_symlink()
    assert AUTH.stat().st_uid == os.getuid() and AUTH.stat().st_mode & 0o777 == 0o600
    assert not any(os.environ.get(k) for k in ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
        'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET'])
    os.environ['MODAL_CONFIG_PATH'] = str(AUTH)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'


def cli(*args):
    # CLI expiry warnings contain a credential identifier: capture all output
    # and emit only generic failures, never raw authentication diagnostics.
    code = ('import warnings; from modal.exception import ServerWarning; '
        'warnings.filterwarnings("ignore",message=r"Your Modal token .* expires",category=ServerWarning); '
        'from modal.__main__ import main; main()')
    child = subprocess.run([str(PYTHON), '-c', code, *args],
                           capture_output=True, text=True, timeout=20)
    if child.returncode:
        raise RuntimeError('Supported Modal operation failed: '+' '.join(args[:2]))
    return json.loads(child.stdout) if '--json' in args else {'exit_code': child.returncode}


def make_remote_entry():
    # No local configuration/state/credential closure is serialized.
    def entry():
        import hashlib, os, signal, subprocess, sys, time
        from pathlib import Path
        root = Path('/opt/architecture-paired')
        output = Path('/tmp/architecture-paired-result.json')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'worker.py'),
            '--output', str(output), '--deadline', '110'], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            code = 124
        if not output.is_file() or output.is_symlink():
            return dict(exit_code=code, error='Worker result absent',
                worker_wall_seconds=time.monotonic()-started)
        if output.stat().st_size > 200_000:
            return dict(exit_code=code, error='Worker result exceeds 200KB limit',
                worker_wall_seconds=time.monotonic()-started)
        raw = output.read_bytes()
        return dict(exit_code=code, report_json=raw.decode('utf-8'),
            report_sha256=hashlib.sha256(raw).hexdigest(),
            worker_wall_seconds=time.monotonic()-started)
    return entry


def verified_result(reply):
    assert isinstance(reply, dict) and not reply.get('error'), 'Worker result unavailable'
    assert isinstance(reply.get('exit_code'), int)
    raw = reply['report_json'].encode('utf-8')
    assert len(raw) <= MAX_RESULT
    assert hashlib.sha256(raw).hexdigest() == reply['report_sha256']
    report = json.loads(raw)
    assert isinstance(report, dict)
    return raw, report


def launch(head):
    ledger = HERE/'paired-run.local.json'
    assert not ledger.exists(), 'Existing request record: reconcile; never duplicate'
    assert not (HERE/'result.json').exists(), 'Existing result: reconcile; never duplicate'
    paths, hashes = frozen_payload(head)
    credential_environment()
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    assert Path(modal.config.user_config_path).resolve() == AUTH.resolve()
    subprocess.run(['git', 'check-ignore', '-q', 'local-notes/modal-auth.toml'], cwd=ORIGINAL, check=True)
    assert not subprocess.check_output(['git', 'ls-files', 'local-notes/modal-auth.toml'], cwd=ORIGINAL)
    login = json.loads((ORIGINAL/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(login['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    assert login['state'] == 'verified' and (expiry-datetime.now(timezone.utc)).total_seconds() > 260
    pre_path = CONTROL/'billing-ui.json'
    assert pre_path.is_file() and not pre_path.is_symlink()
    pre = json.loads(pre_path.read_text())
    assert pre['workspace'] == 'xuefeng-zhu' and Decimal(pre['verified_net_spend_cap_usd']) == 0
    assert Decimal(pre['total_spend_usd']) == 0 and not pre['changes_made']
    observed = datetime.fromisoformat(pre['observed_utc'].replace('Z', '+00:00'))
    assert -5 <= (datetime.now(timezone.utc)-observed).total_seconds() < 900
    prior_budget = json.loads((ATTENTION/'modal-budget.json').read_text())
    assert not prior_budget.get('active_reservation'), 'Prior reservation requires reconciliation'
    slot = json.loads((ORIGINAL/'local-notes/gpu-slot.local.json').read_text())
    assert slot['state'] in ['idle', 'terminal'], 'Prior GPU slot requires reconciliation'
    # The compile controller's first shutdown read may have seen stopping. This
    # independent subsequent receipt establishes terminal before this new call.
    compiled = json.loads((CONTROL/'modal-run.local.json').read_text())
    assert compiled.get('compile_status') == 'compiled' and compiled.get('exit_code') == 0
    assert compiled.get('artifact_retrieval_complete') is True
    assert compiled['payload_sha256']['candidate.py'] == SOURCE_SHA256
    shutdown = json.loads((CONTROL/'shutdown-observation.json').read_text())
    exact = next((row for row in shutdown if row['app_id'] == compiled['app_id']), None)
    assert exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0
    billing = cli('billing', 'summary', '--json')
    assert Decimal(billing['billed_cost']) == 0
    apps = cli('app', 'list', '--env', 'main', '--json')
    assert all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in apps)
    rates = cli('billing', 'rates', '--json')
    rate = resource_rate(rates)
    cost_reserve = reserve(rate=rate)
    gross = billed_gross(billing)
    free_floor = Decimal(pre['verified_credit_balance_usd'])-max(Decimal(0), gross-Decimal(pre['total_usage_usd']))
    assert free_floor > cost_reserve, 'Verified free credits cannot cover bounded job'
    image = modal.Image.from_id(IMAGE)
    for name, path in paths.items():
        image = image.add_local_file(path, '/opt/architecture-paired/'+name, copy=False)
    app = modal.App('okc-m64-bf16-paired', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=135, startup_timeout=20, retries=0, serialized=True, include_source=False,
        block_network=True, restrict_modal_access=True)(make_remote_entry())
    record = dict(experiment='M64 BF16 representative correctness/replay/paired timing',
        state='reserved', app_id=None, created_utc=now(), source_commit=head,
        payload_sha256={name: hashes[name] for name in paths},
        controller_sha256=hashes['controller.py'], base_image=IMAGE,
        billing_before=billing, apps_before=apps, cost_reserve_usd=str(cost_reserve),
        resource_rate_usd_per_second=str(rate), relevant_rates={key: rates[key] for key in
            ['gpu_hour_cost_h200', 'cpu_hour_cost', 'mem_gib_hour_cost']},
        rates_verified_utc=now(), conservative_free_credit_floor_usd=str(free_floor),
        credit_cap_evidence=dict(verified_utc=pre['observed_utc'],
            credit_balance_usd=pre['verified_credit_balance_usd'], net_spend_cap_usd='0',
            receipt_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest()),
        actual_paid_limit_usd='0', artificial_stage_ceiling_removed=True,
        prior_compile_app_id=compiled['app_id'], prior_compile_shutdown_verified=True,
        resources=dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            function_timeout=135, startup_timeout=20, worker_deadline=110,
            child_hard_timeout=120, app_watchdog=170, retry_policy=0, explicit_invocations=1),
        official_check=False, formal_entry=False)
    with ledger.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    started = time.monotonic()
    stop_lock = threading.Lock()
    stop_result = {}
    expired = threading.Event()

    def stop():
        with stop_lock:
            if not app.app_id or stop_result:
                return
            stop_result.update(requested_utc=now(), app_id=app.app_id)
            try:
                stop_result.update(cli('app', 'stop', app.app_id, '--yes', '--env', 'main'))
            except Exception as exc:
                stop_result['error_type'] = type(exc).__name__

    def watchdog():
        expired.set()
        stop()

    timer = threading.Timer(170, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name='main'):
            record.update(state='accepted', app_id=app.app_id, accepted_utc=now())
            save(ledger, record)
            print('Accepted single bounded paired app '+app.app_id, flush=True)
            if expired.is_set():
                raise TimeoutError('App watchdog expired before invocation')
            reply = fn.remote()
            raw, report = verified_result(reply)
            with (HERE/'result.json').open('xb') as stream:
                stream.write(raw)
            record.update(state='function-terminal', result_retrieved=True,
                exit_code=reply['exit_code'], worker_status=report.get('status'),
                worker_passed=report.get('passed'), report_sha256=reply['report_sha256'],
                worker_wall_seconds=reply['worker_wall_seconds'])
            save(ledger, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('Paired request requires reconciliation; do not relaunch.', flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            states = cli('app', 'list', '--env', 'main', '--json')
            exact = next((row for row in states if row['app_id'] == app.app_id), None)
            record['shutdown_observation'] = exact
            record['shutdown_verified'] = bool(exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0)
            record['apps_after'] = states
            record['all_apps_stopped_zero_tasks'] = all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in states)
            record['billing_after'] = cli('billing', 'summary', '--json')
            record['observed_gross_delta_usd'] = str(max(Decimal(0), billed_gross(record['billing_after'])-gross))
            record['paid_after_usd'] = record['billing_after']['billed_cost']
            assert Decimal(record['paid_after_usd']) == 0
        except Exception as exc:
            record.update(shutdown_verified=False, shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stop_result, client_finished_utc=now(), client_wall_seconds=time.monotonic()-started)
        save(ledger, record)
        print(json.dumps({key: record.get(key) for key in ['app_id', 'state', 'exit_code',
            'worker_status', 'worker_passed', 'shutdown_verified', 'all_apps_stopped_zero_tasks',
            'observed_gross_delta_usd', 'paid_after_usd', 'client_wall_seconds']}), flush=True)
    return 0 if record.get('result_retrieved') and record.get('shutdown_verified') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-head', required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.frozen_head))
