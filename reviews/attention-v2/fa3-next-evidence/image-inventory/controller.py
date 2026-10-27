"""One CPU-only, read-only image eligibility inventory; default is offline preflight.

Reuses the prior CPU inventory's protected config, bounded child/transport,
exclusive ledger and explicit stop/reconciliation guards. No GPU is requested.
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
REPO = HERE.parents[3]
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
EXPERIMENT = 'attention-fa3-image-inventory'
LICENSE_HASHES = {
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
RESERVE = Decimal('.02')
MAX_RESULT = 256 * 1024
WORKER_SECONDS, CHILD_SECONDS = 30, 35
FUNCTION_SECONDS, STARTUP_SECONDS, WATCHDOG_SECONDS = 45, 10, 60


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def cli(*args):
    # Expiry diagnostics include a credential identifier: capture, never forward.
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
        sum((Decimal(value) for value in billing['metered_cost_breakdown'].values()), Decimal(0)))


def resource_rate(rates):
    # CPU1 and 2 GiB only. Deliberately never read a GPU-price field.
    return (Decimal(rates['cpu_hour_cost'])+2*Decimal(rates['mem_gib_hour_cost']))/3600


def reserve(rate):
    modeled = Decimal(WATCHDOG_SECONDS)*rate*2+Decimal('.005')
    if modeled > RESERVE:
        raise ValueError('Current CPU-only rate exceeds the fixed free-credit reserve')
    return modeled


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Billing observation timezone required')
    return result


def validate_budget(receipt, billing, current=None):
    current = current or datetime.now(timezone.utc)
    if not (receipt['workspace'] == 'xuefeng-zhu'
            and Decimal(receipt['verified_net_spend_cap_usd']) == 0
            and Decimal(receipt['total_spend_usd']) == 0 and not receipt['changes_made']
            and Decimal(billing['billed_cost']) == 0):
        raise ValueError('Existing workspace paid-$0 guardrail not verified')
    if not -5 <= (current-timestamp(receipt['observed_utc'])).total_seconds() < 900:
        raise ValueError('Billing guardrail receipt stale')
    floor = Decimal(receipt['verified_credit_balance_usd'])-max(
        Decimal(0), gross(billing)-Decimal(receipt['total_usage_usd']))
    if floor <= RESERVE:
        raise ValueError('Verified free credit floor cannot cover CPU reserve')
    return floor


def stopped(apps):
    return all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in apps)


def configure_existing_auth():
    if not (AUTH.is_file() and not AUTH.is_symlink() and AUTH.stat().st_uid == os.getuid()
            and AUTH.stat().st_mode & 0o777 == 0o600):
        raise ValueError('Existing protected Modal config required')
    if any(os.environ.get(key) for key in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
            'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET')):
        raise ValueError('Expanded credential environment forbidden')
    os.environ['MODAL_CONFIG_PATH'] = str(AUTH)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'
    receipt = json.loads((ORIGINAL/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(receipt['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    if not (receipt['state'] == 'verified'
            and (expiry-datetime.now(timezone.utc)).total_seconds() > WATCHDOG_SECONDS+60):
        raise ValueError('Existing verified authentication expires too soon; no renewal')


def payload():
    return {path.relative_to(REPO).as_posix(): path for path in
            [HERE/'worker.py', REPO/'LICENSE', REPO/'THIRD_PARTY_NOTICES.md']}


def frozen_payload(head):
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('Exact reviewed commit required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() != head:
        raise ValueError('HEAD differs from reviewed freeze')
    paths, hashes = payload(), {}
    controller_key = Path(__file__).relative_to(REPO).as_posix()
    for name, path in [*paths.items(), (controller_key, Path(__file__))]:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Frozen payload path guard: '+name)
        raw = path.read_bytes()
        if raw != subprocess.check_output(['git', 'show', head+':'+name], cwd=REPO):
            raise ValueError('Committed inventory payload changed: '+name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if path.suffix == '.py':
            compile(raw, name, 'exec')
    if any(hashes[name] != expected for name, expected in LICENSE_HASHES.items()):
        raise ValueError('License payload hash mismatch')
    return paths, hashes


def make_remote_entry():
    # No local config, credential object or receipt is captured or uploaded.
    def entry():
        import hashlib, os, signal, subprocess, sys, time
        from pathlib import Path
        root = Path('/opt/fa3-image-inventory')
        output = Path('/tmp/fa3-image-inventory-result.json')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'worker.py'), '--output', str(output), '--deadline', '30'],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            code = 124
        if not output.is_file() or output.is_symlink() or output.stat().st_size > 256 * 1024:
            return dict(exit_code=code, error='Inventory result missing or outside size guard',
                        worker_wall_seconds=time.monotonic()-started)
        raw = output.read_bytes()
        return dict(exit_code=code, report_json=raw.decode('utf-8'),
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                    worker_wall_seconds=time.monotonic()-started)
    return entry


def verify_result(reply, worker_hash):
    if not (isinstance(reply, dict) and not reply.get('error')
            and type(reply.get('exit_code')) is int and isinstance(reply.get('report_json'), str)):
        raise ValueError('Inventory reply unavailable')
    raw = reply['report_json'].encode('utf-8')
    if len(raw) > MAX_RESULT or hashlib.sha256(raw).hexdigest() != reply.get('report_sha256'):
        raise ValueError('Inventory result transport mismatch')
    report = json.loads(raw)
    if not (isinstance(report, dict) and report.get('worker_sha256') == worker_hash
            and report.get('gpu_requested') is False and report.get('cuda_context_requested') is False
            and type(report.get('candidate_kernel_launches')) is int and report['candidate_kernel_launches'] == 0):
        raise ValueError('Inventory source or CPU-only contract mismatch')
    return raw, report


def launch(head, receipt_path):
    ledger, result_path = HERE/'inventory-run.local.json', HERE/'result.json'
    if ledger.exists() or result_path.exists():
        raise ValueError('Existing inventory request/result: reconcile, never duplicate')
    paths, hashes = frozen_payload(head)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError('Existing regular billing receipt required')
    receipt = json.loads(receipt_path.read_text())
    configure_existing_auth()
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    if Path(modal.config.user_config_path).resolve() != AUTH.resolve():
        raise ValueError('Supported Modal config path mismatch')
    subprocess.run(['git', 'check-ignore', '-q', 'local-notes/modal-auth.toml'], cwd=ORIGINAL, check=True)
    if subprocess.check_output(['git', 'ls-files', 'local-notes/modal-auth.toml'], cwd=ORIGINAL):
        raise ValueError('Credential file unexpectedly tracked')
    billing, apps = cli('billing', 'summary', '--json'), cli('app', 'list', '--env', 'main', '--json')
    if not stopped(apps):
        raise ValueError('Existing Modal apps/tasks must be closed before inventory admission')
    rates = cli('billing', 'rates', '--json')
    rate = resource_rate(rates)
    modeled = reserve(rate)
    floor = validate_budget(receipt, billing)
    image = modal.Image.from_id(IMAGE)
    for path in paths.values():
        image = image.add_local_file(path, '/opt/fa3-image-inventory/'+path.name, copy=False)
    app = modal.App('okc-attention-fa3-cpu-image-inventory', include_source=False)
    # CPU inventory only: no gpu keyword, GPU allocation, or GPU-price reserve.
    fn = app.function(image=image, cpu=(1, 1), memory=(2048, 2048),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS, retries=0,
        serialized=True, include_source=False, block_network=True,
        restrict_modal_access=True)(make_remote_entry())
    validate_budget(receipt, billing)
    record = dict(experiment=EXPERIMENT, state='reserved', app_id=None, created_utc=now(),
        source_commit=head, payload_sha256=hashes, base_image=IMAGE,
        gpu_requested=False, cuda_context_requested=False, candidate_kernel_launches=0,
        billing_before=billing, apps_before=apps, cost_reserve_usd=str(RESERVE),
        modeled_reserve_usd=str(modeled), resource_rate_usd_per_second=str(rate),
        verified_free_floor_usd=str(floor), actual_paid_limit_usd='0',
        billing_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        artificial_stage_ceiling_removed=True, official_check=False, formal_entry=False,
        resources=dict(gpu_requested=False, cpu=[1, 1], memory_mib=[2048, 2048],
            max_containers=1, worker_alarm=WORKER_SECONDS, child_hard_timeout=CHILD_SECONDS,
            function_timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS,
            app_watchdog=WATCHDOG_SECONDS, retries=0, explicit_invocations=1))
    with ledger.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    lock, stop_result, expired = threading.Lock(), {}, threading.Event()
    started = time.monotonic()

    def stop():
        with lock:
            if stop_result or not app.app_id:
                return
            stop_result.update(requested_utc=now(), app_id=app.app_id)
            try:
                stop_result.update(cli('app', 'stop', app.app_id, '--yes', '--env', 'main'))
            except Exception as exc:
                stop_result['error_type'] = type(exc).__name__

    def watchdog():
        expired.set()
        stop()

    timer = threading.Timer(WATCHDOG_SECONDS, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name='main'):
            record.update(state='accepted', app_id=app.app_id, accepted_utc=now())
            save(ledger, record)
            print('Accepted CPU-only FA3 image inventory '+app.app_id, flush=True)
            if expired.is_set():
                raise TimeoutError('Inventory watchdog expired before invocation')
            reply = fn.remote()
            worker_key = (HERE/'worker.py').relative_to(REPO).as_posix()
            raw, report = verify_result(reply, hashes[worker_key])
            with result_path.open('xb') as stream:
                stream.write(raw)
            record.update(state='function-terminal', result_retrieved=True,
                worker_status=report.get('status'), exit_code=reply['exit_code'],
                report_sha256=reply['report_sha256'], worker_wall_seconds=reply['worker_wall_seconds'])
            save(ledger, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('CPU image inventory needs reconciliation; do not relaunch.', flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            states = cli('app', 'list', '--env', 'main', '--json')
            exact = next((row for row in states if row['app_id'] == app.app_id), None)
            record['shutdown_observation'] = dict(observed_utc=now(), app=exact)
            record['shutdown_verified'] = bool(exact and stopped([exact]))
            record['apps_after'] = states
            record['all_apps_stopped_zero_tasks'] = stopped(states)
            after = cli('billing', 'summary', '--json')
            record['billing_after'] = after
            record['observed_gross_delta_usd'] = str(max(Decimal(0), gross(after)-gross(billing)))
            record['paid_after_usd'] = after['billed_cost']
            if Decimal(after['billed_cost']) != 0:
                raise ValueError('Actual paid-$0 invariant failed')
        except Exception as exc:
            record.update(shutdown_verified=False, shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stop_result, client_finished_utc=now(),
                      client_wall_seconds=time.monotonic()-started)
        save(ledger, record)
        print(json.dumps({key: record.get(key) for key in ('app_id', 'state', 'worker_status',
            'exit_code', 'shutdown_verified', 'all_apps_stopped_zero_tasks',
            'observed_gross_delta_usd', 'paid_after_usd')}), flush=True)
    return 0 if (record.get('exit_code') == 0 and record.get('result_retrieved')
                 and record.get('shutdown_verified')) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-head', required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--billing-receipt', type=Path)
    args = parser.parse_args()
    if args.launch:
        if args.billing_receipt is None:
            parser.error('--launch requires an existing billing receipt')
        raise SystemExit(launch(args.frozen_head, args.billing_receipt))
    _, hashes = frozen_payload(args.frozen_head)
    print(json.dumps(dict(mode='offline-preflight', source_commit=args.frozen_head,
                         payload_sha256=hashes, cpu_app_launched=False, gpu_requested=False), indent=2))
