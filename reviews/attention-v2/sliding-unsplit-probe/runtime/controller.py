"""Explicitly launched, single H200 sliding-unsplit diagnostic; default is offline preflight.

Only committed allowlisted payload files are uploaded. Authentication remains in
the existing protected Modal config; no credential bytes enter the payload.
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
PROBE = HERE.parent
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
BASE_SHA = '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0'
CANDIDATE_SHA = '586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf'
EXPERIMENT = 'attention-sliding-unsplit'
MAX_RESULT = 500_000
WORKER_SECONDS, CHILD_SECONDS = 160, 170
FUNCTION_SECONDS, STARTUP_SECONDS, WATCHDOG_SECONDS = 185, 20, 210
RESERVE = Decimal('.58')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def payload():
    paths = [HERE/'worker.py', HERE/'identity.py', PROBE/'baseline.py', PROBE/'candidate.py',
             ATTENTION/'modal_worker.py', ATTENTION/'official/reference.py',
             ATTENTION/'official/checks.py', REPO/'LICENSE', REPO/'THIRD_PARTY_NOTICES.md']
    return {p.relative_to(REPO).as_posix(): p for p in paths}


def frozen_payload(head):
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('Exact reviewed commit required')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if actual != head:
        raise ValueError('HEAD differs from reviewed freeze')
    paths, hashes = payload(), {}
    controller_key = Path(__file__).relative_to(REPO).as_posix()
    for name, path in [*paths.items(), (controller_key, Path(__file__))]:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Payload path guard: '+name)
        raw = path.read_bytes()
        committed = subprocess.check_output(['git', 'show', head+':'+name], cwd=REPO)
        if raw != committed:
            raise ValueError('Committed payload changed: '+name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if path.suffix == '.py':
            compile(raw, name, 'exec')  # Parse only: no imports/CUDA/network.
    if hashes[(PROBE/'baseline.py').relative_to(REPO).as_posix()] != BASE_SHA:
        raise ValueError('Baseline hash mismatch')
    if hashes[(PROBE/'candidate.py').relative_to(REPO).as_posix()] != CANDIDATE_SHA:
        raise ValueError('Candidate hash mismatch')
    return paths, hashes


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Timezone required')
    return result


def validate_slot(slot, head, current=None):
    current = current or datetime.now(timezone.utc)
    if not (slot['state'] == 'granted' and slot['experiment'] == EXPERIMENT
            and slot['source_commit'] == head and slot['gpu'] == 'H200'):
        raise ValueError('Explicit parent serialized-slot grant required')
    if not (-5 <= (current-timestamp(slot['granted_utc'])).total_seconds() < 900
            and (timestamp(slot['expires_utc'])-current).total_seconds() > WATCHDOG_SECONDS):
        raise ValueError('Serialized-slot grant stale or too short')


def resource_rate(rates):
    return (Decimal(rates['gpu_hour_cost_h200'])+2*Decimal(rates['cpu_hour_cost'])
            +8*Decimal(rates['mem_gib_hour_cost']))/3600


def reserve(rate):
    # The watchdog already exceeds function+startup by 5s. Use its full duration,
    # double the current resource rate and add slack before admitting fixed $.58.
    if Decimal(WATCHDOG_SECONDS)*rate*2+Decimal('.02') > RESERVE:
        raise ValueError('Current rate exceeds reviewed fixed free-credit reserve')
    return RESERVE


def gross(billing):
    return max(Decimal(billing['metered_cost']),
               sum((Decimal(v) for v in billing['metered_cost_breakdown'].values()), Decimal(0)))


def validate_budget(receipt, billing, cost_reserve, current=None):
    current = current or datetime.now(timezone.utc)
    if not (receipt['workspace'] == 'xuefeng-zhu'
            and Decimal(receipt['verified_net_spend_cap_usd']) == 0
            and Decimal(receipt['total_spend_usd']) == 0
            and not receipt['changes_made'] and Decimal(billing['billed_cost']) == 0):
        raise ValueError('Paid-$0 guardrail not verified')
    if not -5 <= (current-timestamp(receipt['observed_utc'])).total_seconds() < 900:
        raise ValueError('Billing guardrail receipt stale')
    free_floor = Decimal(receipt['verified_credit_balance_usd'])-max(
        Decimal(0), gross(billing)-Decimal(receipt['total_usage_usd']))
    if not free_floor > cost_reserve:
        raise ValueError('Verified free credit floor cannot cover reserve')
    return free_floor


def stopped(apps):
    return all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in apps)


def credential_environment():
    if not (AUTH.is_file() and not AUTH.is_symlink() and AUTH.stat().st_uid == os.getuid()
            and AUTH.stat().st_mode & 0o777 == 0o600):
        raise ValueError('Existing protected Modal config required')
    if any(os.environ.get(k) for k in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
            'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET')):
        raise ValueError('Expanded credential environment forbidden')
    os.environ['MODAL_CONFIG_PATH'] = str(AUTH)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'
    receipt = json.loads((ORIGINAL/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(receipt['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    if not (receipt['state'] == 'verified'
            and (expiry-datetime.now(timezone.utc)).total_seconds() > WATCHDOG_SECONDS+60):
        raise ValueError('Existing verified authentication expires too soon; no renewal')


def cli(*args):
    # SDK expiry diagnostics include a token identifier: capture, never forward.
    code = ('import warnings; from modal.exception import ServerWarning; '
            'warnings.filterwarnings("ignore",message=r"Your Modal token .* expires",category=ServerWarning); '
            'from modal.__main__ import main; main()')
    child = subprocess.run([str(PYTHON), '-c', code, *args], capture_output=True,
                           text=True, timeout=20)
    if child.returncode:
        raise RuntimeError('Supported Modal operation failed: '+' '.join(args[:2]))
    return json.loads(child.stdout) if '--json' in args else {'exit_code': child.returncode}


def make_remote_entry():
    # Deliberately closure-free: local config, credentials and receipts are absent.
    def entry():
        import hashlib, os, signal, subprocess, sys, time
        from pathlib import Path
        root = Path('/opt/sliding-unsplit/reviews/attention-v2/sliding-unsplit-probe/runtime')
        output = Path('/tmp/sliding-unsplit-result.json')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'worker.py'), '--output', str(output),
            '--deadline', '160'], cwd=root, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=170)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            code = 124
        if not output.is_file() or output.is_symlink() or output.stat().st_size > 500_000:
            return dict(exit_code=code, error='Missing or oversized bounded worker result',
                        worker_wall_seconds=time.monotonic()-started)
        raw = output.read_bytes()
        return dict(exit_code=code, report_json=raw.decode('utf-8'),
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                    worker_wall_seconds=time.monotonic()-started)
    return entry


def verified_result(reply):
    if not isinstance(reply, dict) or reply.get('error') or not isinstance(reply.get('exit_code'), int):
        raise ValueError('Worker reply unavailable')
    raw = reply['report_json'].encode('utf-8')
    if len(raw) > MAX_RESULT or hashlib.sha256(raw).hexdigest() != reply['report_sha256']:
        raise ValueError('Worker result transport mismatch')
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError('Worker result must be an object')
    return raw, report


def launch(head, slot_path, billing_path):
    ledger, result_path = HERE/'run.local.json', HERE/'result.json'
    if ledger.exists() or result_path.exists():
        raise ValueError('Existing request/result: reconcile; never duplicate')
    paths, hashes = frozen_payload(head)
    slot, cap = json.loads(slot_path.read_text()), json.loads(billing_path.read_text())
    validate_slot(slot, head)
    credential_environment()
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
        raise ValueError('Serialized GPU slot not clear: active Modal tasks/apps')
    rates = cli('billing', 'rates', '--json')
    rate = resource_rate(rates)
    cost_reserve = reserve(rate)
    free_floor = validate_budget(cap, billing, cost_reserve)
    image = modal.Image.from_id(IMAGE)
    for name, path in paths.items():
        image = image.add_local_file(path, '/opt/sliding-unsplit/'+name, copy=False)
    app = modal.App('okc-attention-sliding-unsplit', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS, retries=0,
        serialized=True, include_source=False, block_network=True,
        restrict_modal_access=True)(make_remote_entry())
    # Recheck receipt time bounds after SDK/CLI preflight, immediately before
    # reserving a request. No elapsed time is treated as permission.
    validate_slot(slot, head)
    validate_budget(cap, billing, cost_reserve)
    record = dict(experiment=EXPERIMENT, state='reserved', app_id=None, created_utc=now(),
        source_commit=head, payload_sha256=hashes, base_image=IMAGE,
        billing_before=billing, apps_before=apps, cost_reserve_usd=str(cost_reserve),
        resource_rate_usd_per_second=str(rate), conservative_free_credit_floor_usd=str(free_floor),
        slot_receipt_sha256=hashlib.sha256(slot_path.read_bytes()).hexdigest(),
        billing_guardrail_receipt_sha256=hashlib.sha256(billing_path.read_bytes()).hexdigest(),
        paid_limit_usd='0', artificial_stage_ceiling_removed=True,
        resources=dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            worker_deadline=WORKER_SECONDS, child_timeout=CHILD_SECONDS,
            function_timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS,
            watchdog=WATCHDOG_SECONDS, retry_policy=0, explicit_invocations=1),
        official_check=False, formal_entry=False)
    with ledger.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    started = time.monotonic()
    lock, expired, stop_result = threading.Lock(), threading.Event(), {}

    def stop():
        with lock:
            if not app.app_id or stop_result:
                return
            stop_result.update(app_id=app.app_id, requested_utc=now())
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
            print('Accepted single sliding-unsplit diagnostic app '+app.app_id, flush=True)
            if expired.is_set():
                raise TimeoutError('Watchdog expired before invocation')
            reply = fn.remote()
            raw, report = verified_result(reply)
            with result_path.open('xb') as stream:
                stream.write(raw)
            record.update(state='function-terminal', result_retrieved=True,
                exit_code=reply['exit_code'], worker_status=report.get('status'),
                worker_passed=report.get('passed'), report_sha256=reply['report_sha256'],
                worker_wall_seconds=reply['worker_wall_seconds'])
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('Sliding-unsplit request requires reconciliation; do not relaunch.', flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            states = cli('app', 'list', '--env', 'main', '--json')
            exact = next((r for r in states if r['app_id'] == app.app_id), None)
            record['shutdown_observation'] = exact
            record['shutdown_verified'] = bool(exact and stopped([exact]))
            record['all_apps_stopped_zero_tasks'] = stopped(states)
            record['billing_after'] = cli('billing', 'summary', '--json')
            record['observed_gross_delta_usd'] = str(max(Decimal(0), gross(record['billing_after'])-gross(billing)))
            record['paid_after_usd'] = record['billing_after']['billed_cost']
            if Decimal(record['paid_after_usd']) != 0:
                raise ValueError('Paid-$0 invariant failed')
        except Exception as exc:
            record.update(shutdown_verified=False, shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stop_result, client_finished_utc=now(),
                      client_wall_seconds=time.monotonic()-started)
        save(ledger, record)
        print(json.dumps({key: record.get(key) for key in ('app_id', 'state', 'exit_code',
            'worker_status', 'worker_passed', 'shutdown_verified', 'all_apps_stopped_zero_tasks',
            'observed_gross_delta_usd', 'paid_after_usd')}), flush=True)
    return 0 if (record.get('exit_code') == 0 and record.get('worker_passed')
                 and record.get('shutdown_verified')) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-head', required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--slot-receipt', type=Path)
    parser.add_argument('--billing-receipt', type=Path)
    args = parser.parse_args()
    if args.launch:
        if not args.slot_receipt or not args.billing_receipt:
            parser.error('--launch requires explicit slot and billing receipts')
        raise SystemExit(launch(args.frozen_head, args.slot_receipt, args.billing_receipt))
    _, hashes = frozen_payload(args.frozen_head)
    print(json.dumps(dict(mode='offline-preflight', source_commit=args.frozen_head,
                          payload_sha256=hashes, gpu_launched=False), indent=2))
