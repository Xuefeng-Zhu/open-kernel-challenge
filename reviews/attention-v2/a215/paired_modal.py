"""One bounded A2-15 paired diagnostic using an explicit file allowlist.

Local self-test never imports Modal or starts a job. Launch is fail-closed on
existing records, pending jobs, changed hashes, missing billing/auth evidence,
or a budget mismatch. No credential creation or renewal is implemented.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = Decimal('1.849496218')
RATE = Decimal('.00130496')
LIMIT = Decimal('2') - BASE
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
SOURCES = {
    'mixed_fa3_fp8_qk.py': 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',
    'mixed_fa3_page_lookahead.py': 'fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def reserve(seconds):
    return Decimal(math.ceil(seconds + 5)) * RATE * Decimal('1.1') + Decimal('.01')


def payload():
    return {**{n: HERE.parent / n for n in SOURCES},
            'paired_worker.py': HERE / 'paired_worker.py',
            'modal_worker.py': HERE.parent / 'modal_worker.py',
            'official/reference.py': HERE.parent / 'official/reference.py',
            'official/checks.py': HERE.parent / 'official/checks.py',
            'legal/LICENSE': ROOT / 'LICENSE',
            'legal/THIRD_PARTY_NOTICES.md': ROOT / 'THIRD_PARTY_NOTICES.md'}


def remote_entry():
    def run():
        import hashlib
        import json
        from pathlib import Path
        import signal
        import subprocess
        import sys
        import time
        root = Path('/opt/a215-pair')
        dest = Path('/tmp/a215-paired-result.json')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root / 'paired_worker.py'),
            '--output', str(dest), '--deadline', '52'], cwd=root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            output, _ = child.communicate(timeout=55)
            code = child.returncode
        except subprocess.TimeoutExpired:
            os_module = __import__('os')
            os_module.killpg(child.pid, signal.SIGKILL)
            output, _ = child.communicate(timeout=2)
            code = -9
        raw = dest.read_bytes() if dest.exists() else b'{}'
        if len(raw) > 100_000:
            return dict(exit_code=code, error='Result exceeded fixed 100KB guard',
                        worker_wall_seconds=time.monotonic()-started)
        # Worker contains no credentials. Bound errors rather than returning a
        # large compiler log or filesystem dump through a blob-backed result.
        return dict(exit_code=code, report_json=raw.decode(),
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                    diagnostic_output=output.decode(errors='replace')[-4000:] if code else '',
                    worker_wall_seconds=time.monotonic()-started)
    return run


def launch():
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    auth = ROOT / 'local-notes/modal-auth.toml'
    assert auth.is_file() and not auth.is_symlink()
    assert auth.stat().st_uid == os.getuid() and auth.stat().st_mode & 0o777 == 0o600
    assert Path(modal.config.user_config_path).resolve() == auth.resolve()
    assert not any(os.environ.get(k) for k in ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
        'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET'])
    subprocess.run(['git', 'check-ignore', '-q', str(auth.relative_to(ROOT))], cwd=ROOT, check=True)
    assert not subprocess.check_output(['git', 'ls-files', str(auth.relative_to(ROOT))], cwd=ROOT)
    login = json.loads((ROOT/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(login['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    assert login['state'] == 'verified' and (expiry-datetime.now(timezone.utc)).total_seconds() > 180
    pre = json.loads((ROOT/'local-notes/a215-pair-preflight.local.json').read_text())
    assert pre['workspace'] == 'xuefeng-zhu' and Decimal(pre['verified_net_spend_cap_usd']) == 0
    assert Decimal(pre['verified_credit_balance_usd']) > LIMIT
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(pre['billing_ui_verified_utc'])).total_seconds() < 900
    record_path = HERE/'paired-modal-run.json'
    assert not record_path.exists(), 'Existing run record: inspect, never relaunch'
    budget_path = HERE.parent/'modal-budget.json'
    budget = json.loads(budget_path.read_text())
    assert Decimal(budget['conservative_stage_spent_usd']) == BASE and not budget.get('active_reservation')
    assert reserve(90) < LIMIT
    slot = ROOT/'local-notes/gpu-slot.local.json'
    assert json.loads(slot.read_text())['state'] in ['idle', 'terminal']

    def cli(*args):
        p = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'modal', *args],
                           capture_output=True, text=True, timeout=15)
        if p.returncode:
            raise RuntimeError('Read-only Modal check failed: '+' '.join(args[:2]))
        return json.loads(p.stdout)

    billing = cli('billing', 'summary', '--json')
    assert Decimal(billing['billed_cost']) == 0
    apps = cli('app', 'list', '--env', 'main', '--json')
    assert all(a['state'] == 'stopped' and int(a['tasks']) == 0 for a in apps)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = payload()
    hashes = {}
    for name, path in files.items():
        assert path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        assert data == subprocess.check_output(['git', 'show', head+':'+path.relative_to(ROOT).as_posix()], cwd=ROOT)
        hashes[name] = hashlib.sha256(data).hexdigest()
    assert all(hashes[n] == h for n, h in SOURCES.items())
    image = modal.Image.from_id(IMAGE)
    for name, path in files.items():
        image = image.add_local_file(path, '/opt/a215-pair/'+name, copy=False)
    app = modal.App('okc-a215-paired', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=60, startup_timeout=15, retries=0, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(remote_entry())
    record = dict(experiment='A2-15 paired correctness/timing', state='reserved', app_id=None,
        created_utc=now(), source_commit=head, payload_sha256=hashes,
        prior_conservative_cumulative_usd=str(BASE), remaining_ceiling_usd=str(LIMIT),
        conservative_reserve_usd=str(reserve(90)), billing_before=billing,
        base_image=IMAGE, resources=dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192],
            max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
            timeout=60, startup_timeout=15, worker_timeout=55, internal_deadline=52,
            app_watchdog=75, stop_allowance=15, retries=0, explicit_invocations=1))
    save(record_path, record)
    budget['active_reservation'] = dict(experiment=record['experiment'], reserve_usd=str(reserve(90)), app_id=None, reserved_utc=now())
    save(budget_path, budget)
    save(slot, dict(state='reserved', provider='modal', experiment='A2-15-paired', app_id=None))
    stop_lock = threading.Lock()
    stop_result = {}
    started = time.monotonic()

    def stop():
        with stop_lock:
            if not app.app_id or stop_result:
                return
            stop_result.update(app_id=app.app_id, requested_utc=now())
            try:
                p = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'modal', 'app', 'stop',
                    app.app_id, '--yes', '--env', 'main'], capture_output=True, text=True, timeout=15)
                stop_result['exit_code'] = p.returncode
            except Exception as exc:
                stop_result['error_type'] = type(exc).__name__
            print('Stop requested for this same paired diagnostic app.', flush=True)

    timer = threading.Timer(75, stop)
    timer.daemon = True
    timer.start()
    try:
        with app.run(detach=False, environment_name='main'):
            record.update(state='accepted', app_id=app.app_id, accepted_utc=now())
            budget['active_reservation']['app_id'] = app.app_id
            save(budget_path, budget); save(record_path, record)
            save(slot, dict(state='accepted', provider='modal', experiment='A2-15-paired', app_id=app.app_id))
            print('Accepted paired diagnostic app '+app.app_id, flush=True)
            if time.monotonic()-started >= 75:
                stop()
                raise TimeoutError('Startup exhausted app deadline; no function call')
            result = fn.remote()
            record['remote_exit_code'] = result['exit_code']
            record['worker_wall_seconds'] = result['worker_wall_seconds']
            if 'report_json' in result:
                raw = result['report_json'].encode()
                assert len(raw) <= 100_000 and hashlib.sha256(raw).hexdigest() == result['report_sha256']
                parsed = json.loads(raw)
                (HERE/'paired-result.json').write_bytes(raw)
                record.update(result_sha256=result['report_sha256'], result_retrieval_complete=True,
                              worker_status=parsed.get('status'))
            if result.get('diagnostic_output'):
                record['worker_diagnostic'] = result['diagnostic_output']
            if result.get('error'):
                record['worker_error'] = result['error']
            record['state'] = 'function-terminal'
            save(record_path, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('Paired diagnostic requires reconciliation of this exact app: '+type(exc).__name__, flush=True)
    finally:
        stop()
        timer.cancel()
        record.update(explicit_stop=stop_result, client_finished_utc=now(), client_wall_seconds=time.monotonic()-started)
        try:
            apps = cli('app', 'list', '--env', 'main', '--json')
            exact = next((a for a in apps if a['app_id'] == app.app_id), None)
            record['shutdown_observation'] = exact
            record['shutdown_verified'] = bool(exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0)
            record['billing_after'] = cli('billing', 'summary', '--json')
        except Exception as exc:
            record['postcheck_error_type'] = type(exc).__name__
        save(record_path, record)
        save(slot, dict(state='terminal' if record.get('shutdown_verified') else 'uncertain', provider='modal',
                        experiment='A2-15-paired', app_id=app.app_id))
        print(json.dumps({k: record.get(k) for k in ['app_id', 'state', 'remote_exit_code', 'shutdown_verified', 'worker_status']}), flush=True)


def self_test():
    import ast
    assert reserve(90) == Decimal('.146368320') < LIMIT < reserve(93)
    files = payload()
    assert len(files) == 8 and all('local-notes' not in str(p) for p in files.values())
    assert all(hashlib.sha256(files[n].read_bytes()).hexdigest() == h for n, h in SOURCES.items())
    tree = ast.parse(Path(__file__).read_text())
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'remote'
               for n in ast.walk(tree)) == 1
    print('4 local budget/payload/single-invocation guards passed; no Modal call.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'self-test'])
    args = parser.parse_args()
    launch() if args.action == 'launch' else self_test()
