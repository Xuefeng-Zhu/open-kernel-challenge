"""Single bounded compile-only Modal invocation; no credential creation/renewal.

This controller uploads only the two reviewed probe payload files. The existing
protected Modal configuration stays in the original workspace and is never copied.
"""
import argparse
import base64
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
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROBE = HERE.parent/'probe'
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
SOURCE_SHA256 = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
RATE = (Decimal('4.54')+2*Decimal('.0473')+8*Decimal('.008'))/3600
MAX_RAW = 16*1024*1024
MAX_ENCODED = 24*1024*1024
MAX_PACKED = 2*1024*1024
CHUNK = 16*1024


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def reserve(seconds=125, rate=RATE):
    return Decimal(math.ceil(seconds+5))*rate*Decimal('1.1')+Decimal('.01')


def resource_rate(rates):
    return (Decimal(rates['gpu_hour_cost_h200'])+2*Decimal(rates['cpu_hour_cost'])
        +8*Decimal(rates['mem_gib_hour_cost']))/3600


def billed_gross(billing):
    return max(Decimal(billing['metered_cost']),
        sum((Decimal(v) for v in billing['metered_cost_breakdown'].values()), Decimal(0)))


def credential_environment():
    assert AUTH.is_file() and not AUTH.is_symlink()
    assert AUTH.stat().st_uid == os.getuid() and AUTH.stat().st_mode & 0o777 == 0o600
    assert not any(os.environ.get(k) for k in ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
        'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET'])
    os.environ['MODAL_CONFIG_PATH'] = str(AUTH)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'


def cli(*args):
    # Suppress only the SDK expiry warning, which includes a credential identifier.
    # Capture all child output; never forward raw authentication diagnostics.
    code = ('import warnings; from modal.exception import ServerWarning; '
        'warnings.filterwarnings("ignore",message=r"Your Modal token .* expires",category=ServerWarning); '
        'from modal.__main__ import main; main()')
    p = subprocess.run([str(PYTHON), '-c', code, *args], capture_output=True, text=True, timeout=20)
    if p.returncode:
        raise RuntimeError('Supported Modal operation failed: '+' '.join(args[:2]))
    return json.loads(p.stdout) if '--json' in args else {'exit_code': p.returncode}


def frozen_payload(head):
    assert re.fullmatch(r'[0-9a-f]{40}', head), 'Require exact reviewed source commit'
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert actual == head, 'HEAD differs from reviewed freeze'
    paths = {n: PROBE/n for n in ['candidate.py', 'worker.py']}
    frozen = {}
    for path in [*paths.values(), Path(__file__).resolve()]:
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        committed = subprocess.check_output(['git', 'show', head+':'+path.relative_to(REPO).as_posix()], cwd=REPO)
        assert raw == committed, 'Reviewed bytes changed: '+path.name
        frozen[path.name] = hashlib.sha256(raw).hexdigest()
    assert frozen['candidate.py'] == SOURCE_SHA256
    return paths, frozen


def make_remote_entry():
    # This returned function has no local configuration, credential or state closure.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path('/opt/architecture-probe')
        folder = Path('/tmp/architecture-probe-native')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'worker.py'),
            '--output-dir', str(folder), '--deadline', '55'], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            code = 124
        files = sorted(folder.glob('*')) if folder.exists() else []
        allowed = {'summary.json'} | {f'{operand}.b{batch}.fused.{ext}'
            for operand in ['fp8', 'bf16'] for batch in [16, 32]
            for ext in ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass']}
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in files):
            yield {'kind': 'failure', 'reason': 'Artifact path guard', 'exit_code': code}
            return
        if sum(p.stat().st_size for p in files) > 16*1024*1024:
            yield {'kind': 'failure', 'reason': 'Artifact raw-size guard', 'exit_code': code}
            return
        payload = {p.name: base64.b64encode(p.read_bytes()).decode() for p in files}
        raw = json.dumps(payload, sort_keys=True).encode()
        if len(raw) > 24*1024*1024:
            yield {'kind': 'failure', 'reason': 'Artifact encoded-size guard', 'exit_code': code}
            return
        packed = zlib.compress(raw, 9)
        if len(packed) > 2*1024*1024:
            yield {'kind': 'failure', 'reason': 'Artifact packed-size guard', 'exit_code': code}
            return
        count = (len(packed)+16383)//16384
        yield {'kind': 'manifest', 'chunks': count, 'packed_bytes': len(packed),
            'raw_bytes': len(raw), 'sha256': hashlib.sha256(packed).hexdigest(),
            'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'exit_code': code, 'worker_wall_seconds': time.monotonic()-started}
        for i in range(count):
            yield {'kind': 'chunk', 'index': i, 'data': packed[i*16384:(i+1)*16384]}
        yield {'kind': 'complete', 'chunks': count}
    return entry


def unpack_artifacts(manifest, chunks, dest):
    assert len(chunks) == manifest['chunks'] and len(chunks) <= MAX_PACKED//CHUNK+1
    packed = b''.join(chunks)
    assert len(packed) == manifest['packed_bytes'] <= MAX_PACKED
    assert hashlib.sha256(packed).hexdigest() == manifest['sha256']
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, MAX_ENCODED+1)
    assert len(raw) == manifest['raw_bytes'] <= MAX_ENCODED and decoder.eof and not decoder.unconsumed_tail
    decoded = json.loads(raw)
    assert set(decoded) == set(manifest['artifact_sha256'])
    verified = {}
    for name, b64 in decoded.items():
        assert name == 'summary.json' or re.fullmatch(r'(fp8|bf16)\.b(16|32)\.fused\.(ttir|ttgir|llir|ptx|cubin|sass)', name)
        content = base64.b64decode(b64, validate=True)
        assert hashlib.sha256(content).hexdigest() == manifest['artifact_sha256'][name]
        verified[name] = content
    assert sum(map(len, verified.values())) <= MAX_RAW
    assert 'summary.json' in verified, 'Worker summary missing'
    dest.mkdir(exist_ok=False)
    for name, content in verified.items():
        (dest/name).write_bytes(content)
    return json.loads(verified['summary.json'])


def launch(head):
    ledger = HERE/'modal-run.local.json'
    assert not ledger.exists(), 'Existing request record: reconcile, never launch duplicate'
    paths, hashes = frozen_payload(head)
    credential_environment()
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    assert Path(modal.config.user_config_path).resolve() == AUTH.resolve()
    subprocess.run(['git', 'check-ignore', '-q', str(AUTH.relative_to(ORIGINAL))], cwd=ORIGINAL, check=True)
    assert not subprocess.check_output(['git', 'ls-files', str(AUTH.relative_to(ORIGINAL))], cwd=ORIGINAL)
    login = json.loads((ORIGINAL/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(login['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    assert login['state'] == 'verified' and (expiry-datetime.now(timezone.utc)).total_seconds() > 180
    # Parent supplied a fresh read-only live billing receipt. The artificial
    # stage ceiling is removed; no cap/payment/credential is changed.
    pre_path = HERE/'billing-ui.json'
    assert pre_path.is_file() and not pre_path.is_symlink()
    pre = json.loads(pre_path.read_text())
    assert pre['workspace'] == 'xuefeng-zhu' and Decimal(pre['verified_net_spend_cap_usd']) == 0
    assert Decimal(pre['total_spend_usd']) == 0 and not pre['changes_made']
    observed = datetime.fromisoformat(pre['observed_utc'].replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc)-observed).total_seconds()
    assert -5 <= age < 900, 'Fresh verified credit/cap receipt required'
    prior = json.loads((REPO/'reviews/attention-v2/modal-budget.json').read_text())
    assert not prior.get('active_reservation'), 'Prior reservation requires reconciliation'
    slot = json.loads((ORIGINAL/'local-notes/gpu-slot.local.json').read_text())
    assert slot['state'] in ['idle', 'terminal'], 'Prior GPU slot requires reconciliation'
    billing = cli('billing', 'summary', '--json')
    assert Decimal(billing['billed_cost']) == 0, 'Actual paid cost must remain zero'
    apps = cli('app', 'list', '--env', 'main', '--json')
    assert all(a['state'] == 'stopped' and int(a['tasks']) == 0 for a in apps), 'GPU slot is not idle'
    rates = cli('billing', 'rates', '--json')
    rate = resource_rate(rates)
    cost_reserve = reserve(rate=rate)
    gross = billed_gross(billing)
    free_floor = Decimal(pre['verified_credit_balance_usd'])-max(Decimal(0), gross-Decimal(pre['total_usage_usd']))
    assert free_floor > cost_reserve, 'Conservative verified free-credit headroom cannot cover bounded job'
    image = modal.Image.from_id(IMAGE)
    for name, path in paths.items():
        image = image.add_local_file(path, '/opt/architecture-probe/'+name, copy=False)
    app = modal.App('okc-m64-bf16-compile-probe', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=75, startup_timeout=20, retries=None, serialized=True, include_source=False,
        block_network=True, restrict_modal_access=True)(make_remote_entry())
    # SDK1.6.2 generators require retries=None; no explicit retry invocation exists.
    record = dict(experiment='M64 BF16 QK compile-only', state='reserved', app_id=None,
        created_utc=now(), source_commit=head, payload_sha256={n: hashes[n] for n in paths},
        controller_sha256=hashes['launch.py'], base_image=IMAGE,
        billing_before=billing, apps_before=apps, cost_reserve_usd=str(cost_reserve),
        resource_rate_usd_per_second=str(rate), rates_verified_utc=now(),
        relevant_rates={k: rates[k] for k in ['gpu_hour_cost_h200', 'cpu_hour_cost', 'mem_gib_hour_cost']},
        verified_free_credit_evidence=dict(value_usd=pre['verified_credit_balance_usd'],
            verified_utc=pre['observed_utc'], net_spend_cap_usd='0',
            fresh_credit_balance_verified=True, live_cap_verified=True,
            receipt_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest()),
        conservative_free_credit_floor_usd=str(free_floor),
        conservative_floor_method='Fresh verified remaining credits minus positive gross-usage increase since UI snapshot',
        artificial_stage_ceiling_removed=True, actual_paid_limit_usd='0',
        resources=dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            timeout=75, startup_timeout=20, worker_alarm=55, child_hard_timeout=60,
            app_watchdog=110, retry_policy=None, explicit_invocations=1),
        native_compile_only=True, correctness_tested=False, performance_tested=False)
    with ledger.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    stop_lock = threading.Lock()
    stop_result = {}
    timer = None
    started = time.monotonic()
    watchdog_expired = threading.Event()

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
        watchdog_expired.set()
        stop()

    try:
        timer = threading.Timer(110, watchdog)
        timer.daemon = True
        timer.start()
        with app.run(detach=False, environment_name='main'):
            record.update(state='accepted', app_id=app.app_id, accepted_utc=now())
            save(ledger, record)
            print('Accepted single compile probe '+app.app_id, flush=True)
            if watchdog_expired.is_set():
                raise TimeoutError('Application watchdog expired before invocation')
            manifest = None
            chunks = []
            complete = False
            for item in fn.remote_gen():
                if item['kind'] == 'manifest':
                    assert manifest is None and 0 <= item['packed_bytes'] <= MAX_PACKED
                    assert 0 <= item['raw_bytes'] <= MAX_ENCODED
                    manifest = item
                    record['transport_manifest'] = item
                    save(ledger, record)
                elif item['kind'] == 'chunk':
                    assert manifest and item['index'] == len(chunks) and len(item['data']) <= CHUNK
                    chunks.append(item['data'])
                    assert sum(map(len, chunks)) <= MAX_PACKED
                elif item['kind'] == 'complete':
                    assert manifest and item['chunks'] == len(chunks) == manifest['chunks']
                    complete = True
                else:
                    raise RuntimeError('Worker artifact guard failure')
            assert complete and manifest
            summary = unpack_artifacts(manifest, chunks, HERE.parent/'native_artifacts')
            record.update(state='function-terminal', artifact_retrieval_complete=True,
                exit_code=manifest['exit_code'], compile_status=summary['status'])
            save(ledger, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('Compile request requires reconciliation; do not relaunch.', flush=True)
    finally:
        stop()
        if timer:
            timer.cancel()
        try:
            states = cli('app', 'list', '--env', 'main', '--json')
            exact = next((a for a in states if a['app_id'] == app.app_id), None)
            record['shutdown_observation'] = exact
            record['shutdown_verified'] = bool(exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0)
            record['apps_after'] = states
            record['all_apps_stopped_zero_tasks'] = all(a['state'] == 'stopped' and int(a['tasks']) == 0 for a in states)
            record['billing_after'] = cli('billing', 'summary', '--json')
            record['observed_gross_delta_usd'] = str(max(Decimal(0), billed_gross(record['billing_after'])-gross))
            record['paid_after_usd'] = record['billing_after']['billed_cost']
            assert Decimal(record['paid_after_usd']) == 0
        except Exception as exc:
            record.update(shutdown_verified=False, shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stop_result, client_finished_utc=now(), client_wall_seconds=time.monotonic()-started)
        save(ledger, record)
        print(json.dumps({k: record.get(k) for k in ['app_id', 'state', 'exit_code', 'compile_status',
            'shutdown_verified', 'all_apps_stopped_zero_tasks', 'observed_gross_delta_usd', 'paid_after_usd', 'client_wall_seconds']}), flush=True)
    return 0 if record.get('artifact_retrieval_complete') and record.get('shutdown_verified') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-head', required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.frozen_head))
