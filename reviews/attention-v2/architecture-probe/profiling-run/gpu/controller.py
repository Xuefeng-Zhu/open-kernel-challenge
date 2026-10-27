"""One bounded activity-only H200 profiling call after CPU inventory.

Only explicit reviewed payload files are uploaded. Existing authentication stays
in its protected original config. Counter collection is separately source-gated
by the actual frozen CPU inventory and the reviewed optional counter helper.
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
REPO = HERE.parents[4]
ARCHITECTURE = HERE.parents[1]
ATTENTION = HERE.parents[2]
CONTROL = HERE.parent/'control'
INVENTORY = HERE.parent/'inventory'
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
SOURCE_SHA256 = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
HELPER_SHA256 = '59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82'
RATE = (Decimal('4.54')+2*Decimal('.0473')+8*Decimal('.008'))/3600
RESERVE = Decimal('.65')
MAX_RAW = 8*1024*1024
MAX_ENCODED = 12*1024*1024
MAX_PACKED = 4*1024*1024
CHUNK = 16*1024
ALLOWED_ARTIFACTS = {'summary.json', 'activity-trace.json',
    'fa3.ncu-rep', 'bf16_custom.ncu-rep',
    'fa3-counter-check.json', 'bf16_custom-counter-check.json'}


def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

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

def payload():
    return {'worker.py': HERE/'worker.py',
        'counters.py': HERE/'counters.py',
        'counter_target.py': HERE/'counter_target.py',
        'inventory.json': HERE/'inventory.json',
        'candidate.py': ARCHITECTURE/'probe/candidate.py',
        'candidatecustomhelper.py': ARCHITECTURE/'paired/candidatecustomhelper.py',
        'modal_worker.py': ATTENTION/'modal_worker.py',
        'official/reference.py': ATTENTION/'official/reference.py',
        'official/checks.py': ATTENTION/'official/checks.py',
        'legal/LICENSE': REPO/'LICENSE',
        'legal/THIRD_PARTY_NOTICES.md': REPO/'THIRD_PARTY_NOTICES.md'}


def frozen_payload(head):
    assert re.fullmatch(r'[0-9a-f]{40}', head), 'Require exact reviewed source commit'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
    paths, hashes = payload(), {}
    for name, path in [*paths.items(), ('controller.py', Path(__file__).resolve())]:
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert raw == subprocess.check_output(['git', 'show', head+':'+path.relative_to(REPO).as_posix()], cwd=REPO), name
        hashes[name] = hashlib.sha256(raw).hexdigest()
    assert hashes['candidate.py'] == SOURCE_SHA256
    assert hashes['candidatecustomhelper.py'] == HELPER_SHA256
    assert paths['inventory.json'].read_bytes() == (INVENTORY/'result.json').read_bytes()
    return paths, hashes


def make_remote_entry():
    # No local state, credentials, client or configuration closure is serialized.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path('/opt/profiling-run')
        folder = Path('/tmp/profiling-run')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'worker.py'),
            '--output-dir', str(folder), '--deadline', '165'], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=175)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            code = 124
        files = sorted(folder.glob('*')) if folder.exists() else []
        allowed = {'summary.json', 'activity-trace.json', 'fa3.ncu-rep', 'bf16_custom.ncu-rep',
                   'fa3-counter-check.json', 'bf16_custom-counter-check.json'}
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in files):
            yield {'kind': 'failure', 'reason': 'Artifact path guard', 'exit_code': code}
            return
        raw_bytes = sum(p.stat().st_size for p in files)
        if raw_bytes > 8*1024*1024:
            yield {'kind': 'failure', 'reason': 'Artifact raw-size guard', 'exit_code': code}
            return
        data = {p.name: base64.b64encode(p.read_bytes()).decode() for p in files}
        raw = json.dumps(data, sort_keys=True).encode()
        if len(raw) > 12*1024*1024:
            yield {'kind': 'failure', 'reason': 'Artifact encoded-size guard', 'exit_code': code}
            return
        packed = zlib.compress(raw, 9)
        if len(packed) > 4*1024*1024:
            yield {'kind': 'failure', 'reason': 'Artifact packed-size guard', 'exit_code': code}
            return
        count = (len(packed)+16383)//16384
        yield {'kind': 'manifest', 'chunks': count, 'packed_bytes': len(packed),
            'raw_bytes': len(raw), 'artifact_bytes': raw_bytes,
            'sha256': hashlib.sha256(packed).hexdigest(),
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
    assert len(raw) == manifest['raw_bytes'] <= MAX_ENCODED
    assert decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
    decoded = json.loads(raw)
    assert set(decoded) == set(manifest['artifact_sha256'])
    assert set(decoded) <= ALLOWED_ARTIFACTS and 'summary.json' in decoded
    verified = {}
    for name, b64 in decoded.items():
        content = base64.b64decode(b64, validate=True)
        assert hashlib.sha256(content).hexdigest() == manifest['artifact_sha256'][name]
        verified[name] = content
    assert sum(map(len, verified.values())) == manifest['artifact_bytes'] <= MAX_RAW
    summary = json.loads(verified['summary.json'])
    sessions = summary['hardware_counter_sessions']
    if sessions is None:
        assert summary.get('hardware_counter_sessions_upper_bound') == 2
        assert summary['counter_branch'].get('status') == 'unavailable_or_partial'
        assert summary.get('activity_passed') is True
    else:
        assert isinstance(sessions, int) and 0 <= sessions <= 2, 'Counter session bound exceeded'
    dest.mkdir(exist_ok=False)
    for name, content in verified.items():
        (dest/name).write_bytes(content)
    return summary


def inventory_terminal_evidence():
    ledger = json.loads((INVENTORY/'inventory-run.local.json').read_text())
    result = json.loads((INVENTORY/'result.json').read_text())
    assert ledger.get('result_retrieved') is True and ledger.get('exit_code') == 0
    assert result['status'] == 'complete' and result['gpu_requested'] is False
    assert result['cuda_context_requested'] is False and result['candidate_kernel_launches'] == 0
    receipt = json.loads((INVENTORY/'reconciliation.json').read_text())
    apps = receipt['apps']
    exact = next((row for row in apps if row['app_id'] == ledger['app_id']), None)
    assert exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0
    return ledger['app_id'], hashlib.sha256((INVENTORY/'result.json').read_bytes()).hexdigest()


def launch(head):
    ledger = HERE/'gpu-run.local.json'
    assert not ledger.exists() and not (HERE/'artifacts').exists(), 'Existing request/artifacts: reconcile; never duplicate'
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
    assert login['state'] == 'verified' and (expiry-datetime.now(timezone.utc)).total_seconds() > 310
    inventory_app, inventory_sha = inventory_terminal_evidence()
    pre_path = CONTROL/'billing-ui.json'
    assert pre_path.is_file() and not pre_path.is_symlink()
    pre = json.loads(pre_path.read_text())
    assert pre['workspace'] == 'xuefeng-zhu' and Decimal(pre['verified_net_spend_cap_usd']) == 0
    assert Decimal(pre['total_spend_usd']) == 0 and not pre['changes_made']
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(pre['observed_utc'].replace('Z','+00:00'))).total_seconds()
    assert -5 <= age < 900, 'Fresh authorized live credit/cap receipt required'
    billing = cli('billing','summary','--json')
    assert Decimal(billing['billed_cost']) == 0
    apps = cli('app','list','--env','main','--json')
    assert all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in apps)
    rates = cli('billing','rates','--json')
    rate = resource_rate(rates)
    modeled_reserve = Decimal(245)*rate*Decimal('1.1')+Decimal('.01')
    assert modeled_reserve <= RESERVE, 'Current-rate bounded cost exceeds reviewed reservation'
    floor = Decimal(pre['verified_credit_balance_usd'])-max(Decimal(0), billed_gross(billing)-Decimal(pre['total_usage_usd']))
    assert floor > RESERVE
    image = modal.Image.from_id(IMAGE)
    for name, path in paths.items():
        image = image.add_local_file(path, '/opt/profiling-run/'+name, copy=False)
    app = modal.App('okc-profile-bounded-diagnostic', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2,2), memory=(8192,8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=190, startup_timeout=20, retries=None, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(make_remote_entry())
    # SDK generator functions require retries=None; there is no explicit retry.
    record = dict(experiment='One activity session with optional bounded hardware counters', state='reserved', app_id=None,
        created_utc=now(), source_commit=head, payload_sha256={name: hashes[name] for name in paths},
        controller_sha256=hashes['controller.py'], base_image=IMAGE,
        billing_before=billing, apps_before=apps, cost_reserve_usd=str(RESERVE),
        modeled_reserve_usd=str(modeled_reserve), resource_rate_usd_per_second=str(rate),
        relevant_rates={key: rates[key] for key in ['gpu_hour_cost_h200','cpu_hour_cost','mem_gib_hour_cost']},
        verified_free_floor_usd=str(floor), actual_paid_limit_usd='0', artificial_stage_ceiling_removed=True,
        billing_receipt_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        prior_inventory_app_id=inventory_app, prior_inventory_result_sha256=inventory_sha,
        prior_inventory_shutdown_verified=True,
        counter_branch='Reviewed optional helper; first permission denial stops both arms',
        max_hardware_counter_sessions=2, max_activity_sessions=1,
        resources=dict(gpu='H200', cpu=[2,2], memory_mib=[8192,8192], max_containers=1,
            worker_alarm=165, child_hard_timeout=175, function_timeout=190,
            startup_timeout=20, app_watchdog=210, retry_policy=None, explicit_invocations=1),
        artifact_allowlist=sorted(ALLOWED_ARTIFACTS))
    with ledger.open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n')
    started = time.monotonic()
    lock, stopped, expired = threading.Lock(), {}, threading.Event()

    def stop():
        with lock:
            if stopped or not app.app_id:
                return
            stopped.update(requested_utc=now(), app_id=app.app_id)
            try:
                stopped.update(cli('app','stop',app.app_id,'--yes','--env','main'))
            except Exception as exc:
                stopped['error_type'] = type(exc).__name__

    def watchdog():
        expired.set()
        stop()

    timer = threading.Timer(210,watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name='main'):
            record.update(state='accepted',app_id=app.app_id,accepted_utc=now())
            save(ledger,record)
            print('Accepted one bounded profiling app '+app.app_id,flush=True)
            if expired.is_set():
                raise TimeoutError('App watchdog expired before invocation')
            manifest, chunks, complete = None, [], False
            for item in fn.remote_gen():
                if item['kind'] == 'manifest':
                    assert manifest is None and 0 <= item['packed_bytes'] <= MAX_PACKED
                    assert 0 <= item['raw_bytes'] <= MAX_ENCODED and 0 <= item['artifact_bytes'] <= MAX_RAW
                    manifest = item
                    record['transport_manifest'] = item
                    save(ledger,record)
                elif item['kind'] == 'chunk':
                    assert manifest and item['index'] == len(chunks) and len(item['data']) <= CHUNK
                    chunks.append(item['data'])
                    assert sum(map(len,chunks)) <= MAX_PACKED
                elif item['kind'] == 'complete':
                    assert manifest and item['chunks'] == len(chunks) == manifest['chunks']
                    complete = True
                else:
                    raise RuntimeError('Worker artifact guard failure')
            assert manifest and complete
            summary = unpack_artifacts(manifest,chunks,HERE/'artifacts')
            assert summary['worker_sha256'] == hashes['worker.py']
            record.update(state='function-terminal',artifacts_retrieved=True,
                exit_code=manifest['exit_code'],worker_status=summary['status'],worker_passed=summary['passed'])
            save(ledger,record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation',app_id=app.app_id,error_type=type(exc).__name__)
        print('Activity request requires reconciliation; do not relaunch.',flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            states = cli('app','list','--env','main','--json')
            exact = next((row for row in states if row['app_id'] == app.app_id),None)
            record['shutdown_observation'] = exact
            record['shutdown_verified'] = bool(exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0)
            record['apps_after'] = states
            record['all_apps_stopped_zero_tasks'] = all(row['state'] == 'stopped' and int(row['tasks']) == 0 for row in states)
            after = cli('billing','summary','--json')
            record['billing_after'] = after
            record['observed_gross_delta_usd'] = str(max(Decimal(0),billed_gross(after)-billed_gross(billing)))
            assert Decimal(after['billed_cost']) == 0
        except Exception as exc:
            record.update(shutdown_verified=False,shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stopped,client_finished_utc=now(),client_wall_seconds=time.monotonic()-started)
        save(ledger,record)
        print(json.dumps({key: record.get(key) for key in ['app_id','state','exit_code','worker_status',
            'worker_passed','shutdown_verified','all_apps_stopped_zero_tasks','observed_gross_delta_usd']}),flush=True)
    return 0 if record.get('artifacts_retrieved') and record.get('shutdown_verified') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-head',required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.frozen_head))
