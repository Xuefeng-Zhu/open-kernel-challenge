"""Single source-bound B32/L8 selected FA3 native capture executor; offline by default.

Only the explicit flattened source allowlist is uploaded. Existing credentials
stay in the protected local config and never enter the remote closure/payload.
"""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import io
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
ATTENTION = HERE.parents[1]
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
EXPERIMENT = 'attention-fa3-b32-capture-executor'
WORKER_SECONDS, CHILD_SECONDS = 190, 205
FUNCTION_SECONDS, STARTUP_SECONDS, WATCHDOG_SECONDS = 220, 20, 250
MAX_RESULT, MAX_TRACE, MAX_GZIP = 1024*1024, 32*1024*1024, 6*1024*1024
MAX_TRACE_ENCODING = 8*1024*1024
MAX_NCU_REPORT = 64*1024*1024
MAX_CSV = 4*1024*1024
CHUNK_ENCODING = 1024*1024
MAX_ARTIFACT_ENCODING = 112*1024*1024
MAX_EXPECTED_NATIVE = 64*1024
REQUIRED_ARTIFACTS = {'baseline.ncu-rep': MAX_NCU_REPORT,
    'profile-original.csv': MAX_CSV, 'profile-imported.csv': MAX_CSV}
METRICS = ('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active')
RESERVE = Decimal('.70')
CASE = {'batch': 32, 'q_len': 8, 'ctx': 80000, 'heads': 'tp8',
    'layer': 'full', 'page': 1, 'kv_scale': 1., 'spread': 'agentic',
    'dist': 'coherent', 'seed': 135741}
EXPECTED = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
    'modal_worker.py': '9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',
    'counter_helper.py': 'd984cbd44691cfead1c889cbb2ae8b0bde2e11e0a1d703bbdd78a7ecf1b30637',
    'cli_contract.py': '7391280b4577a2d1f242bdffdc42777a26a180b1cdcd6e021dc47c2866c1137a',
    'trace_helpers.py': 'd89e2f6dac4d8afd9105b713ce255407df8e9253c330a1e727d8193a220c5ec8',
    'cpu_identity.py': '29e4f87f294142a38f926a7bf6f57e8032d615be7ab0da4a17e2f0962a50ec92',
    'capture_plan.py': 'be067afbc933020f41ea12f103c15263bc0e1c991c4a80ec1a1b6c1b763c26db',
    'expected-native.json': 'f0ac3f62f101201d15e66534ce62b1534cbceaa71243d97c72c2865ad2170795',
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def payload():
    return {'worker.py': HERE/'worker.py',
        'selected.py': ATTENTION/'fusion-lookahead-probe/candidate.py',
        'reference.py': ATTENTION/'official/reference.py',
        'checks.py': ATTENTION/'official/checks.py',
        'modal_worker.py': ATTENTION/'modal_worker.py',
        'counter_helper.py': HERE.parent/'baseline-runtime-identity/counter_gate.py',
        'cli_contract.py': HERE.parent/'eager-counter-repair/worker.py',
        'trace_helpers.py': HERE.parent/'baseline-runtime-identity/analyze_partial.py',
        'cpu_identity.py': HERE.parent/'dispatch-identity-cpu/worker.py',
        'csv_helpers.py': HERE/'csv_helpers.py',
        'capture_plan.py': HERE/'capture_plan.py',
        'expected-native.json': HERE/'expected-native.json',
        'LICENSE': REPO/'LICENSE', 'THIRD_PARTY_NOTICES.md': REPO/'THIRD_PARTY_NOTICES.md'}


def frozen_payload(head):
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('Exact reviewed commit required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() != head:
        raise ValueError('HEAD differs from reviewed freeze')
    paths, hashes = payload(), {}
    for name, path in [*paths.items(), ('controller.py', Path(__file__))]:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Regular frozen payload file required: '+name)
        raw = path.read_bytes()
        if name == 'expected-native.json' and not 0 < len(raw) <= MAX_EXPECTED_NATIVE:
            raise ValueError('Expected native JSON absent or exceeds64KiB')
        key = path.relative_to(REPO).as_posix()
        if raw != subprocess.check_output(['git', 'show', head+':'+key], cwd=REPO):
            raise ValueError('Committed source mismatch: '+name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if path.suffix == '.py':
            compile(raw, name, 'exec')
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise ValueError('Fixed source/license binding mismatch: '+name)
    return paths, hashes


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Timezone required')
    return result


def validate_slot(slot, head, current=None):
    current = current or datetime.now(timezone.utc)
    if not (slot.get('state') == 'granted' and slot.get('experiment') == EXPERIMENT
            and slot.get('source_commit') == head and slot.get('gpu') == 'H200'):
        raise ValueError('Source-bound serialized slot receipt required')
    if not (-5 <= (current-timestamp(slot['granted_utc'])).total_seconds() < 900
            and (timestamp(slot['expires_utc'])-current).total_seconds() > WATCHDOG_SECONDS):
        raise ValueError('Slot receipt stale or too short')


def gross(billing):
    return max(Decimal(billing['metered_cost']), sum(
        (Decimal(v) for v in billing['metered_cost_breakdown'].values()), Decimal(0)))


def resource_rate(rates):
    return (Decimal(rates['gpu_hour_cost_h200'])+2*Decimal(rates['cpu_hour_cost'])
        +8*Decimal(rates['mem_gib_hour_cost']))/3600


def reserve(rate):
    if rate <= 0 or Decimal(WATCHDOG_SECONDS)*rate*2+Decimal('.02') > RESERVE:
        raise ValueError('Live resource rate exceeds fixed free-credit reserve')
    return RESERVE


def validate_budget(receipt, billing, current=None):
    current = current or datetime.now(timezone.utc)
    if not (receipt.get('workspace') == 'xuefeng-zhu'
            and Decimal(receipt['verified_net_spend_cap_usd']) == 0
            and Decimal(receipt['total_spend_usd']) == 0 and receipt.get('changes_made') is False
            and Decimal(billing['billed_cost']) == 0):
        raise ValueError('Existing paid-$0 guardrail required')
    if not -5 <= (current-timestamp(receipt['observed_utc'])).total_seconds() < 900:
        raise ValueError('Billing observation stale')
    floor = Decimal(receipt['verified_credit_balance_usd'])-max(
        Decimal(0), gross(billing)-Decimal(receipt['total_usage_usd']))
    if floor <= RESERVE:
        raise ValueError('Verified free-credit floor cannot cover reserve')
    return floor


def stopped(apps):
    return all(row.get('state') == 'stopped' and int(row.get('tasks', -1)) == 0 for row in apps)


def configure_existing_auth():
    if not (AUTH.is_file() and not AUTH.is_symlink() and AUTH.stat().st_uid == os.getuid()
            and AUTH.stat().st_mode & 0o777 == 0o600):
        raise ValueError('Existing protected config required')
    if any(os.environ.get(key) for key in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
            'MODAL_OAUTH_REFRESH_TOKEN', 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET')):
        raise ValueError('Credential environment override forbidden')
    os.environ['MODAL_CONFIG_PATH'] = str(AUTH)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'


def cli(*args):
    # Token expiry warnings may contain an identifier; capture without forwarding.
    code = ('import warnings; from modal.exception import ServerWarning; '
        'warnings.filterwarnings("ignore",message=r"Your Modal token .* expires",category=ServerWarning); '
        'from modal.__main__ import main; main()')
    env = os.environ.copy()
    env['TZ'] = 'UTC'
    child = subprocess.run([str(PYTHON), '-c', code, *args], env=env,
        capture_output=True, text=True, timeout=25)
    if child.returncode:
        raise RuntimeError('Supported Modal operation failed: '+' '.join(args[:2]))
    return json.loads(child.stdout) if '--json' in args else {'exit_code': child.returncode}


def verify_existing_auth(current=None):
    # Read-only CLI status, not credential generation. Never emit raw stdout.
    env = os.environ.copy()
    env['TZ'] = 'UTC'
    child = subprocess.run([str(PYTHON.parent/'modal'), 'token', 'info', '--profile', 'xuefeng-zhu'],
        env=env, capture_output=True, text=True, timeout=25)
    if child.returncode:
        raise RuntimeError('Existing Modal authentication status failed; no renewal')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', child.stdout)
    workspace = re.search(r'^Workspace:\s*([^\s(]+)', text, re.M)
    expiry = re.search(r'^Expires at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\s*$', text, re.M)
    if not workspace or workspace.group(1) != 'xuefeng-zhu' or not expiry:
        raise ValueError('Expected workspace and finite UTC expiry required')
    expires = datetime.strptime(expiry.group(1), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    current = current or datetime.now(timezone.utc)
    if (expires-current).total_seconds() <= WATCHDOG_SECONDS+60:
        raise ValueError('Existing credential expires too soon; no renewal')
    return {'workspace': 'xuefeng-zhu', 'expires_at_utc': expires.isoformat(), 'status': 'verified'}


def make_remote_entry():
    # Closure-free: protected auth and local billing/slot receipts are absent.
    def entry():
        import base64, gzip, hashlib, os, signal, subprocess, sys, time
        from pathlib import Path
        root, output = Path('/opt/attention-fa3-b32-capture-executor'), Path('/tmp/attention-fa3-b32-capture-executor')
        log_path = Path('/tmp/attention-fa3-b32-capture-executor-worker.log')
        started = time.monotonic()
        with log_path.open('wb') as log:
            child = subprocess.Popen([sys.executable, str(root/'worker.py'), '--output-dir', str(output),
                '--deadline', '190'], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=205)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=3)
                code = 124
        reply = {'exit_code': code, 'remote_entry_elapsed_seconds': time.monotonic()-started}
        summary = output/'summary.json'
        if summary.is_file() and not summary.is_symlink() and summary.stat().st_size <= 1024*1024:
            raw = summary.read_bytes()
            reply.update(report_json=raw.decode('utf-8'), report_sha256=hashlib.sha256(raw).hexdigest())
        else:
            reply['error'] = 'Missing or oversized summary'
        # Explicit worker-owned outputs only; no directory glob or private files.
        reply['artifacts'] = {}
        reply['artifact_errors'] = []
        encoded_total = 0
        for name, limit in (('baseline.ncu-rep', 64*1024*1024),
                ('profile-original.csv', 4*1024*1024), ('profile-imported.csv', 4*1024*1024)):
            path = output/name
            if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= limit:
                reply['artifact_errors'].append(name+': absent, empty or exceeds fixed limit')
                continue
            before = path.stat()
            with path.open('rb') as stream:
                data = stream.read(limit+1)
            after = path.stat()
            if not 0 < len(data) <= limit:
                reply['artifact_errors'].append(name+': read exceeds fixed limit')
                continue
            packed = gzip.compress(data, compresslevel=1, mtime=0)
            if len(packed) > limit+1024*1024:
                reply['artifact_errors'].append(name+': compressed output exceeds bound')
                continue
            encoding = base64.b64encode(packed).decode('ascii')
            encoded_total += len(encoding)
            if encoded_total > 112*1024*1024:
                reply['artifact_errors'].append(name+': total encoded transport bound')
                break
            reply['artifacts'][name] = dict(raw_bytes=len(data),
                raw_sha256=hashlib.sha256(data).hexdigest(), compression='gzip',
                gzip_bytes=len(packed), gzip_sha256=hashlib.sha256(packed).hexdigest(),
                stable_file_snapshot=(before.st_ino, before.st_size, before.st_mtime_ns)
                    == (after.st_ino, after.st_size, after.st_mtime_ns),
                chunks_base64=[encoding[i:i+1024*1024] for i in range(0, len(encoding), 1024*1024)])
        trace, compressed = output/'activity-trace.json', output/'activity-trace.json.gz'
        if trace.exists() or compressed.exists():
            if (not trace.is_file() or trace.is_symlink() or trace.stat().st_size > 32*1024*1024
                    or not compressed.is_file() or compressed.is_symlink() or compressed.stat().st_size > 6*1024*1024):
                reply['artifact_error'] = 'Trace absent or exceeds fixed raw/compressed limit'
            else:
                raw, packed = trace.read_bytes(), compressed.read_bytes()
                encoding = base64.b64encode(packed).decode('ascii')
                if len(encoding) > 8*1024*1024:
                    reply['artifact_error'] = 'Trace encoding exceeds transport limit'
                else:
                    reply['trace'] = dict(raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                        gzip_bytes=len(packed), gzip_sha256=hashlib.sha256(packed).hexdigest(), gzip_base64=encoding)
        if code:
            with log_path.open('rb') as stream:
                stream.seek(max(0, log_path.stat().st_size-4096))
                reply['worker_log_tail'] = stream.read(4096).decode('utf-8', errors='replace')
        return reply
    return entry


def verified_summary(reply):
    if not isinstance(reply, dict) or type(reply.get('exit_code')) is not int:
        raise ValueError('Invalid remote reply')
    if not isinstance(reply.get('report_json'), str):
        return None, None
    raw = reply['report_json'].encode('utf-8')
    if len(raw) > MAX_RESULT or hashlib.sha256(raw).hexdigest() != reply.get('report_sha256'):
        raise ValueError('Summary source/size digest mismatch')
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError('Summary must be an object')
    return raw, report


def decode_artifacts(values):
    """Lossless bounded durable outputs, including partial outputs on failure."""
    if not isinstance(values, dict) or not set(values) <= set(REQUIRED_ARTIFACTS):
        raise ValueError('Unexpected durable artifact names')
    total = 0
    result = {}
    for name, row in values.items():
        limit = REQUIRED_ARTIFACTS[name]
        chunks = row.get('chunks_base64')
        if (row.get('compression') != 'gzip' or not isinstance(chunks, list)
                or not 0 < len(chunks) <= 128
                or any(not isinstance(value, str) or not 0 < len(value) <= CHUNK_ENCODING for value in chunks)):
            raise ValueError('Durable artifact chunk structure exceeds bound: '+name)
        total += sum(len(value) for value in chunks)
        if total > MAX_ARTIFACT_ENCODING:
            raise ValueError('Durable artifact total encoding bound')
        packed = base64.b64decode(''.join(chunks), validate=True)
        if (not 0 < len(packed) <= limit+1024*1024 or len(packed) != row.get('gzip_bytes')
                or hashlib.sha256(packed).hexdigest() != row.get('gzip_sha256')):
            raise ValueError('Durable artifact compressed digest mismatch: '+name)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(limit+1)
        if (not 0 < len(raw) <= limit or len(raw) != row.get('raw_bytes')
                or hashlib.sha256(raw).hexdigest() != row.get('raw_sha256')):
            raise ValueError('Durable artifact raw digest mismatch: '+name)
        result[name] = {'raw': raw, 'manifest': {key: value for key, value in row.items()
            if key != 'chunks_base64'}}
    return result


def verified_result(reply, _verified_durable=None):
    # Internal callers may first persist decode_artifacts() outputs, so a
    # separate trace failure cannot discard an otherwise intact durable report.
    durable = (decode_artifacts(reply.get('artifacts', {}))
        if _verified_durable is None else _verified_durable)
    raw, report = verified_summary(reply)
    if raw is None:
        return None, None, None, durable
    trace = reply.get('trace')
    artifact = None
    if trace is not None:
        if not isinstance(trace.get('gzip_base64'), str) or len(trace['gzip_base64']) > MAX_TRACE_ENCODING:
            raise ValueError('Trace transport bound')
        packed = base64.b64decode(trace['gzip_base64'], validate=True)
        if (len(packed) > MAX_GZIP or len(packed) != trace.get('gzip_bytes')
                or hashlib.sha256(packed).hexdigest() != trace.get('gzip_sha256')):
            raise ValueError('Compressed trace digest mismatch')
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            decoded = stream.read(MAX_TRACE+1)
        if (len(decoded) > MAX_TRACE or len(decoded) != trace.get('raw_bytes')
                or hashlib.sha256(decoded).hexdigest() != trace.get('raw_sha256')):
            raise ValueError('Raw trace digest mismatch')
        artifact = {'gzip': packed, 'raw': decoded,
            'manifest': {k: v for k, v in trace.items() if k != 'gzip_base64'}}
    return raw, report, artifact, durable


def verify_csv_roundtrip(durable, parser):
    original = parser(durable['profile-original.csv']['raw'].decode('utf-8'), METRICS)
    imported = parser(durable['profile-imported.csv']['raw'].decode('utf-8'), METRICS)
    if (original.get('row_count') != 1 or imported.get('row_count') != 1
            or original.get('kernel_name') != imported.get('kernel_name')
            or original.get('grid') != imported.get('grid')
            or original.get('block') != imported.get('block')):
        raise ValueError('Transported capture/import CSV signatures or launch shapes differ')
    if set(original.get('metrics', {})) != set(METRICS) or set(imported.get('metrics', {})) != set(METRICS):
        raise ValueError('Transported CSV requested metrics differ')
    for name in METRICS:
        a, b = original['metrics'][name], imported['metrics'][name]
        if a.get('value') != b.get('value') or a.get('unit') != b.get('unit'):
            raise ValueError('Transported CSV metric values or units differ: '+name)
    return {'signature_equal': True, 'launch_equal': True,
        'metric_values_units_equal': True, 'one_row_each': True,
        'kernel_name': original['kernel_name'], 'grid': original['grid'], 'block': original['block']}


def local_csv_parser(hashes):
    path = HERE/'csv_helpers.py'
    if (path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get('csv_helpers.py')):
        raise ValueError('Frozen local CSV parser differs before report validation')
    spec = importlib.util.spec_from_file_location('recapture_controller_csv', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_ncu_csv


def local_capture_plan(hashes):
    path = HERE/'capture_plan.py'
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get('capture_plan.py'):
        raise ValueError('Frozen capture-plan helper differs')
    spec = importlib.util.spec_from_file_location('native_executor_capture_plan', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_native(hashes):
    path = HERE/'expected-native.json'
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_EXPECTED_NATIVE:
        raise ValueError('Bounded expected native identity file required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != hashes.get('expected-native.json'):
        raise ValueError('Frozen expected native identity differs')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected native identity must be an object')
    return value


def runtime_bindings(report):
    cpu = report.get('cpu_file_identity', {})
    def one(rows, suffix):
        selected = [row['sha256'] for row in rows if row.get('path', '').endswith(suffix)]
        if len(selected) != 1:
            raise ValueError('Exactly one runtime vendor identity required: '+suffix)
        return selected[0]
    result = {'selected_sha256': report['verified_hashes']['selected.py'],
        'reference_sha256': report['verified_hashes']['reference.py'],
        'wrapper_sha256': one(cpu.get('files', []), '/sgl_kernel/flash_attn.py'),
        'binary_sha256': one(cpu.get('binary_files', []), '/sgl_kernel/flash_ops.abi3.so')}
    result.update({'fixture.'+name: value['sha256'] for name, value in report['fixture']['tensors'].items()})
    return result


def validate_report(report, hashes, artifact, durable):
    """Bind completion claims to frozen sources and the verified raw trace.

    The caller saves raw results first, including failed/foreign-source evidence.
    A partial diagnostic is preserved but cannot satisfy completion admission.
    """
    if not isinstance(report, dict):
        raise ValueError('No diagnostic summary retrieved')
    expected = {name: digest for name, digest in hashes.items()
        if name not in ('controller.py', 'worker.py')}
    if report.get('worker_sha256') != hashes['worker.py'] or report.get('verified_hashes') != expected:
        raise ValueError('Retrieved worker/source bindings differ from frozen payload')
    if report.get('passed') is not True or report.get('status') != 'complete':
        return False
    if (report.get('case') != CASE
            or any(report.get(field) is not False for field in (
                'selected_candidate_changed', 'official_check', 'formal_entry',
                'security_changes', 'performance_variant'))
            or report.get('counter_sessions_attempted') != 1
            or report.get('activity_sessions') != 1):
        raise ValueError('Completion does not match unchanged single-case diagnostic scope')
    if artifact is None:
        raise ValueError('Complete activity diagnostic must retain its raw trace')
    trace = report.get('trace_artifacts', {})
    manifest = artifact['manifest']
    for name, bytes_key, hash_key in (
        ('activity-trace.json', 'raw_bytes', 'raw_sha256'),
        ('activity-trace.json.gz', 'gzip_bytes', 'gzip_sha256')):
        if trace.get(name) != {'bytes': manifest[bytes_key], 'sha256': manifest[hash_key]}:
            raise ValueError('Summary trace binding differs from retained artifact')
    if set(durable) != set(REQUIRED_ARTIFACTS):
        raise ValueError('Complete recapture requires nonempty durable report and both CSV outputs')
    metadata = report.get('artifact_manifest', {})
    for name in REQUIRED_ARTIFACTS:
        row = durable[name]['manifest']
        if row.get('stable_file_snapshot') is not True:
            raise ValueError('Durable output changed during transport: '+name)
        if metadata.get(name) != {'bytes': row['raw_bytes'], 'sha256': row['raw_sha256']}:
            raise ValueError('Summary durable artifact identity mismatch: '+name)
    descriptor = report.get('ncu_report_artifact', {})
    saved = durable['baseline.ncu-rep']['manifest']
    if descriptor != {'filename': 'baseline.ncu-rep', 'bytes': saved['raw_bytes'],
            'sha256': saved['raw_sha256']}:
        raise ValueError('Durable NCU report descriptor mismatch')
    trip = report.get('report_roundtrip', {})
    if (trip.get('passed') is not True or type(trip.get('import_exit_code')) is not int
            or trip['import_exit_code'] != 0
            or any(trip.get(field) is not True for field in (
                'signature_equal', 'launch_equal', 'metric_values_units_equal', 'report_unchanged'))):
        raise ValueError('Offline import/equivalence did not pass for the same report')
    expected_report = {'bytes': saved['raw_bytes'], 'sha256': saved['raw_sha256']}
    if trip.get('report_before_import') != expected_report or trip.get('report_after_import') != expected_report:
        raise ValueError('Offline import was not bound to the transported NCU report')
    commands = report.get('commands', [])
    captures = [row for row in commands if '--export' in row.get('argv', [])]
    imports = [row for row in commands if '--import' in row.get('argv', [])]
    if (len(captures) != 1 or len(imports) != 1 or any(row.get('state') != 'complete'
            or type(row.get('returncode')) is not int or row['returncode'] != 0
            for row in [*captures, *imports])):
        raise ValueError('One successful capture and same-report import command required')
    capture_path = captures[0]['argv'][captures[0]['argv'].index('--export')+1]
    import_path = imports[0]['argv'][imports[0]['argv'].index('--import')+1]
    if (capture_path != import_path or not Path(import_path).is_absolute()
            or Path(import_path).name != 'baseline.ncu-rep'):
        raise ValueError('Capture and offline import must refer to the same absolute report')
    parser = local_csv_parser(hashes)
    verify_csv_roundtrip(durable, parser)
    plan = local_capture_plan(hashes)
    expected = expected_native(hashes)
    bindings = runtime_bindings(report)
    if bindings != report.get('native_capture_bindings'):
        raise ValueError('Native source/wrapper/binary/fixture bindings differ from runtime evidence')
    capture = {'returncode': captures[0]['returncode'],
        'stdout': durable['profile-original.csv']['raw'].decode('utf-8'),
        'report_path': capture_path, 'report': expected_report,
        'bindings': bindings}
    imported = {'returncode': imports[0]['returncode'],
        'stdout': durable['profile-imported.csv']['raw'].decode('utf-8'),
        'report_path': import_path, 'report': expected_report,
        'bindings': bindings}
    capture['row'] = parser(capture['stdout'], METRICS)
    imported['row'] = parser(imported['stdout'], METRICS)
    native = plan.capture_import_gate(capture, imported, expected)
    if native.get('accepted') is not True or report.get('native_capture_identity') != native:
        raise ValueError('Native capture acceptance differs from the same local identity gate')
    return True


def materialize_result(reply, destination, hashes):
    """Actual launch and offline dryrun share this bounded persistence/gate path."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw, report = verified_summary(reply)
    if raw is not None:
        with (destination/'result.json').open('xb') as stream:
            stream.write(raw)
    durable = decode_artifacts(reply.get('artifacts', {}))
    if durable:
        target = destination/'artifacts'
        target.mkdir(exist_ok=False)
        for name, item in durable.items():
            with (target/name).open('xb') as stream:
                stream.write(item['raw'])
        save(destination/'durable-artifacts.json', {name: item['manifest'] for name, item in durable.items()})
    raw, report, trace, _ = verified_result(reply, _verified_durable=durable)
    if trace is not None:
        (destination/'activity-trace.json.gz').write_bytes(trace['gzip'])
        save(destination/'trace-manifest.json', trace['manifest'])
    complete = reply['exit_code'] == 0 and validate_report(report, hashes, trace, durable)
    return {'report': report, 'completion_verified': complete,
        'durable_artifacts_retrieved': sorted(durable), 'trace_retrieved': trace is not None}


def launch(head, slot_path, billing_path):
    ledger, result_path = HERE/'run.local.json', HERE/'result.json'
    if ledger.exists() or result_path.exists():
        raise ValueError('Existing request/result: reconcile; never duplicate')
    paths, hashes = frozen_payload(head)
    slot, cap = json.loads(slot_path.read_text()), json.loads(billing_path.read_text())
    validate_slot(slot, head)
    configure_existing_auth()
    auth_status = verify_existing_auth()
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    if Path(modal.config.user_config_path).resolve() != AUTH.resolve():
        raise ValueError('Supported config path mismatch')
    subprocess.run(['git', 'check-ignore', '-q', 'local-notes/modal-auth.toml'], cwd=ORIGINAL, check=True)
    if subprocess.check_output(['git', 'ls-files', 'local-notes/modal-auth.toml'], cwd=ORIGINAL):
        raise ValueError('Protected credential config unexpectedly tracked')
    billing, apps = cli('billing', 'summary', '--json'), cli('app', 'list', '--env', 'main', '--json')
    if not stopped(apps):
        raise ValueError('Serialized slot unavailable: existing Modal app/tasks active')
    rates = cli('billing', 'rates', '--json')
    rate = resource_rate(rates)
    cost_reserve, free_floor = reserve(rate), validate_budget(cap, billing)
    image = modal.Image.from_id(IMAGE)
    for name, path in paths.items():
        image = image.add_local_file(path, '/opt/attention-fa3-b32-capture-executor/'+name, copy=False)
    app = modal.App('okc-attention-fa3-b32-capture-executor', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS, retries=0,
        serialized=True, include_source=False, block_network=True,
        restrict_modal_access=True)(make_remote_entry())
    validate_slot(slot, head)
    validate_budget(cap, billing)
    record = dict(experiment=EXPERIMENT, state='reserved', app_id=None, created_utc=now(),
        source_commit=head, payload_sha256=hashes, base_image=IMAGE,
        verified_auth=auth_status, billing_before=billing, apps_before=apps,
        resource_rate_usd_per_second=str(rate), cost_reserve_usd=str(cost_reserve),
        conservative_free_credit_floor_usd=str(free_floor), paid_limit_usd='0',
        slot_receipt_sha256=hashlib.sha256(slot_path.read_bytes()).hexdigest(),
        billing_guardrail_receipt_sha256=hashlib.sha256(billing_path.read_bytes()).hexdigest(),
        resources=dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            worker_deadline=WORKER_SECONDS, child_timeout=CHILD_SECONDS,
            function_timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS,
            watchdog=WATCHDOG_SECONDS, retries=0, explicit_invocations=1),
        official_check=False, formal_entry=False, candidate_change=False, performance_variant=False, security_changes=False)
    with ledger.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    started = time.monotonic()
    lock, expired, stop_result = threading.Lock(), threading.Event(), {}

    def stop():
        with lock:
            if not app.app_id or stop_result.get('exit_code') == 0:
                return
            # Retry only a failed stop request; never create another app/invocation.
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
            print('Accepted single selected FA3 B32/L8 native capture executor '+app.app_id, flush=True)
            if expired.is_set():
                raise TimeoutError('Watchdog expired before invocation')
            reply = fn.remote()
            raw, report = verified_summary(reply)
            record.update(state='function-terminal', result_retrieved=raw is not None,
                exit_code=reply['exit_code'], worker_status=(report or {}).get('status'),
                worker_passed=(report or {}).get('passed'), report_sha256=reply.get('report_sha256'),
                remote_entry_elapsed_seconds=reply.get('remote_entry_elapsed_seconds'),
                error=reply.get('error'), worker_log_tail=reply.get('worker_log_tail'))
            save(ledger, record)
            outcome = materialize_result(reply, HERE, hashes)
            record.update(completion_verified=outcome['completion_verified'],
                trace_retrieved=outcome['trace_retrieved'],
                durable_artifacts_retrieved=outcome['durable_artifacts_retrieved'],
                durable_artifact_errors=reply.get('artifact_errors', []))
            save(ledger, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('B32 FA3 identity recapture requires reconciliation; do not relaunch.', flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            states = cli('app', 'list', '--env', 'main', '--json')
            exact = next((row for row in states if row.get('app_id') == app.app_id), None)
            observations = []
            for _ in range(3):
                observations.append(dict(observed_utc=now(), app=exact))
                if exact and stopped([exact]):
                    break
                time.sleep(2)
                states = cli('app', 'list', '--env', 'main', '--json')
                exact = next((row for row in states if row.get('app_id') == app.app_id), None)
            record['shutdown_observations'] = observations
            record['shutdown_observation'] = dict(observed_utc=now(), app=exact)
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
            'worker_status', 'worker_passed', 'trace_retrieved', 'shutdown_verified',
            'all_apps_stopped_zero_tasks', 'observed_gross_delta_usd', 'paid_after_usd')}), flush=True)
    return 0 if (record.get('exit_code') == 0 and record.get('completion_verified') is True
        and record.get('worker_passed') is True and record.get('worker_status') == 'complete'
        and record.get('shutdown_verified')) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-head', required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--slot', type=Path)
    parser.add_argument('--billing-receipt', type=Path)
    args = parser.parse_args()
    if args.launch:
        if args.slot is None or args.billing_receipt is None:
            parser.error('--launch requires --slot and --billing-receipt')
        raise SystemExit(launch(args.frozen_head, args.slot, args.billing_receipt))
    _, hashes = frozen_payload(args.frozen_head)
    print(json.dumps(dict(mode='offline-preflight', source_commit=args.frozen_head,
        payload_sha256=hashes, app_created=False, gpu_requested=False), indent=2))
