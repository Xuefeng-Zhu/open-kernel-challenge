"""Single source-bound physical page locality diagnostic; offline by default.

Only the explicit flattened source allowlist is uploaded. Existing credentials
stay in the protected local config and never enter the remote closure/payload.
"""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
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
REPO = HERE.parents[2]
ATTENTION = HERE.parent
ORIGINAL = Path('/Users/frank/Documents/Codex/2026-10-23/task')
AUTH = ORIGINAL/'local-notes/modal-auth.toml'
PYTHON = ORIGINAL/'.venv/bin/python'
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
EXPERIMENT = 'attention-locality-b32l8-probe'
WORKER_SECONDS, CHILD_SECONDS = 160, 175
FUNCTION_SECONDS, STARTUP_SECONDS, WATCHDOG_SECONDS = 190, 20, 220
MAX_RESULT, MAX_TRACE, MAX_GZIP = 2*1024*1024, 32*1024*1024, 8*1024*1024
MAX_TRACE_ENCODING = 4*((MAX_GZIP+2)//3)
RESERVE = Decimal('.70')
EXPECTED = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
    'fixture.py': '203240b0f4de073b4c6222c62daeeaff20ce0b3de32b519f8d9f59c5528e0e28',
    'trace_helpers.py': 'd89e2f6dac4d8afd9105b713ce255407df8e9253c330a1e727d8193a220c5ec8',
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
    return {'worker.py': HERE/'worker.py', 'fixture.py': HERE/'fixture.py',
        'selected.py': ATTENTION/'fusion-lookahead-probe/candidate.py',
        'reference.py': ATTENTION/'official/reference.py',
        'checks.py': ATTENTION/'official/checks.py',
        'trace_helpers.py': ATTENTION/'fa3-next-evidence/baseline-runtime-identity/analyze_partial.py',
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
        import base64, hashlib, os, signal, subprocess, sys, time
        from pathlib import Path
        root, output = Path('/opt/attention-locality-b32l8'), Path('/tmp/attention-locality-b32l8')
        log_path = Path('/tmp/attention-locality-b32l8-worker.log')
        started = time.monotonic()
        with log_path.open('wb') as log:
            child = subprocess.Popen([sys.executable, str(root/'worker.py'), '--execute', '--activity',
                '--deadline', '160', '--payload-root', str(root), '--output', str(output/'summary.json')], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=175)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=3)
                code = 124
        reply = {'exit_code': code, 'worker_wall_seconds': time.monotonic()-started}
        summary = output/'summary.json'
        if summary.is_file() and not summary.is_symlink() and summary.stat().st_size <= 2*1024*1024:
            raw = summary.read_bytes()
            reply.update(report_json=raw.decode('utf-8'), report_sha256=hashlib.sha256(raw).hexdigest())
        else:
            reply['error'] = 'Missing or oversized summary'
        trace, compressed = output/'activity-trace.json', output/'activity-trace.json.gz'
        if trace.exists() or compressed.exists():
            if (not trace.is_file() or trace.is_symlink() or trace.stat().st_size > 32*1024*1024
                    or not compressed.is_file() or compressed.is_symlink() or compressed.stat().st_size > 8*1024*1024):
                reply['artifact_error'] = 'Trace absent or exceeds fixed raw/compressed limit'
            else:
                raw, packed = trace.read_bytes(), compressed.read_bytes()
                encoding = base64.b64encode(packed).decode('ascii')
                if len(encoding) > 4*((8*1024*1024+2)//3):
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


def verified_result(reply):
    if not isinstance(reply, dict) or type(reply.get('exit_code')) is not int:
        raise ValueError('Invalid remote reply')
    if not isinstance(reply.get('report_json'), str):
        return None, None, None
    raw = reply['report_json'].encode('utf-8')
    if len(raw) > MAX_RESULT or hashlib.sha256(raw).hexdigest() != reply.get('report_sha256'):
        raise ValueError('Summary source/size digest mismatch')
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError('Summary must be an object')
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
    return raw, report, artifact


def worker_protocol(expected_sha256=None):
    # CPU-only module import: vendor imports live inside execute(), never here.
    import importlib.util
    worker = HERE/'worker.py'
    if not worker.is_file() or worker.is_symlink():
        raise ValueError('Local protocol worker must be a regular nonsymlink file')
    if expected_sha256 is not None and hashlib.sha256(worker.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Local protocol worker differs from frozen source hash')
    spec = importlib.util.spec_from_file_location('locality_controller_protocol', worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_report(report, hashes, artifact):
    """Completion and scientific KEEP/DROP are independent; neither is a score."""
    if not isinstance(report, dict):
        raise ValueError('No diagnostic summary retrieved')
    expected = {name: digest for name, digest in hashes.items()
        if name not in ('controller.py', 'worker.py')}
    if report.get('worker_sha256') != hashes['worker.py'] or report.get('verified_hashes') != expected:
        raise ValueError('Retrieved worker/source bindings differ from frozen payload')
    if report.get('passed') is not True or report.get('status') != 'complete':
        return False
    if artifact is None:
        raise ValueError('Complete activity diagnostic must retain its raw trace')
    activity = report.get('activity', {})
    manifest = artifact['manifest']
    wanted_artifact = dict(name='activity-trace.json', gzip_name='activity-trace.json.gz', **manifest)
    if activity.get('artifact') != wanted_artifact:
        raise ValueError('Summary trace binding differs from retained artifact')
    w = worker_protocol(hashes['worker.py'])
    protocol = report.get('protocol', {})
    if (protocol.get('cases') != list(w.CASES) or protocol.get('seeds') != list(w.SEEDS)
            or protocol.get('arms') != list(w.ARMS) or protocol.get('expected_counts') != w.expected_counts(True)
            or protocol.get('warmup_rounds_per_fixture') != 4 or protocol.get('reverse_pairs_per_fixture') != 6
            or protocol.get('cold_l2_eviction_bytes') != 256*1024*1024):
        raise ValueError('Declared single-case/two-fixture replay protocol changed')
    for name in ('same_graph_per_case', 'private_default_capture_pools', 'raw_whole_call',
            'page_table_always_present', 'provided_decode_metadata_preserved', 'diagnostic_only'):
        if protocol.get(name) is not True:
            raise ValueError('Required diagnostic protocol invariant: '+name)
    if protocol.get('replay_cost_subtracted') is not False or protocol.get('candidate_source_changed') is not False:
        raise ValueError('Timed call or candidate changed')
    for name in ('selected_candidate_changed', 'official_check', 'formal_entry'):
        if report.get(name) is not False:
            raise ValueError('Diagnostic-only boundary failed: '+name)
    if report.get('hardware_counter_sessions') != 0:
        raise ValueError('No hardware counter session admitted')
    if (report.get('correctness_complete') is not True or report.get('timing_complete') is not True
            or report.get('default_graph_pool_handles_distinct') is not True):
        raise ValueError('Correctness/timing/graph completion incomplete')
    counts = report.get('counts', {})
    expected_counts = w.expected_counts(True)
    required_counts = ('original_oracle_calls', 'warmup_checked_replays', 'measured_checked_replays',
        'activity_checked_replays', 'total_checked_graph_replays', 'original_oracle_calls_attempted',
        'graph_replays_attempted', 'graph_replays_completed')
    if (set(counts) != set(required_counts)
            or any(type(counts[k]) is not int or counts[k] != expected_counts[k] for k in required_counts)):
        raise ValueError('Expected exactly70 checked graph replays and4 original oracles')
    cases = report.get('cases', [])
    if len(cases) != 1 or cases[0].get('case') != w.CASES[0] or cases[0].get('graph_captures') != 1:
        raise ValueError('Expected one B32/L8 full case and one captured graph')
    fixtures = cases[0].get('fixtures', [])
    if len(fixtures) != 2 or [row.get('seed') for row in fixtures] != list(w.SEEDS):
        raise ValueError('Expected two declared fresh fixtures')
    for fixture in fixtures:
        if fixture.get('plan', {}).get('full_capacity_permutation') is not True:
            raise ValueError('Complete physical-cache permutation was not verified')
        oracle = fixture.get('oracle', [])
        if (len(oracle) != 2 or [r.get('arm') for r in oracle] != list(w.ARMS)
                or any(any(r.get(k) is not True for k in ('passed', 'ownership', 'stable_addresses')) for r in oracle)):
            raise ValueError('Original oracle/cache ownership evidence incomplete')
        if fixture.get('post_call_full_cache_inverse_exact') is not True:
            raise ValueError('Inverse full-cache byte equality was not verified')
        if len(fixture.get('warmup', [])) != 4:
            raise ValueError('Expected4 checked warmup rounds per fixture')
        contrast = fixture.get('locality_contrast', {})
        edges = contrast.get('adjacent_attended_pairs', 0)
        if not (contrast.get('moved_attended_history_pages', 0) > 0 and edges > 0
                and contrast.get('canonical_layout_contiguous_adjacent_pairs') == edges
                and 0 <= contrast.get('random_layout_contiguous_adjacent_pairs', edges) < edges):
            raise ValueError('No verified attended historical locality contrast')
    if (activity.get('status') != 'complete' or activity.get('seed') != w.SEEDS[-1]
            or activity.get('sessions') != 1 or activity.get('native_identity_fully_observed') is not True):
        raise ValueError('Native activity identity is incomplete or UNKNOWN')
    groups = activity.get('summary', {}).get('groups', {})
    labels = ['PROFILE_SELECTED_FULL_VERIFY_B32_'+arm.upper() for arm in w.ARMS]
    if set(groups) != set(labels):
        raise ValueError('Expected exactly two graph-correlated activity arms')
    for label in labels:
        kernels = groups[label].get('kernels', [])
        if (not kernels or sum('FlashAttnFwdSm90' in row.get('name', '') for row in kernels) != 1):
            raise ValueError('Expected exactly one observed selected FA3 mainloop per activity arm')
    comparison = w.native_identity_comparison(*(groups[label].get('kernels', []) for label in labels))
    if comparison.get('status') != 'observed_equal':
        raise ValueError('Native symbols/launch metadata differ or are UNKNOWN')
    if activity.get('cross_layout_native_identity') != [dict(case='full_verify_b32', **comparison)]:
        raise ValueError('Claimed activity identity differs from retained kernel metadata')
    # Recompute prospective15%/6-pair gate locally. DROP is valid completion.
    if report.get('decision') != w.threshold_decision(fixtures, True):
        raise ValueError('Scientific KEEP/DROP differs from frozen prospective gate')
    return True


def materialize_result(reply, destination, hashes):
    """Persist bounded raw/partial artifacts before digest and scientific gates."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = reply.get('report_json') if isinstance(reply, dict) else None
    if isinstance(summary, str) and len(summary.encode('utf-8')) <= MAX_RESULT:
        with (destination/'result.json').open('xb') as stream:
            stream.write(summary.encode('utf-8'))
    trace = reply.get('trace') if isinstance(reply, dict) else None
    if isinstance(trace, dict):
        save(destination/'trace-manifest.json', {k:v for k,v in trace.items() if k != 'gzip_base64'})
        encoding = trace.get('gzip_base64')
        if isinstance(encoding, str) and len(encoding) <= MAX_TRACE_ENCODING:
            packed = base64.b64decode(encoding, validate=True)
            if len(packed) <= MAX_GZIP:
                with (destination/'activity-trace.json.gz').open('xb') as stream:
                    stream.write(packed)
    raw, report, artifact = verified_result(reply)
    complete = validate_report(report, hashes, artifact) if report is not None else False
    return dict(report=report, result_retrieved=raw is not None,
        completion_verified=complete and reply['exit_code'] == 0,
        trace_retrieved=artifact is not None, report_sha256=reply.get('report_sha256'))


def observe_shutdown(app_id, attempts=6, interval=2):
    """A stop acknowledgement may precede task teardown; bounded read-only polls."""
    observations = []
    for index in range(attempts):
        states = cli('app', 'list', '--env', 'main', '--json')
        exact = next((row for row in states if row.get('app_id') == app_id), None)
        observations.append(dict(observed_utc=now(), app=exact))
        if exact and stopped([exact]):
            return states, exact, observations
        if index+1 < attempts:
            time.sleep(interval)
    return states, exact, observations


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
        image = image.add_local_file(path, '/opt/attention-locality-b32l8/'+name, copy=False)
    app = modal.App('okc-attention-locality-b32l8', include_source=False)
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
        official_check=False, formal_entry=False, candidate_change=False)
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
            print('Accepted single Attention B32/L8 locality diagnostic '+app.app_id, flush=True)
            if expired.is_set():
                raise TimeoutError('Watchdog expired before invocation')
            reply = fn.remote()
            outcome = materialize_result(reply, HERE, hashes)
            report = outcome.pop('report')
            record.update(state='function-terminal', **outcome,
                exit_code=reply['exit_code'], worker_status=(report or {}).get('status'),
                worker_passed=(report or {}).get('passed'), scientific_decision=(report or {}).get('decision'),
                worker_wall_seconds=reply.get('worker_wall_seconds'),
                error=reply.get('error'), artifact_error=reply.get('artifact_error'),
                worker_log_tail=reply.get('worker_log_tail'))
            save(ledger, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('Locality diagnostic requires reconciliation; do not relaunch.', flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            states, exact, observations = observe_shutdown(app.app_id)
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
