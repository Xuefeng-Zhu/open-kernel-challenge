"""Bounded CPU-only import of the already retained report; no profiling target.

Uses existing authentication and image, no GPU allocation, install or security
change. Original native result and frozen gate are never rewritten.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import warnings

HERE = Path(__file__).resolve().parent
REPORT_SHA = '4104eaf52a0f44bbfc03703af2355902d16cb1f19bebde4816c3b87a58a3b6b6'


def load_controller():
    spec = importlib.util.spec_from_file_location('cpu_export_guard', HERE/'controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_existing_report():
    import hashlib, re, shutil, subprocess
    from pathlib import Path
    path = Path('/opt/attention-retained-report/baseline.ncu-rep')
    expected = '4104eaf52a0f44bbfc03703af2355902d16cb1f19bebde4816c3b87a58a3b6b6'
    commands = []
    def call(argv, seconds):
        try:
            child = subprocess.run(argv, capture_output=True, text=True, timeout=seconds)
        except subprocess.TimeoutExpired as error:
            def text(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            commands.append(dict(argv=argv, returncode=None, state='timeout',
                stdout=text(error.stdout)[:256*1024], stderr=text(error.stderr)[:64*1024]))
            raise
        if len(child.stdout.encode()) > 256*1024 or len(child.stderr.encode()) > 64*1024:
            raise ValueError('Offline command output exceeds fixed bound')
        result = dict(argv=argv, returncode=child.returncode,
            stdout=child.stdout, stderr=child.stderr)
        commands.append(result)
        if child.returncode:
            raise ValueError('Offline NCU command failed: '+str(child.returncode))
        return result
    phase = 'report_validation'
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size != 35756:
            raise ValueError('Retained report file/size mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Retained report digest mismatch')
        ncu = shutil.which('ncu')
        if ncu is None:
            raise ValueError('Existing NCU missing; no install')
        phase = 'version_help'
        version = call([ncu, '--config-file', '0', '--version'], 3)
        help_result = call([ncu, '--config-file', '0', '--help'], 5)
        argv = [ncu, '--config-file', '0', '--import', str(path), '--rename-kernels', '0',
            '--print-kernel-base', 'demangled', '--print-units', 'base', '--print-fp',
            '--page', 'raw', '--csv']
        for flag in (arg for arg in argv if arg.startswith('--')):
            if not re.search(r'(?<![\w-])'+re.escape(flag)+r'(?![\w-])', help_result['stdout']):
                raise ValueError('Installed help lacks offline option: '+flag)
        phase = 'first_import'
        first = call(argv, 10)
        phase = 'second_import'
        second = call(argv, 10)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if after != expected:
            raise ValueError('Report changed during offline import')
        return dict(kind='CPU-only retained-report imports', status='complete', passed=True,
            gpu_requested=False, profiling_target_launched=False, report_bytes=path.stat().st_size,
            report_before_sha256=expected, report_after_sha256=after,
            version=version['stdout'], commands=commands,
            canonical_stdout_identical=first['stdout'] == second['stdout'])
    except Exception as error:
        return dict(kind='CPU-only retained-report imports', status='failed', passed=False,
            phase=phase, error_type=type(error).__name__, error=str(error)[:1200],
            gpu_requested=False, profiling_target_launched=False, commands=commands,
            expected_report_sha256=expected,
            report_after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file() and not path.is_symlink() and path.stat().st_size <= 64*1024 else None)


def launch():
    c = load_controller()
    ledger, result_path = HERE/'offline-export.local.json', HERE/'OFFLINE_EXPORT.json'
    if ledger.exists() or result_path.exists():
        raise ValueError('Existing CPU export request: reconcile, never duplicate')
    report = HERE/'artifacts/baseline.ncu-rep'
    if report.stat().st_size != 35756 or hashlib.sha256(report.read_bytes()).hexdigest() != REPORT_SHA:
        raise ValueError('Retained report differs')
    c.configure_existing_auth()
    auth = c.verify_existing_auth()
    billing = c.cli('billing', 'summary', '--json')
    cap = json.loads((HERE/'billing.local.json').read_text())
    free_floor = c.validate_budget(cap, billing)
    apps = c.cli('app', 'list', '--env', 'main', '--json')
    if not c.stopped(apps):
        raise ValueError('Other apps active; CPU export not started')
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=ServerWarning)
    app = modal.App('okc-attention-retained-report-cpu', include_source=False)
    image = modal.Image.from_id(c.IMAGE).add_local_file(report,
        '/opt/attention-retained-report/baseline.ncu-rep', copy=False)
    fn = app.function(image=image, gpu=None, cpu=(1, 1), memory=(2048, 2048),
        min_containers=0, max_containers=1, buffer_containers=0,
        timeout=35, startup_timeout=20, retries=0, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(import_existing_report)
    record = dict(kind='CPU-only retained-report export', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=REPORT_SHA, auth=auth, billing_before=billing,
        conservative_free_credit_floor_usd=str(free_floor), gpu=None,
        timeout_seconds=35, watchdog_seconds=60, app_id=None)
    c.save(ledger, record)
    lock = threading.Lock()
    def stop():
        with lock:
            if app.app_id and not record.get('stop_requested'):
                record['stop_requested'] = True
                c.cli('app', 'stop', app.app_id, '--yes', '--env', 'main')
    timer = threading.Timer(60, stop)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name='main'):
            record.update(app_id=app.app_id, admitted_utc=datetime.now(timezone.utc).isoformat())
            c.save(ledger, record)
            print('Accepted CPU-only report import '+app.app_id, flush=True)
            result = fn.remote()
            c.save(result_path, result)
            record['result_retrieved'] = True
            record['result_passed'] = result.get('passed') is True
            stop()
    finally:
        stop()
        timer.cancel()
        states = c.cli('app', 'list', '--env', 'main', '--json')
        row = next((r for r in states if r.get('app_id') == app.app_id), None)
        record['cleanup_observation'] = row
        record['stopped_zero_tasks'] = bool(row and c.stopped([row]))
        after = c.cli('billing', 'summary', '--json')
        record['billing_after'] = after
        record['observed_account_delta_usd'] = str(max(Decimal(0), c.gross(after)-c.gross(billing)))
        c.save(ledger, record)
        print(json.dumps({k:record.get(k) for k in ('app_id','result_retrieved','stopped_zero_tasks','observed_account_delta_usd')}))
        if Decimal(after['billed_cost']) != 0:
            raise ValueError('Paid-$0 invariant failed')
    return 0 if record.get('result_retrieved') and record.get('result_passed') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    if parser.parse_args().launch:
        raise SystemExit(launch())
    print('Offline by default; no service called. --launch imports only the existing report on CPU.')
