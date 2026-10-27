"""Standalone single-counter capability gate. No FA3, graph, NVTX or auth."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import traceback

HELPER_SHA = 'd984cbd44691cfead1c889cbb2ae8b0bde2e11e0a1d703bbdd78a7ecf1b30637'
FIXED = {'counter_helper.py': HELPER_SHA,
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(root):
    path = root/'counter_helper.py'
    if path.is_symlink() or sha(path) != HELPER_SHA:
        raise ValueError('Frozen counter helper differs')
    spec = importlib.util.spec_from_file_location('eager_counter_helper', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile_argv(ncu, root, metric):
    return [str(ncu), '--replay-mode', 'kernel', '--target-processes',
        'application-only', '--kernel-name-base', 'demangled', '--kernel-name',
        'regex:.*okc_counter_gate.*', '--launch-skip', '1', '--launch-count', '1',
        '--metrics', metric, '--page', 'raw', '--csv',
        '--clock-control', 'none', sys.executable, str(root/'worker.py'), '--tiny-child']


def validate_installed_help(help_text, argv, query_argv=None):
    """Check actual installed CLI grammar before any profiling target starts."""
    options = argv[1:argv.index(sys.executable)] + (query_argv or [])[1:]
    flags = sorted({value for value in options if value.startswith('--')})
    if '--units' in flags:
        raise ValueError('Known unsupported --units must never be used')
    supported = set(re.findall(r'(?<![\w-])--[a-z][a-z0-9-]*', help_text))
    missing = sorted(set(flags)-supported)
    if missing:
        raise ValueError('Installed NCU help does not advertise flags: '+', '.join(missing))
    for value in ('application-only', 'demangled', 'kernel', 'suffix'):
        if value not in help_text:
            raise ValueError('Installed help does not advertise required choice: '+value)
    return {'used_ncu_flags': flags, 'all_flags_advertised': True,
        'known_unsupported_units_absent': True,
        'help_sha256': hashlib.sha256(help_text.encode()).hexdigest(),
        'csv_base_units': 'No units override is requested; preserve raw CSV values/unit fields'}


def progress_rows(text):
    rows = []
    for line in text.splitlines():
        if line.startswith('TINY_PROGRESS_JSON='):
            rows.append(json.loads(line.split('=', 1)[1]))
    return rows


def classify(command, helper):
    out, err = command.get('stdout', ''), command.get('stderr', '')
    rows = progress_rows(out)
    checks = [row for row in rows if row.get('phase') in ('pre', 'measured', 'post')]
    correctness = [row.get('phase') for row in checks] == ['pre', 'measured', 'post'] \
        and all(row.get('passed') is True for row in checks)
    values = helper.metric_values(out)
    denial = 'ERR_NVGPUCTRPERM' in out+'\n'+err
    passed = command['state'] == 'complete' and command['returncode'] == 0 \
        and correctness and len(values) == 1
    return dict(passed=passed, counter_values=values, child_progress=rows,
        pre_measured_post_correctness=correctness, permission_denial_observed=denial,
        command_state=command['state'],
        interpretation='Capability for one tiny eager kernel and one metric only; no FA3 bottleneck or performance inference')


def tiny_child(root):
    def emit(phase, **extra):
        print('TINY_PROGRESS_JSON='+json.dumps(dict(phase=phase, **extra)), flush=True)
    emit('child_started')
    helper = load_helper(root)
    emit('helper_verified')
    emit('vendor_import_start')
    import torch
    import triton
    emit('vendor_import_complete', torch=torch.__version__, triton=triton.__version__)
    emit('cuda_init_start')
    torch.cuda.init()
    emit('cuda_init_complete')
    for phase in ('pre', 'measured', 'post'):
        emit(phase+'_starting')
        passed = helper._tiny()
        emit(phase, passed=passed)
        if not passed:
            return 1
    return 0


def run(root, output, seconds):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='Standalone deterministic eager single-counter capability',
        status='starting', passed=False, worker_sha256=sha(__file__), commands=[],
        hardware_counter_sessions=0, phase='source_binding',
        selected_candidate_changed=False, fa3_kernel_executed=False,
        cuda_graph=False, nvtx=False, application_replay=False,
        security_changes=False, credential_actions=False)
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        raw = json.dumps(report, indent=2)+'\n'
        if len(raw.encode()) > 1024*1024:
            raise ValueError('Summary exceeds bound')
        path = output/'summary.tmp'
        path.write_text(raw)
        path.replace(output/'summary.json')
    def mark(phase):
        report['phase'] = phase
        report.setdefault('progress', []).append(dict(phase=phase,
            utc=datetime.now(timezone.utc).isoformat(), elapsed=time.monotonic()-started))
        save()
    def alarm(*_):
        raise TimeoutError('65-second worker deadline')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(seconds)
    save()
    try:
        report['verified_hashes'] = {name: sha(root/name) for name in FIXED}
        if report['verified_hashes'] != FIXED:
            raise ValueError('Payload hash differs')
        helper = load_helper(root)
        report['metric'] = helper.METRIC
        ncu = shutil.which('ncu')
        report['ncu_path'] = ncu
        if not ncu:
            report.update(status='unavailable', reason='No existing NCU on PATH')
            save()
            return 1
        mark('installed_version_start')
        version = helper._command([ncu, '--version'], output, 'version', 3)
        report['commands'].append(version)
        mark('installed_help_start')
        help_result = helper._command([ncu, '--help'], output, 'help', 5)
        report['commands'].append(help_result)
        if version['state'] != 'complete' or help_result['state'] != 'complete':
            raise ValueError('Installed version/help command failed; no profiling permitted')
        report['installed_cli_contract'] = validate_installed_help(
            help_result.get('stdout', ''), profile_argv(ncu, root, helper.METRIC),
            helper.query_argv(ncu))
        mark('installed_help_validated')
        mark('query_metric_start')
        query = helper._command(helper.query_argv(ncu), output, 'query', 5)
        report['commands'].append(query)
        mark('query_metric_terminal')
        if query['state'] != 'complete' or not helper.metric_supported(query.get('stdout', '')):
            report.update(status='query_failed_or_unsupported', reason='One fixed hardware metric not confirmed')
            save()
            return 1
        mark('eager_counter_session_start')
        report['hardware_counter_sessions'] = 1
        save()
        command = helper._command(profile_argv(ncu, root, helper.METRIC), output, 'eager-counter', 30)
        report['commands'].append(command)
        report['capability'] = classify(command, helper)
        mark('eager_counter_session_terminal')
        report.update(status='complete' if report['capability']['passed'] else 'capability_not_established',
            passed=report['capability']['passed'])
        save()
        return 0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='incomplete_or_failed', error_type=type(exc).__name__,
            error=str(exc)[:1000], traceback=traceback.format_exc()[-4000:])
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny-child', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--deadline', type=int, default=65)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.tiny_child:
        raise SystemExit(tiny_child(root))
    if args.output_dir is None or not 1 <= args.deadline <= 65:
        parser.error('Output directory and deadline1..65 required')
    raise SystemExit(run(root, args.output_dir, args.deadline))
