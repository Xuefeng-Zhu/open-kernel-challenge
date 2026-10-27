"""One bounded eager/kernel-replay NCU capability check, then one FA3 target.

No graph/NVTX/application replay, installs, permission changes, metric fallback
or credential access. Importing this module uses only the Python standard library.
CLI query syntax: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

METRIC_BASE = 'sm__cycles_elapsed'
METRIC = 'sm__cycles_elapsed.avg'
LOG_LIMIT = 128 * 1024
SOURCE_SHA = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
FILTERS = {'tiny': 'regex:.*okc_counter_gate.*',
           'selected': 'regex:.*FlashAttnFwdSm90.*'}


def metric_supported(text):
    return bool(re.search(r'(?<![\w.])' + re.escape(METRIC) + r'(?![\w.])', text))


def metric_values(text):
    """Read positive finite values from either long or wide NCU raw CSV.

    Never treat a command echo, metric-name occurrence or child success alone
    as evidence that the hardware metric was actually collected.
    """
    rows = list(csv.reader(io.StringIO(text)))
    values = []
    for index, row in enumerate(rows):
        if 'Metric Name' in row and 'Metric Value' in row:
            ni, vi = row.index('Metric Name'), row.index('Metric Value')
            for later in rows[index+1:]:
                if len(later) > max(ni, vi) and later[ni] == METRIC:
                    values.append(later[vi])
        elif METRIC in row:
            ci = row.index(METRIC)
            for later in rows[index+1:]:
                if len(later) > ci and later and later[0].isdigit():
                    values.append(later[ci])
    parsed = []
    for value in values:
        try:
            number = float(value.replace(',', ''))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            parsed.append(number)
    return parsed


def query_argv(ncu):
    return [str(ncu), '--query-metrics', '--query-metrics-mode', 'suffix',
            '--metrics', METRIC_BASE]


def profile_argv(ncu, root, mode):
    if mode not in FILTERS:
        raise ValueError('Unknown fixed counter target')
    return [str(ncu), '--replay-mode', 'kernel', '--target-processes',
        'application-only', '--launch-count', '1',
        '--kernel-name-base', 'demangled', '--kernel-name', FILTERS[mode],
        '--metrics', METRIC, '--page', 'raw', '--csv', '--units', 'base',
        '--clock-control', 'none', sys.executable, str(Path(root)/'counter_gate.py'),
        '--child', mode, '--root', str(root)]


def _command(argv, dest, stem, seconds):
    """Retain complete bounded streams; terminate the own process group on timeout."""
    out, err = dest/(stem+'.stdout.txt'), dest/(stem+'.stderr.txt')
    started = time.monotonic()
    result = {'argv': argv, 'timeout_seconds': seconds, 'state': 'not_started'}
    if out.exists() or err.exists():
        raise ValueError('Existing counter logs: never duplicate a request')
    with out.open('xb') as so, err.open('xb') as se:
        proc = subprocess.Popen(argv, stdout=so, stderr=se, cwd=dest,
                                start_new_session=True)
        result['state'] = 'running'
        while proc.poll() is None:
            timed_out = time.monotonic()-started >= seconds
            oversized = out.stat().st_size > LOG_LIMIT or err.stat().st_size > LOG_LIMIT
            if timed_out or oversized:
                result['state'] = 'timeout' if timed_out else 'output_limit'
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=2)
                break
            time.sleep(.025)
        result.update(returncode=proc.returncode,
            wall_seconds=time.monotonic()-started, own_process_group_stopped=proc.poll() is not None)
        if result['state'] == 'running':
            result['state'] = 'complete' if proc.returncode == 0 else 'nonzero_exit'
    for label, path in (('stdout', out), ('stderr', err)):
        raw = path.read_bytes()
        result[label+'_bytes'] = len(raw)
        result[label+'_sha256'] = hashlib.sha256(raw).hexdigest()
        # If the producer exceeded its bound, preserve file/hash but do not
        # truncate it and misrepresent the excerpt as the complete stream.
        if len(raw) <= LOG_LIMIT:
            result[label] = raw.decode('utf-8', errors='replace')
        else:
            result[label+'_omitted'] = 'full stream exceeds bound; file retained remotely'
    return result


def _child_passed(text, mode):
    marker = 'COUNTER_GATE_CHILD_JSON='
    found = []
    for line in text.splitlines():
        if line.startswith(marker):
            try:
                row = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                continue
            found.append(row)
    return len(found) == 1 and found[0].get('mode') == mode and found[0].get('passed') is True


def run_gate(root, outdir, remaining_seconds):
    root, dest = Path(root).resolve(), Path(outdir).resolve()
    report = {'state': 'not_attempted', 'metric': METRIC,
        'replay_mode': 'kernel', 'cuda_graph': False, 'nvtx': False,
        'application_replay': False, 'security_changes': False,
        'hardware_counter_sessions': 0, 'commands': [],
        'permission_denial_observed': False,
        'qualification': 'A timeout is not permission denial. Counter timing is instrumented, not whole-call benchmark latency.'}
    if remaining_seconds < 50:
        report['reason'] = 'Less than 50 seconds remain after core identity/activity correctness stages'
        return report
    ncu = shutil.which('ncu')
    if not ncu:
        report['reason'] = 'NCU executable not present on PATH'
        return report
    for name, expected in SOURCE_SHA.items():
        path = root/name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Counter target source binding mismatch: '+name)
    query = _command(query_argv(ncu), dest, 'counter-gate-query', 5)
    report['commands'].append(query)
    if query['state'] != 'complete' or not metric_supported(query.get('stdout', '')):
        report.update(state='unsupported_or_query_failed', reason='The one fixed hardware metric was not confirmed by the bounded query')
        return report
    for mode in ('tiny', 'selected'):
        command = _command(profile_argv(ncu, root, mode), dest, 'counter-gate-'+mode, 20)
        report['commands'].append(command)
        report['hardware_counter_sessions'] += 1
        combined = command.get('stdout', '')+'\n'+command.get('stderr', '')
        denial = 'ERR_NVGPUCTRPERM' in combined
        report['permission_denial_observed'] |= denial
        values = metric_values(command.get('stdout', ''))
        passed = (command['state'] == 'complete' and command['returncode'] == 0
            and bool(values) and _child_passed(command.get('stdout', ''), mode))
        report[mode] = {'passed': passed, 'counter_values': values,
            'explicit_permission_denial': denial, 'state': command['state']}
        if not passed:
            report.update(state=mode+'_failed', reason='No complete positive counter plus child correctness result; later targets not attempted')
            return report
    report.update(state='complete', reason='One queried hardware counter collected for deterministic eager kernel and unchanged selected FA3 target')
    return report


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tiny():
    global tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def okc_counter_gate(X, Y):
        offsets = tl.arange(0, 256)
        x = tl.load(X+offsets)
        tl.store(Y+offsets, x*2.+1.)

    x = torch.arange(256, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    okc_counter_gate[(1,)](x, y, num_warps=4)
    torch.cuda.synchronize()
    return bool(torch.equal(y, x*2.+1.))


def _selected(root):
    import torch
    sys.path.insert(0, str(root))
    for name, expected in SOURCE_SHA.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Selected child source mismatch')
    reference = _load(root/'reference.py', 'counter_original_reference')
    selected = _load(root/'selected.py', 'counter_unchanged_selected')
    data = reference.generate_input(batch=32, q_len=8, ctx=80000, heads='tp8',
        layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent',
        seed=135741, device='cuda')
    output = selected.kernel_fn(data)
    torch.cuda.synchronize()
    ok, _ = reference.check_implementation(data, reference.collect(data, output))
    return bool(ok)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=('tiny', 'selected'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    passed = _tiny() if args.child == 'tiny' else _selected(args.root.resolve())
    print('COUNTER_GATE_CHILD_JSON='+json.dumps({'mode': args.child, 'passed': passed}), flush=True)
    raise SystemExit(0 if passed else 1)
