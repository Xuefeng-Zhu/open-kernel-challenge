"""CPU-only proposed NCU capture plan; nothing here executes a command.

The stable family selector chooses the forward mainloop. It does not establish
cross-demangler graph ABI identity. Capture/import artifact parity is a separate
gate, and the prior empty capture remains a failure.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

FAMILY = r'.*FlashAttnFwdSm90.*'
SELECTOR = 'regex:'+FAMILY
METRICS = (
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
)
MAX_REPORT = 64*1024*1024
REQUIRED_BINDING_KEYS = frozenset((
    'selected_sha256', 'reference_sha256', 'wrapper_sha256', 'binary_sha256',
    'fixture.qkv', 'fixture.positions', 'fixture.cache_loc', 'fixture.cache_seqlens',
    'fixture.cu_seqlens_q', 'fixture.k_cache', 'fixture.v_cache', 'fixture.k_scale',
    'fixture.v_scale', 'fixture.page_table', 'fixture.seq_lens',
))


def report_path(value):
    path = Path(value)
    if not path.is_absolute() or path.suffix != '.ncu-rep':
        raise ValueError('An absolute durable .ncu-rep path is required')
    return str(path)


def capture_command(ncu, python, worker, report):
    """Proposed argv only; the runner must refuse an existing report path."""
    if not Path(worker).is_absolute() or not Path(python).is_absolute():
        raise ValueError('Absolute worker and Python paths are required')
    return [str(ncu), '--config-file', '0', '--replay-mode', 'kernel',
        '--target-processes', 'application-only', '--kernel-name-base', 'demangled',
        '--kernel-name', SELECTOR, '--launch-skip', '1', '--launch-count', '1',
        '--metrics', ','.join(METRICS), '--rename-kernels', '0',
        '--print-kernel-base', 'demangled', '--export', report_path(report),
        '--import-sass', 'off', '--page', 'raw', '--csv', '--clock-control', 'none',
        str(python), str(worker), '--counter-child']


def import_command(ncu, report):
    return [str(ncu), '--config-file', '0', '--import', report_path(report),
        '--rename-kernels', '0', '--print-kernel-base', 'demangled',
        '--page', 'raw', '--csv']


def mangled_import_command(ncu, report):
    """Optional offline evidence only; not a replacement for demangled parity."""
    result = import_command(ncu, report)
    result[result.index('--print-kernel-base')+1] = 'mangled'
    return result


def matches_family(name):
    return isinstance(name, str) and re.fullmatch(FAMILY, name) is not None


def has_metric_csv_header(text):
    required = {'ID', 'Kernel Name', 'Grid Size', 'Block Size', *METRICS}
    for line in text.splitlines():
        try:
            row = next(csv.reader(io.StringIO(line), strict=True))
        except (csv.Error, StopIteration):
            continue
        if required <= set(row):
            return True
    return False


def valid_report_meta(meta):
    return isinstance(meta, dict) and type(meta.get('bytes')) is int \
        and 0 < meta['bytes'] <= MAX_REPORT and isinstance(meta.get('sha256'), str) \
        and re.fullmatch(r'[0-9a-f]{64}', meta['sha256']) is not None


def capture_prerequisites(returncode, stdout, meta):
    empty_warning = re.search(r'No kernels were profiled', stdout, re.I) is not None
    checks = dict(exit_zero=returncode == 0, no_empty_capture_warning=not empty_warning,
        csv_header_present=has_metric_csv_header(stdout), durable_report_present=valid_report_meta(meta))
    return dict(passed=all(checks.values()), **checks,
        permission_denial_observed='ERR_NVGPUCTRPERM' in stdout,
        qualification='Exit zero or passing target correctness alone cannot prove a capture.')


def _valid_row(row):
    if not isinstance(row, dict) or row.get('row_count') != 1 or not matches_family(row.get('kernel_name')):
        return False
    for name in ('grid', 'block'):
        if not isinstance(row.get(name), list) or len(row[name]) != 3 \
                or not all(type(n) is int and n > 0 for n in row[name]):
            return False
    metrics = row.get('metrics', {})
    if set(metrics) != set(METRICS):
        return False
    for value in metrics.values():
        if not isinstance(value, dict):
            return False
        n = value.get('value')
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) \
                or n < 0 or value.get('unit') != '%':
            return False
    return True


def capture_import_gate(capture, imported, expected_identity):
    """Accept artifact parity and explicitly frozen native identity only.

    Rows must come from the existing strict CSV parser. States contain argv-
    bound absolute report_path, report metadata, returncode, stdout, row and
    bindings. Expected identity must be frozen from saved evidence beforehand.
    Neither artifact parity nor native binding proves graph ABI/scheduler state.
    """
    pre = capture_prerequisites(capture.get('returncode'), capture.get('stdout', ''), capture.get('report'))
    left, right = capture.get('row'), imported.get('row')
    rows_valid = _valid_row(left) and _valid_row(right)
    expected = expected_identity if isinstance(expected_identity, dict) else {}
    bindings = expected.get('bindings', {})
    expected_valid = _valid_row(dict(expected, row_count=1,
        metrics={m: {'value': 0, 'unit': '%'} for m in METRICS})) \
        and isinstance(bindings, dict) and set(bindings) == REQUIRED_BINDING_KEYS \
        and all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
            for value in bindings.values())
    checks = dict(capture_prerequisites_passed=pre['passed'], import_exit_zero=imported.get('returncode') == 0,
        import_csv_header_present=has_metric_csv_header(imported.get('stdout', '')),
        no_import_empty_warning=re.search(r'No kernels were profiled', imported.get('stdout', ''), re.I) is None,
        same_absolute_report_path=isinstance(capture.get('report_path'), str)
            and Path(capture['report_path']).is_absolute()
            and capture['report_path'].endswith('.ncu-rep')
            and capture['report_path'] == imported.get('report_path'),
        report_unchanged=valid_report_meta(imported.get('report'))
            and capture.get('report') == imported.get('report'), rows_valid=rows_valid,
        expected_identity_valid=bool(expected_valid),
        expected_native_signature_and_launch=bool(rows_valid and expected_valid and all(
            row['kernel_name'] == expected['kernel_name'] and row['grid'] == expected['grid']
            and row['block'] == expected['block'] for row in (left, right))),
        expected_source_binary_fixture_bindings=bool(expected_valid
            and capture.get('bindings') == bindings and imported.get('bindings') == bindings),
        exact_ncu_signature_equal=bool(rows_valid and left['kernel_name'] == right['kernel_name']),
        launch_equal=bool(rows_valid and left['grid'] == right['grid'] and left['block'] == right['block']),
        metric_values_units_equal=bool(rows_valid and all(
            left['metrics'][m]['value'] == right['metrics'][m]['value']
            and left['metrics'][m]['unit'] == right['metrics'][m]['unit'] for m in METRICS)))
    return dict(accepted=all(checks.values()), **checks,
        acceptance_scope='NCU artifact parity and frozen native signature/launch/source bindings only',
        graph_abi_equivalence='UNRESOLVED: requires a stable mangled/binary binding, not family membership',
        scheduler_split_equivalence='UNKNOWN', execution_state='plan_only_not_executed')


def graph_name_gate(graph_name, ncu_name):
    return dict(raw_string_equal=graph_name == ncu_name,
        both_match_forward_family=matches_family(graph_name) and matches_family(ncu_name),
        cross_demangler_equivalence='UNRESOLVED', scheduler_split_equivalence='UNKNOWN',
        qualification='No casts, booleans, namespaces or T1::Params are normalized away.')


def saved_regression_fixtures(result_file):
    """Extract real spellings and the failed empty-capture log without edits."""
    raw = Path(result_file).read_bytes()
    result = json.loads(raw)
    graph = result['graph_activity']['full_mainloop_symbols'][0]
    command = next(c for c in result['commands'] if '--export' in c['argv'])
    text = command.get('stdout', '')
    available = []
    in_list = False
    for line in text.splitlines():
        if line.strip() == 'Available Kernels:':
            in_list = True
            continue
        if in_list:
            match = re.match(r'^\s*\d+\.\s+(.*)$', line)
            if match:
                available.append(match.group(1))
    native = [name for name in available if matches_family(name)]
    if len(native) != 1:
        raise ValueError('Saved available list does not identify one FA3 family spelling')
    graph_kernels = [row['name'] for row in result['graph_activity']['events'] if row['category'] == 'kernel']
    cpu = result['cpu_file_identity']
    bindings = dict(selected_sha256=result['verified_hashes']['selected.py'],
        reference_sha256=result['verified_hashes']['reference.py'],
        wrapper_sha256=next(row['sha256'] for row in cpu['files']
            if row['path'].endswith('/sgl_kernel/flash_attn.py')),
        binary_sha256=next(row['sha256'] for row in cpu['binary_files']
            if row['path'].endswith('/sgl_kernel/flash_ops.abi3.so')))
    bindings.update({'fixture.'+name: row['sha256'] for name, row in result['fixture']['tensors'].items()})
    resource, = result['graph_activity']['mainloop_resources']
    return dict(result_sha256=hashlib.sha256(raw).hexdigest(), graph_name=graph,
        graph_characters=len(graph), ncu_available_name=native[0], ncu_available_characters=len(native[0]),
        available_kernel_names=available, graph_kernel_names=graph_kernels,
        failed_stdout=text, failed_returncode=command['returncode'], report_meta=result.get('ncu_report_artifact'),
        expected_native_identity=dict(kernel_name=native[0], grid=resource['grid'],
            block=resource['block'], bindings=bindings),
        child_correctness=[row for row in result.get('counter_child_progress', [])
            if row.get('phase') in ('pre', 'measured', 'post')])


def no_filter_assessment(fixtures):
    return dict(safe_for_current_child=False,
        observed_graph_kernels=len(fixtures['graph_kernel_names']),
        observed_available_kernels=len(fixtures['available_kernel_names']),
        reason='The selected call contains preparation, scheduler, mainloop, combine and norm/gate. '
            'Input generation and original-oracle Torch kernels also run in the current child. '
            'An unfiltered ordinal launch cannot safely select the FA3 mainloop.',
        required_selection='Help-supported stable FA3 family regex plus exactly one validated captured row.')
