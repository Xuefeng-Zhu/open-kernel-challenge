"""CPU-only end-to-end executor check with an explicitly simulated NCU.

Runs the production collector's subprocess/file path, production remote-entry
packaging and production local materialization. Native NCU/CUDA and the remote
worker process are substituted. No Modal import, credential read or GPU call.
Synthetic report bytes are NOT an Nsight report or new measurement evidence.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'fa3-b32-identity-recapture'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_csv(expected, successful):
    """Synthetic identity plus saved values, only for acceptance regression."""
    stream = io.StringIO()
    stream.write('OFFLINE SIMULATION: NOT A NEW GPU MEASUREMENT\n')
    out = csv.writer(stream)
    metrics = tuple(successful['metrics'])
    out.writerow(['ID', 'Kernel Name', 'Grid Size', 'Block Size', *metrics])
    out.writerow(['', '', '', '', *(successful['metrics'][m]['unit'] for m in metrics)])
    out.writerow(['0', expected['kernel_name'], '(132, 1, 1)', '(256, 1, 1)',
        *(successful['metrics'][m]['raw'] for m in metrics)])
    return stream.getvalue()


FAKE_NCU = '''import json,sys
from pathlib import Path
c=json.loads(Path(__file__).with_suffix('.json').read_text())
a=sys.argv[1:]
if '--export' in a:
    assert a[a.index('--kernel-name')+1]=='regex:.*FlashAttnFwdSm90.*'
    assert a[a.index('--rename-kernels')+1]=='0'
    assert a[a.index('--config-file')+1]=='0'
    if c['mode']=='pass':
        p=Path(a[a.index('--export')+1])
        p.write_bytes(b'OFFLINE_SIMULATED_REPORT_NOT_NCU\\n'+json.dumps({'csv':c['csv']}).encode())
        print(c['csv'],end='')
    else:
        print(c['failed_stdout'],end='')
elif '--import' in a:
    raw=Path(a[a.index('--import')+1]).read_bytes()
    marker=b'OFFLINE_SIMULATED_REPORT_NOT_NCU\\n'
    assert raw.startswith(marker)
    print(json.loads(raw[len(marker):])['csv'],end='')
else:
    raise SystemExit('Unexpected simulated NCU command')
'''


def packaged_reply(controller, payload, output, exit_code):
    """Exercise unchanged entry packaging; replace only paths/native process."""
    real_path = type(HERE)
    remaps = {
        '/opt/attention-fa3-b32-capture-executor': payload,
        '/tmp/attention-fa3-b32-capture-executor': output,
        '/tmp/attention-fa3-b32-capture-executor-worker.log': output/'worker.log',
    }
    launches = []
    def mapped_path(*args, **kwargs):
        path = real_path(*args, **kwargs)
        return real_path(remaps.get(str(path), path))
    class CompletedNativeSubstitute:
        def __init__(self, argv, **kwargs):
            assert str(payload/'worker.py') in argv
            assert argv[argv.index('--output-dir')+1] == str(output)
            launches.append(argv)
            kwargs['stdout'].write(b'OFFLINE native worker substitute; collector already exercised\n')
        def wait(self, timeout=None):
            return exit_code
    with patch('pathlib.Path', mapped_path), patch('subprocess.Popen', CompletedNativeSubstitute):
        reply = controller.make_remote_entry()()
    assert len(launches) == 1
    return reply


def run(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    worker = load('executor_dry_worker', HERE/'worker.py')
    controller = load('executor_dry_controller', HERE/'controller.py')
    parser = load('executor_dry_csv', HERE/'csv_helpers.py')
    command = load('executor_dry_command', HERE.parent/'baseline-runtime-identity/counter_gate.py')
    cli = load('executor_dry_cli', HERE.parent/'eager-counter-repair/worker.py')
    saved_help = OLD/'1-stdout.txt'
    planned_report = Path('/tmp/attention-fa3-b32-capture-executor/baseline.ncu-rep')
    argv = worker.profile_argv('/usr/local/cuda/bin/ncu', HERE, planned_report)
    help_contract = cli.validate_installed_help(saved_help.read_text(), argv,
        worker.query_argv('/usr/local/cuda/bin/ncu', worker.METRIC_BASES[0]))
    worker.validate_import_help(saved_help.read_text(),
        worker.import_argv('/usr/local/cuda/bin/ncu', planned_report))
    hashes = {name: digest(path) for name, path in controller.payload().items()}
    hashes['controller.py'] = digest(HERE/'controller.py')
    expected = worker.load_expected_native(HERE)
    previous = json.loads((OLD/'result.json').read_text())
    successful = json.loads((HERE.parent/'fa3-b32-counter-diagnostic/result.json').read_text())
    tiny_path = HERE.parent/'eager-counter-repair/3-stdout.txt'
    tiny = parser.parse_ncu_csv(tiny_path.read_text(), ['sm__cycles_elapsed.avg'])
    assert tiny['kernel_name'] == 'okc_counter_gate'
    assert tiny['metrics']['sm__cycles_elapsed.avg']['value'] == 5541.35
    failed_stdout = next(c['stdout'] for c in previous['commands'] if '--export' in c['argv'])
    csv_fixture = fixture_csv(expected, successful['counter_report'])
    results = {}
    with tempfile.TemporaryDirectory(prefix='attention-capture-cpu-') as temporary:
        scratch = Path(temporary)
        for mode in ('pass', 'empty'):
            output = scratch/mode
            output.mkdir()
            fake = output/'fake-ncu'
            fake.write_text('#!'+sys.executable+'\n'+FAKE_NCU)
            fake.chmod(0o700)
            fake.with_suffix('.json').write_text(json.dumps(dict(mode=mode,
                csv=csv_fixture, failed_stdout=failed_stdout)))
            report = copy.deepcopy(previous)
            for key in ('error', 'error_type', 'traceback', 'ncu_report_error',
                    'ncu_report_artifact', 'report_roundtrip', 'counter_report',
                    'native_capture_identity', 'graph_proxy_identity', 'conclusion'):
                report.pop(key, None)
            report.update(kind='OFFLINE SIMULATED EXECUTOR TEST; NOT A NATIVE RESULT',
                dry_run=True, native_execution=False, status='starting', passed=False,
                worker_sha256=hashes['worker.py'],
                verified_hashes={k:v for k,v in hashes.items() if k not in ('worker.py','controller.py')},
                commands=[], artifact_manifest={}, counter_sessions_attempted=0)
            error = None
            try:
                worker.collect_counter(str(fake), HERE, output, command, parser,
                    expected, expected['bindings'], report=report)
                report.update(status='complete', passed=True)
            except ValueError as exc:
                error = str(exc)
                report.update(status='incomplete_or_failed', passed=False, error=error)
            (output/'activity-trace.json.gz').write_bytes((OLD/'activity-trace.json.gz').read_bytes())
            (output/'activity-trace.json').write_bytes(gzip.decompress((OLD/'activity-trace.json.gz').read_bytes()))
            (output/'summary.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
            reply = packaged_reply(controller, HERE, output, 0 if mode=='pass' else 1)
            materialized = controller.materialize_result(reply, destination/mode, hashes)
            assert materialized['completion_verified'] is (mode=='pass')
            commands = report['commands']
            assert sum('--import' in c['argv'] for c in commands) == (1 if mode=='pass' else 0)
            results[mode] = dict(completion_verified=materialized['completion_verified'],
                commands=[c['argv'] for c in commands],
                retained_artifacts=materialized['durable_artifacts_retrieved'],
                trace_retrieved=materialized['trace_retrieved'], error=error,
                native_identity_accepted=report.get('native_capture_identity', {}).get('accepted', False))
            results[mode]['materialized_artifacts'] = {}
            for name in materialized['durable_artifacts_retrieved']:
                path = destination/mode/'artifacts'/name
                meta = {'bytes':path.stat().st_size, 'sha256':digest(path)}
                assert meta == report['artifact_manifest'][name]
                results[mode]['materialized_artifacts'][name] = meta
            if mode=='pass':
                broken = copy.deepcopy(reply)
                broken['artifacts']['baseline.ncu-rep']['raw_sha256'] = '0'*64
                try:
                    controller.materialize_result(broken, destination/'corrupt', hashes)
                except ValueError:
                    results['corrupt_transport_rejected'] = True
                else:
                    raise AssertionError('Corrupt report transport was accepted')
    assert results.get('corrupt_transport_rejected')
    summary = dict(kind='CPU DRY RUN ONLY', passed=True, new_gpu_apps=0,
        native_ncu_report_validated=False, credential_or_service_calls=0,
        substituted_boundaries=['NCU executable', 'native worker process and remote paths'],
        actual_paths_exercised=['worker.collect_counter with real subprocess/file I/O',
            'controller.make_remote_entry artifact packaging', 'controller.materialize_result acceptance'],
        source_hashes=hashes, fixture_hashes={str(p.relative_to(HERE.parent)):digest(p) for p in
            (OLD/'result.json', tiny_path, HERE.parent/'fa3-b32-counter-diagnostic/result.json')},
        successful_tiny_fixture=dict(kernel=tiny['kernel_name'], cycles=5541.35),
        dry_run_source_sha256=digest(Path(__file__)), saved_help_sha256=digest(saved_help),
        saved_help_contract=help_contract,
        saved_version=(OLD/'0-stdout.txt').read_text().strip(),
        native_installed_version_rechecked=False,
        qualification='All new report bytes/signatures/metric combinations are simulated fixture data, not native NCU output or new correctness/timing measurements.',
        scenarios=results)
    (destination/'DRY_RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--output-dir', type=Path, required=True)
    summary = run(args.parse_args().output_dir)
    print(json.dumps({k:summary[k] for k in ('kind','passed','new_gpu_apps',
        'native_ncu_report_validated','credential_or_service_calls')}))
