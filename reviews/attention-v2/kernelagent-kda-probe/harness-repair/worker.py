"""Source-bound H200 diagnostic. CPU import has no torch/CUDA side effects.

Whole-call graph timing and eager kernel-replay counters are separate evidence.
This worker never invokes an official runner, model provider, or network API.
"""
import argparse
import csv
import gc
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
import statistics
import subprocess
import sys
import time
import traceback

SOURCES = {
    'selected.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'v4.py': 'dcfbca2f8e1135762b08c77df931a7f53948f69746e0314797ee3dbef4147e82',
    'reference.py': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
    'checks.py': '9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
SEEDS = (135871, 135872)
ARMS = ('champion', 'v4')
EVICTION_BYTES = 256 * 1024 * 1024
METRICS = (
    'gpu__time_duration.sum',
    'dram__bytes_read.sum',
    'dram__bytes_write.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'launch__registers_per_thread',
)
# Launch metadata and duration are auto-exported by the installed NCU; query
# hardware metrics only. All nine fields remain mandatory in the actual report.
HARDWARE_METRICS = tuple(m for m in METRICS if m not in (
    'gpu__time_duration.sum', 'launch__registers_per_thread'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n'); temporary.replace(path)


def bind_sources(root):
    found = {}
    for name, wanted in SOURCES.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or sha(path) != wanted:
            raise ValueError('Source binding failed: ' + name)
        found[name] = wanted
    return found


def case(batch, length=1, layer='full'):
    return dict(batch=batch, q_len=length, layer=layer, ctx=80000, heads='tp8',
                spread='fixed' if batch == 1 else 'agentic', page=1, kv_scale=1.,
                dist='coherent' if layer == 'full' else 'soft')


def order(index):
    pair, half = divmod(index, 2)
    return ARMS if (pair + half) % 2 == 0 else ARMS[::-1]


def paired_summary(samples):
    if len(samples) != 12 or any(set(x) != set(ARMS) for x in samples):
        raise ValueError('Six complete reverse-order pairs required')
    ratios = []
    for row in samples:
        if any(not math.isfinite(v) or v <= 0 for v in row.values()):
            raise ValueError('Finite positive raw graph times required')
        ratios.append(row['champion'] / row['v4'])
    pairs = [math.sqrt(ratios[i] * ratios[i+1]) for i in range(0, 12, 2)]
    return dict(champion_median_us=statistics.median(x['champion'] for x in samples),
                v4_median_us=statistics.median(x['v4'] for x in samples),
                reverse_pair_ratios=pairs, median_pair_speedup=statistics.median(pairs),
                favorable_pairs=sum(x > 1 for x in pairs))


def family(arm, batch):
    if arm not in ARMS or batch not in (16, 32):
        raise ValueError('Only exact diagnostic targets allowed')
    return '_full2h' if arm == 'v4' else 'FlashAttnFwdSm90' if batch == 16 else '_fused'


def capture_argv(ncu, root, report, arm, batch):
    if not Path(report).is_absolute() or Path(report).suffix != '.ncu-rep':
        raise ValueError('Absolute durable .ncu-rep export required')
    return [str(ncu), '--config-file', '0', '--replay-mode', 'kernel',
        '--target-processes', 'application-only', '--profile-from-start', 'off',
        '--kernel-name-base', 'demangled', '--kernel-name', 'regex:.*' + family(arm, batch) + '.*',
        '--launch-count', '1', '--metrics', ','.join(HARDWARE_METRICS), '--rename-kernels', '0',
        '--print-kernel-base', 'demangled', '--print-units', 'base', '--print-fp',
        '--export', str(report), '--page', 'raw', '--csv', '--clock-control', 'none',
        sys.executable, str(root/'worker.py'), '--payload-root', str(root),
        '--counter-child', arm, str(batch), str(SEEDS[-1])]


def import_argv(ncu, report):
    return [str(ncu), '--config-file', '0', '--import', str(report), '--rename-kernels', '0',
            '--print-kernel-base', 'demangled', '--print-units', 'base', '--print-fp',
            '--page', 'raw', '--csv']


def validate_help(help_text, argv):
    for token in argv:
        if token.startswith('--') and token not in ('--payload-root', '--counter-child'):
            if token not in help_text:
                raise ValueError('Installed NCU does not document option: ' + token)
    if '--units' in argv:
        raise ValueError('Unsupported historical --units option forbidden')


def parse_csv(text, wanted_family):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",') or line.startswith('ID,')), None)
    if start is None or 'No kernels were profiled' in text:
        raise ValueError('Missing/empty NCU metric capture')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    if not {'Kernel Name', 'Grid Size', 'Block Size', *METRICS}.issubset(reader.fieldnames or []):
        raise ValueError('NCU CSV missing required identity/metric fields')
    units = None; rows = []
    for row in reader:
        if not row.get('ID'):
            units = row; continue
        if str(row.get('ID', '')).isdigit():
            rows.append(row)
    if len(rows) != 1 or wanted_family not in rows[0].get('Kernel Name', ''):
        raise ValueError('Expected one intended producer family, got ' + str(len(rows)))
    row = rows[0]
    def launch_size(value):
        if re.fullmatch(r'\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)', value) is None:
            raise ValueError('Missing/invalid launch metadata')
        dims = [int(n) for n in re.findall(r'\d+', value)]
        if any(n <= 0 for n in dims): raise ValueError('Nonpositive launch dimensions')
        return dims
    identity = dict(kernel_name=row['Kernel Name'], grid=launch_size(row['Grid Size']),
                    block=launch_size(row['Block Size']))
    metrics = {}
    for key in METRICS:
        if units is None or not units.get(key):
            raise ValueError('Missing metric unit: ' + key)
        value = float(row[key].replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid metric: ' + key)
        metrics[key] = dict(value=value, unit=(units or {}).get(key))
    return dict(identity=identity, metrics=metrics)


def load(root, name, filename):
    spec = importlib.util.spec_from_file_location(name, root/filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone(torch, data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else dict(v) if isinstance(v, dict) else v
            for k, v in data.items()}


def restore(torch, work, fresh):
    if work.keys() != fresh.keys(): raise ValueError('Input dictionary keys changed')
    for key, value in fresh.items():
        if isinstance(value, torch.Tensor):
            if work[key].shape != value.shape or work[key].stride() != value.stride() or work[key].dtype != value.dtype:
                raise ValueError('Fixed-address layout changed: ' + key)
            work[key].copy_(value)
        elif work[key] != value:
            raise ValueError('Input metadata changed: ' + key)


def pointers(torch, data):
    return {k: (v.data_ptr(), tuple(v.shape), tuple(v.stride()), str(v.dtype))
            for k, v in data.items() if isinstance(v, torch.Tensor)}


def fixture_metadata(data, v4=None):
    """Untimed fixture facts and explicitly labelled shape-derived configuration."""
    cfg = data['config']; batch = cfg['batch']
    lengths = data['seq_lens'].cpu().tolist()
    span = data['page_table'].shape[1] * cfg['page_size']
    result = dict(seq_lens=lengths, seq_len_sum=sum(lengths),
                  page_table_shape=list(data['page_table'].shape), span_max=span)
    if v4 is not None:
        bn, chunk, warps, stages = v4.V4_CFG[batch]
        splits = (span + chunk - 1) // chunk
        result['v4_source_configuration'] = dict(block_n=bn, chunk=chunk, splits=splits,
            num_warps=warps, num_stages=stages, grid=[batch, splits],
            evidence='shape-derived original source; native grid validated separately by NCU')
        if batch == 32:
            splits = (span+8191)//8192
            result['champion_source_configuration'] = dict(block_n=64, chunk=8192, splits=splits,
                num_warps=4, num_stages=3, grid=[batch, cfg['num_kv_heads'], splits],
                evidence='shape-derived original source; native grid validated separately by NCU')
        else:
            result['champion_source_configuration'] = dict(route='FA3', resolved_split_count='unknown')
    return result


def mutated_fixture(torch, fresh):
    """Untimed, same-geometry live data mutation; never used for timing."""
    if fresh['page_table'].shape[1] < 2 or bool((fresh['seq_lens'] < 2).any()):
        raise ValueError('Declared fixture lacks two historical pages per request')
    mutated = clone(torch, fresh)
    mutated['qkv'].mul_(0.93)
    first = mutated['page_table'][:, 0].clone()
    mutated['page_table'][:, 0].copy_(mutated['page_table'][:, 1])
    mutated['page_table'][:, 1].copy_(first)
    if not bytes_equal(torch, mutated['cache_loc'], fresh['cache_loc']):
        raise ValueError('Mutation changed new KV write ownership')
    return mutated


def bytes_equal(torch, a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def check_ownership(torch, work, fresh):
    if work.keys() != fresh.keys(): raise ValueError('Input dictionary keys changed')
    for key, value in fresh.items():
        if key in ('k_cache', 'v_cache'):
            continue
        if isinstance(value, torch.Tensor):
            if not bytes_equal(torch, work[key], value):
                raise ValueError('Immutable input changed: ' + key)
        elif work[key] != value:
            raise ValueError('Immutable metadata changed: ' + key)
    locations = fresh['cache_loc'].long()
    width = fresh['config']['num_kv_heads'] * fresh['config']['head_dim']
    for key in ('k_cache', 'v_cache'):
        got, original = [x.view(-1, width).view(torch.uint8) for x in (work[key], fresh[key])]
        new = got[locations].clone()
        try:
            got[locations] = original[locations]
            if not bool(torch.equal(got, original)):
                raise ValueError('Historical cache rows changed: ' + key)
        finally:
            got[locations] = new


def oracle(torch, ref, work, fresh, output, state):
    torch.cuda.synchronize()
    raw = ref.collect(work, output)
    passed, message = ref.check_implementation(work, raw)
    if not passed:
        raise ValueError('Original oracle: ' + message)
    check_ownership(torch, work, fresh)
    if pointers(torch, work) != state:
        raise ValueError('Captured addresses changed')
    return dict(passed=True, message=message, ownership=True, fixed_addresses=True)


def make_graph(torch, fn, fresh):
    work = clone(torch, fresh)
    state = pointers(torch, work)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(2):
            fn(work)
    torch.cuda.current_stream().wait_stream(warm); torch.cuda.synchronize()
    restore(torch, work, fresh)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(work)
    return work, state, graph, output


def timed(torch, graph, eviction):
    eviction.zero_()  # Outside events, >H200 L2; never subtract profile time.
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(); graph.replay(); end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000


def api_metrics(ncu, report):
    # Use only already-installed NVIDIA Python API; no dependency installation.
    candidates = [p/'extras/python' for p in Path(ncu).resolve().parents]
    for path in candidates:
        if path.is_dir():
            sys.path.insert(0, str(path))
    try:
        import ncu_report
        result = ncu_report.load_report(str(report))
        actions = [result.range_by_idx(i).action_by_idx(j)
                   for i in range(result.num_ranges())
                   for j in range(result.range_by_idx(i).num_actions())]
        if len(actions) != 1:
            raise ValueError('Expected one Python-API action')
        action = actions[0]
        values = {m: action[m].value() for m in METRICS}
        return dict(available=True, kernel_name=action.name(), metrics=values)
    except Exception as exc:
        return dict(available=False, error=type(exc).__name__ + ': ' + str(exc)[:500])


def mapped_flash_binaries():
    """Hash only loaded flash-attention shared objects from this process map."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and '.so' in path and 'flash' in path.lower():
            paths.add(path)
    return [dict(path=p, sha256=sha(p)) for p in sorted(paths) if Path(p).is_file()]


def counter_child(root, arm, batch, seed):
    bind_sources(root)
    import torch
    ref = load(root, 'reference', 'reference.py')
    module = load(root, 'selected' if arm == 'champion' else 'v4',
                  'selected.py' if arm == 'champion' else 'v4.py')
    if module._INTERP: raise ValueError('Native probe forbids TRITON_INTERPRET')
    fresh = ref.generate_input(**case(batch), seed=seed, device='cuda')
    work = clone(torch, fresh); state = pointers(torch, work)
    for _ in range(2): module.kernel_fn(work)
    restore(torch, work, fresh); output = module.kernel_fn(work)
    before = oracle(torch, ref, work, fresh, output, state)
    print('CHILD_CHECK=' + json.dumps(dict(phase='pre', **before)), flush=True)
    restore(torch, work, fresh)
    eviction = torch.empty(EVICTION_BYTES, dtype=torch.uint8, device='cuda'); eviction.zero_()
    torch.cuda.synchronize()
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    output = module.kernel_fn(work)
    torch.cuda.synchronize()
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    after = oracle(torch, ref, work, fresh, output, state)
    print('CHILD_CHECK=' + json.dumps(dict(phase='post', **after)), flush=True)


def run(root, output, seconds):
    started = time.monotonic()
    report = dict(status='running', worker_sha256=sha(__file__), verified_hashes=bind_sources(root),
                  baseline_cases=[], paired_cases=[], counter_reports=[],
                  correctness_complete=False, timing_complete=False, counters_complete=False,
                  candidate_changed=False, official_check=False, ranked_submission=False,
                  protocol=dict(eviction_bytes=EVICTION_BYTES, seeds=list(SEEDS), reverse_pairs=6,
                      graph_timing='raw whole-call CUDA event graph replay; no fixed-cost subtraction',
                      counters='eager isolated kernel replay; not graph internal-state equivalence',
                      promotion='not assessed: v4 paired coverage only B16/B32 full decode'))

    def phase(name):
        report['phase'] = name; report['elapsed_seconds'] = time.monotonic()-started
        save(output, report); print('PROGRESS=' + name, flush=True)
        if report['elapsed_seconds'] > seconds:
            raise TimeoutError('Worker deadline reached at ' + name)

    def command(argv, stem, limit):
        phase(stem)
        remaining = seconds - (time.monotonic()-started)
        if remaining < limit + 3:
            raise TimeoutError('Insufficient remaining time for ' + stem)
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=root, start_new_session=True)
        timeout_error = None
        try:
            stdout, stderr = child.communicate(timeout=limit)
        except subprocess.TimeoutExpired as exc:
            timeout_error = exc
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            stdout, stderr = child.communicate(timeout=3)
        (output.parent/(stem+'.stdout.txt')).write_text(stdout)
        (output.parent/(stem+'.stderr.txt')).write_text(stderr)
        if timeout_error is not None:
            raise TimeoutError(stem + ' exceeded ' + str(limit) + 's; process group killed')
        p = subprocess.CompletedProcess(argv, child.returncode, stdout, stderr)
        if p.returncode:
            raise RuntimeError(stem + ' exit ' + str(p.returncode) + ': ' + p.stderr[-1200:])
        return p

    try:
        phase('load_native_modules')
        import torch
        import triton
        ref = load(root, 'reference', 'reference.py')
        modules = {'champion': load(root, 'selected', 'selected.py'), 'v4': load(root, 'v4', 'v4.py')}
        if any(m._INTERP for m in modules.values()):
            raise ValueError('Native probe forbids TRITON_INTERPRET')
        props = torch.cuda.get_device_properties(0)
        if 'H200' not in props.name or (props.major, props.minor) != (9, 0):
            raise ValueError('Expected actual H200 SM90, got ' + props.name)
        report['runtime'] = dict(gpu=props.name, sm_count=props.multi_processor_count,
                                 torch=torch.__version__, triton=triton.__version__,
                                 operand_types={arm: dict(qk=str(m._QK_DOT), pv=str(m._DOT))
                                                for arm, m in modules.items()})
        eviction = torch.empty(EVICTION_BYTES, dtype=torch.uint8, device='cuda')

        for length in (1, 8):
            for batch in (1, 16, 32):
                for layer in ('sliding', 'full'):
                    spec = case(batch, length, layer); phase('baseline-'+str((batch, length, layer)))
                    fresh = ref.generate_input(**spec, seed=SEEDS[0], device='cuda')
                    work, state, graph, out = make_graph(torch, modules['champion'].kernel_fn, fresh)
                    restore(torch, work, fresh); graph.replay()
                    checked = oracle(torch, ref, work, fresh, out, state)
                    expected = ref.collect(work, out).clone(); samples = []
                    for _ in range(6):
                        restore(torch, work, fresh); samples.append(timed(torch, graph, eviction))
                        if not bytes_equal(torch, ref.collect(work, out), expected):
                            raise ValueError('Baseline graph output/cache changed on replay')
                    checked_after = oracle(torch, ref, work, fresh, out, state)
                    report['baseline_cases'].append(dict(case=spec, seed=SEEDS[0], oracle=checked,
                        fixture=fixture_metadata(fresh),
                        post_oracle=checked_after, graph_samples_us=samples,
                        raw_graph_median_us=statistics.median(samples)))
                    del work, state, graph, out, fresh, expected; gc.collect(); torch.cuda.empty_cache()

        for batch in (16, 32):
            row = dict(case=case(batch), fixtures=[]); report['paired_cases'].append(row)
            for seed in SEEDS:
                phase('paired-b'+str(batch)+'-seed'+str(seed))
                fresh = ref.generate_input(**case(batch), seed=seed, device='cuda')
                arms = {}; fixture = dict(seed=seed, oracle={}, mutation_oracle={}, post_oracle={}, samples=[],
                                         fixture=fixture_metadata(fresh, modules['v4']))
                row['fixtures'].append(fixture)
                for arm in ARMS:
                    arms[arm] = make_graph(torch, modules[arm].kernel_fn, fresh)
                    work, state, graph, out = arms[arm]
                    restore(torch, work, fresh); graph.replay()
                    fixture['oracle'][arm] = oracle(torch, ref, work, fresh, out, state)
                    arms[arm] += (ref.collect(work, out).clone(),)
                    changed = mutated_fixture(torch, fresh)
                    restore(torch, work, changed); graph.replay()
                    fixture['mutation_oracle'][arm] = dict(
                        kind='qkv *0.93; first two historical page entries swapped per request; same graph addresses',
                        timed=False, **oracle(torch, ref, work, changed, out, state))
                    restore(torch, work, fresh); graph.replay()
                    del changed
                if arms['champion'][3].data_ptr() == arms['v4'][3].data_ptr():
                    raise ValueError('Arm graph output alias')
                for index in range(12):
                    sample = {}
                    for arm in order(index):
                        work, state, graph, out, expected = arms[arm]
                        restore(torch, work, fresh); sample[arm] = timed(torch, graph, eviction)
                        if pointers(torch, work) != state or not bytes_equal(torch, ref.collect(work, out), expected):
                            raise ValueError('Paired graph replay/address/cache inconsistency')
                    fixture['samples'].append(sample)
                fixture['summary'] = paired_summary(fixture['samples'])
                for arm in ARMS:
                    work, state, graph, out, _ = arms[arm]
                    fixture['post_oracle'][arm] = oracle(torch, ref, work, fresh, out, state)
                del arms, work, state, graph, out, fresh, expected; gc.collect(); torch.cuda.empty_cache()
        report['correctness_complete'] = True; report['timing_complete'] = True
        report['runtime']['loaded_flash_binaries'] = mapped_flash_binaries()
        phase('native_profile_tools')
        ncu = shutil.which('ncu')
        if ncu is None: raise FileNotFoundError('No installed NCU; no install attempted')
        help_call = command([ncu, '--config-file', '0', '--help'], 'ncu-help', 5)
        (output.parent/'ncu-help.txt').write_text(help_call.stdout+help_call.stderr)
        version = command([ncu, '--config-file', '0', '--version'], 'ncu-version', 3)
        (output.parent/'ncu-version.txt').write_text(version.stdout+version.stderr)
        report['runtime']['ncu_version'] = version.stdout+version.stderr
        query = command([ncu, '--config-file', '0', '--query-metrics', '--query-metrics-mode', 'suffix',
                         '--metrics', ','.join(dict.fromkeys(m.split('.')[0] for m in HARDWARE_METRICS))], 'ncu-query', 8)
        (output.parent/'ncu-query.txt').write_text(query.stdout+query.stderr)
        for name in HARDWARE_METRICS:
            if re.search(r'(?<![\w.])'+re.escape(name)+r'(?![\w.])', query.stdout+query.stderr) is None:
                raise ValueError('Installed NCU metric not proven supported: ' + name)
        for batch in (16, 32):
            for arm in ARMS:
                stem = 'ncu-'+arm+'-b'+str(batch); path = output.parent/(stem+'.ncu-rep')
                if path.exists(): raise ValueError('Never overwrite/retry an existing counter capture')
                argv = capture_argv(ncu, root, path.resolve(), arm, batch)
                validate_help(help_call.stdout+help_call.stderr, argv)
                validate_help(help_call.stdout+help_call.stderr, import_argv(ncu, path))
                p = command(argv, stem, 35)
                (output.parent/(stem+'.csv')).write_text(p.stdout)
                if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 8*1024*1024:
                    raise ValueError('Missing/nonempty/bounded .ncu-rep acceptance failed')
                captured = parse_csv(p.stdout, family(arm, batch)); digest = sha(path)
                imported_call = command(import_argv(ncu, path), stem+'-import', 8)
                (output.parent/(stem+'-imported.csv')).write_text(imported_call.stdout)
                imported = parse_csv(imported_call.stdout, family(arm, batch))
                if captured != imported or digest != sha(path):
                    raise ValueError('Native capture/offline-import exact parity failed')
                checks = {}
                for line in (p.stdout+'\n'+p.stderr).splitlines():
                    if line.startswith('CHILD_CHECK='):
                        c = json.loads(line.split('=', 1)[1]); checks[c['phase']] = c['passed']
                if checks != {'pre': True, 'post': True}:
                    raise ValueError('Native counter child original pre/post oracle evidence missing')
                api = api_metrics(ncu, path); save(output.parent/(stem+'.api.json'), api)
                report['counter_reports'].append(dict(arm=arm, batch=batch,
                    source_sha256=SOURCES['selected.py' if arm == 'champion' else 'v4.py'],
                    artifact_name=path.name, artifact_sha256=digest, artifact_bytes=path.stat().st_size,
                    capture_csv=stem+'.csv', import_csv=stem+'-imported.csv', roundtrip_passed=True,
                    child_checks=checks, api=api, **captured))
        report['counters_complete'] = True; report['status'] = 'complete'; phase('complete')
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
        save(output, report)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started; save(output, report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--payload-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, default=Path('/tmp/kda-h200-trial/summary.json'))
    p.add_argument('--deadline', type=int, default=260)
    p.add_argument('--counter-child', nargs=3, metavar=('ARM', 'BATCH', 'SEED'))
    args = p.parse_args(); root = args.payload_root.resolve()
    if args.counter_child:
        arm, batch, seed = args.counter_child; family(arm, int(batch))
        counter_child(root, arm, int(batch), int(seed))
    else:
        run(root, args.output.resolve(), args.deadline)


if __name__ == '__main__':
    main()
