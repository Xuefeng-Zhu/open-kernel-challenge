"""Exact-source, compile-only H200 diagnostic. Never execute undersized buffers.

Upload allowlist: this program and the two frozen candidate files only.
The synthetic specialization follows the public official attention contract;
no official reference or competitor implementation is imported by this program.
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

SOURCES = {
    'mixed_fa3_fp8_qk.py': 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',
    'mixed_fa3_page_lookahead.py': 'fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c',
}
IMAGE = 'im-cPTbGpNiX3RarKResPgX7h'
RATE = Decimal('.001261') + 2 * Decimal('.0000131') + 8 * Decimal('.00000222')
BASE = Decimal('1.787819802')
LIMIT = min(Decimal('.25'), Decimal('2') - BASE)
MAX_RAW = 16 * 1024 * 1024
MAX_PACKED = 2 * 1024 * 1024
CHUNK = 16 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def cost_bound(seconds):
    return Decimal(math.ceil(seconds + 5)) * RATE * Decimal('1.1') + Decimal('.01')


def suppress_credential_identifier_warning(category):
    # The SDK's expiry warning contains a credential identifier. Expiry is
    # checked from the protected local receipt; preserve all unrelated warnings.
    warnings.filterwarnings('ignore', message=r'Your Modal token .* expires', category=category)


def specialization(lookahead):
    kw = dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16,
              BLOCK_N=64, BLOCK_L=16, SPLITS=11, ROPE=False,
              WINDOWED=False, CHUNK_SIZE=8192, SEPARATE_MERGE=True,
              num_warps=4, num_stages=3)
    if lookahead:
        kw['PAGE_LOOKAHEAD'] = True
    return kw


def worker():
    """Compile, load CUDA handles, and disassemble; no bracket/kernel launches."""
    import importlib.util
    import importlib.metadata
    import platform
    import shutil
    import signal
    import sys
    import traceback

    root = Path(__file__).resolve().parent
    dest = Path('/tmp/a215-native')
    dest.mkdir(exist_ok=True)
    report = dict(kind='Native compile/disassembly only', status='starting',
                  candidate_kernel_launches=0, correctness_tested=False, performance_tested=False,
                  sources=SOURCES, records=[], started_utc=now(),
                  logical_fixture=dict(batch=32, q_len=1, heads='tp8', layer='full',
                      spread='agentic', ctx=80000, page=1, page_table_columns=86001,
                      qkv_width=2304, splits=11, chunk=8192,
                      qk_factor=5.4730064863875, sm_scale=.125))

    def save():
        write_json(dest/'summary.json', report)

    def metadata_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): metadata_value(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [metadata_value(v) for v in value]
        return repr(value)

    def alarm(*_):
        raise TimeoutError('Compile-only worker deadline')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(55)
    started = time.monotonic()
    save()
    try:
        import torch
        import triton
        import triton.language as tl
        assert not os.environ.get('TRITON_INTERPRET'), 'Native compiler required'
        assert torch.cuda.is_available() and 'H200' in torch.cuda.get_device_name()
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5)
        report['environment'] = dict(python=platform.python_version(), torch=torch.__version__,
            triton=triton.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            driver=driver.stdout.strip() if driver.returncode == 0 else None)
        for pkg in ['sglang', 'sgl-kernel']:
            try: report['environment'][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError: report['environment'][pkg] = None
        # Allocator-aligned pointers reproduce JIT types/alignment. They are NOT
        # valid data buffers for these logical shapes. Warmup never launches.
        dtypes = dict(qkv=torch.bfloat16, seq=torch.int32, pt=torch.int32,
            loc=torch.int64, kc=torch.float8_e4m3fn, vc=torch.float8_e4m3fn,
            ks=torch.float32, vs=torch.float32, po=torch.float32,
            pm=torch.float32, pl=torch.float32, out=torch.bfloat16)
        tensors = {k: torch.empty(1, dtype=v, device='cuda') for k, v in dtypes.items()}
        report['pointer_metadata'] = {k: dict(dtype=str(v.dtype), elements=v.numel(),
            alignment_mod16=v.data_ptr() % 16) for k, v in tensors.items()}
        assert all(v.data_ptr() % 16 == 0 for v in tensors.values())
        t = tensors
        args = [t[k] for k in ['qkv', 'seq', 'pt', 'loc', 'kc', 'vc', 'ks', 'vs', 'po', 'pm', 'pl']]
        args += [t['out'], t['out'], 1.0, 1e-5, 5.4730064863875, .125, 0, 1, 2304, 86001]
        report['runtime_scalars'] = args[-8:]
        save()
        for label, filename in [('a213', 'mixed_fa3_fp8_qk.py'), ('a215', 'mixed_fa3_page_lookahead.py')]:
            raw = (root/filename).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == SOURCES[filename]
            spec = importlib.util.spec_from_file_location('native_'+label, root/filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            for which in ['producer', 'merge']:
                before = time.monotonic()
                if which == 'producer':
                    options = specialization(label == 'a215')
                    kernel = module._fused.warmup(*args, DOT=tl.bfloat16, QK_DOT=tl.float8e4nv,
                                                 grid=(32, 2, 11), **options)
                else:
                    options = dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16,
                                   SPLITS=11, BS=16, num_warps=4, num_stages=3)
                    kernel = module._merge.warmup(t['qkv'], t['vs'], t['po'], t['pm'], t['pl'],
                        t['out'], 1e-5, 2304, grid=(32, 16), **options)
                # Driver handle initialization supplies register/spill metadata;
                # it loads the compiled module but does not invoke the kernel.
                kernel._init_handles()
                row = dict(source=filename, kernel=which, options=options,
                    compile_and_load_seconds=time.monotonic()-before,
                    registers=getattr(kernel, 'n_regs', None),
                    driver_spills_raw=getattr(kernel, 'n_spills', None),
                    shared_memory_bytes=getattr(kernel.metadata, 'shared', None),
                    metadata=metadata_value(kernel.metadata._asdict() if hasattr(kernel.metadata, '_asdict') else str(kernel.metadata)),
                    artifacts={})
                report['records'].append(row)
                for kind in ['ttir', 'ttgir', 'llir', 'ptx', 'cubin']:
                    value = kernel.asm.get(kind)
                    if value is None:
                        row['artifacts'][kind] = None
                        continue
                    value = value.encode() if isinstance(value, str) else bytes(value)
                    name = f'{label}.{which}.{kind}'
                    (dest/name).write_bytes(value)
                    row['artifacts'][kind] = dict(file=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                save()
                tool = shutil.which('cuobjdump')
                if tool is None:
                    p = Path('/usr/local/cuda/bin/cuobjdump')
                    tool = str(p) if p.is_file() else None
                if tool is None:
                    p = Path(triton.__file__).parent/'backends/nvidia/bin/cuobjdump'
                    tool = str(p) if p.is_file() else None
                if tool and (dest/f'{label}.{which}.cubin').exists():
                    sass = subprocess.run([tool, '--dump-sass', str(dest/f'{label}.{which}.cubin')],
                                          capture_output=True, timeout=5)
                    row['sass_tool_exit_code'] = sass.returncode
                    if sass.returncode == 0:
                        name = f'{label}.{which}.sass'
                        (dest/name).write_bytes(sass.stdout)
                        row['artifacts']['sass'] = dict(file=name, bytes=len(sass.stdout), sha256=hashlib.sha256(sass.stdout).hexdigest())
                else:
                    row['sass_unavailable_reason'] = 'cuobjdump absent; PTX is not final machine scheduling'
                save()
        report['status'] = 'compiled'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:2000],
                      traceback=traceback.format_exc()[-6000:])
    finally:
        signal.alarm(0)
        report['worker_seconds'] = time.monotonic()-started
        save()
    return 0 if report['status'] == 'compiled' else 1


def make_remote_entry():
    # Only this self-contained generator is serialized, not local config/state.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path('/opt/a215-native')
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(root/'native_diagnostic.py'), 'worker'],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            code = 124
        folder = Path('/tmp/a215-native')
        files = sorted(folder.glob('*')) if folder.exists() else []
        allowed = {'summary.json'} | {f'{v}.{k}.{ext}' for v in ['a213', 'a215']
            for k in ['producer', 'merge'] for ext in ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass']}
        total = sum(p.stat().st_size for p in files)
        if total > 16*1024*1024 or any(p.name not in allowed or p.is_symlink() for p in files):
            yield {'kind': 'failure', 'reason': 'Artifact size/path guard', 'exit_code': code}
            return
        payload = {p.name: base64.b64encode(p.read_bytes()).decode() for p in files}
        raw = json.dumps(payload, sort_keys=True).encode()
        if len(raw) > 24*1024*1024:
            yield {'kind': 'failure', 'reason': 'Encoded artifact size guard', 'exit_code': code}
            return
        packed = zlib.compress(raw, 9)
        if len(packed) > 2*1024*1024:
            yield {'kind': 'failure', 'reason': 'Compressed artifact size guard', 'exit_code': code}
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


def launch():
    import modal
    from modal.exception import ServerWarning
    suppress_credential_identifier_warning(ServerWarning)
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    auth = root/'local-notes/modal-auth.toml'
    assert auth.is_file() and not auth.is_symlink() and (auth.stat().st_mode & 0o777) == 0o600
    assert auth.stat().st_uid == os.getuid()
    overrides = ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'MODAL_OAUTH_REFRESH_TOKEN',
                 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET']
    assert not any(os.environ.get(k) for k in overrides), 'Credential override present; stop'
    os.environ['MODAL_CONFIG_PATH'] = str(auth)
    os.environ['MODAL_PROFILE'] = 'xuefeng-zhu'
    # Modal's module configuration is initialized on import: the caller must
    # already set this same config path before invoking this program.
    assert Path(modal.config.user_config_path).resolve() == auth.resolve()
    ledger_path = here/'native-modal-run.json'
    if ledger_path.exists():
        raise RuntimeError('Existing run record: reconcile, never launch a duplicate')
    pre = json.loads((root/'local-notes/a215-launch-preflight.local.json').read_text())
    assert pre['workspace'] == 'xuefeng-zhu' and Decimal(pre['verified_net_spend_cap_usd']) == 0
    assert Decimal(pre['verified_credit_balance_usd']) > LIMIT
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(pre['billing_ui_verified_utc'])).total_seconds() < 900
    login = json.loads((root/'local-notes/modal-login-20261025-1536.local.json').read_text())
    expiry = datetime.strptime(login['verification']['expires_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    assert login['state'] == 'verified' and (expiry-datetime.now(timezone.utc)).total_seconds() > 180
    cumulative = json.loads((here.parent/'modal-budget.json').read_text())
    assert Decimal(cumulative['conservative_stage_spent_usd']) == BASE
    assert not cumulative.get('active_reservation'), 'Another budget reservation requires reconciliation'
    assert cost_bound(125) < LIMIT
    slot_path = root/'local-notes/gpu-slot.local.json'
    assert json.loads(slot_path.read_text())['state'] in ['idle', 'terminal']

    def cli(*args):
        p = subprocess.run([str(root/'.venv/bin/python'), '-m', 'modal', *args],
                           capture_output=True, text=True, timeout=15)
        if p.returncode:
            raise RuntimeError('Supported Modal operation failed: '+' '.join(args[:2]))
        return json.loads(p.stdout)

    billing = cli('billing', 'summary', '--json')
    assert Decimal(billing['billed_cost']) == 0
    apps = cli('app', 'list', '--env', 'main', '--json')
    assert all(x['state'] == 'stopped' and int(x['tasks']) == 0 for x in apps)
    names = list(SOURCES) + ['native_diagnostic.py']
    paths = {n: here.parent/n if n in SOURCES else here/n for n in names}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    frozen = {}
    for name, path in paths.items():
        data = path.read_bytes()
        assert not path.is_symlink()
        assert data == subprocess.check_output(['git', 'show', head+':'+path.relative_to(root).as_posix()], cwd=root)
        frozen[name] = hashlib.sha256(data).hexdigest()
    assert all(frozen[n] == sha for n, sha in SOURCES.items())
    image = modal.Image.from_id(IMAGE)
    for name, path in paths.items():
        image = image.add_local_file(path, '/opt/a215-native/'+name, copy=False)
    app = modal.App('okc-a215-native-compile', include_source=False)
    fn = app.function(image=image, gpu='H200', cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=75, startup_timeout=20, retries=None, serialized=True, include_source=False,
        block_network=True, restrict_modal_access=True)(make_remote_entry())
    # Modal1.6.2 forbids explicit retry policies for generators, including zero;
    # retries=None provides no application retry policy. Never resend a call.
    record = dict(experiment='A2-15 native compile', state='reserved', app_id=None,
        created_utc=now(), source_commit=head, payload_sha256=frozen,
        base_image=IMAGE, billing_before=billing, prior_conservative_cumulative_usd=str(BASE),
        remaining_ceiling_usd=str(LIMIT), cost_reserve_usd=str(cost_bound(125)),
        resources=dict(gpu='H200', cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            timeout=75, startup_timeout=20, worker_hard_timeout=60, app_watchdog=110,
            retry_policy=None, explicit_invocations=1),
        native_compile_only=True, correctness_tested=False, performance_tested=False)
    write_json(ledger_path, record)
    cumulative['active_reservation'] = dict(experiment='A2-15 native compile',
        reserve_usd=str(cost_bound(125)), reserved_utc=now(), app_id=None)
    write_json(here.parent/'modal-budget.json', cumulative)
    write_json(slot_path, dict(state='reserved', provider='modal', experiment='A2-15', app_id=None))
    stop_lock = threading.Lock()
    stop_result = {}
    timer = None
    started = time.monotonic()

    def stop():
        with stop_lock:
            if not app.app_id or stop_result:
                return
            stop_result.update(requested_utc=now(), app_id=app.app_id)
            try:
                p = subprocess.run([str(root/'.venv/bin/python'), '-m', 'modal', 'app', 'stop',
                    app.app_id, '--yes', '--env', 'main'], capture_output=True, text=True, timeout=15)
                stop_result['exit_code'] = p.returncode
            except Exception as exc:
                stop_result['error_type'] = type(exc).__name__
            print('Stop requested for the same diagnostic app; verification follows.', flush=True)

    try:
        with app.run(detach=False, environment_name='main'):
            record.update(state='accepted', app_id=app.app_id, accepted_utc=now())
            cumulative['active_reservation']['app_id'] = app.app_id
            write_json(here.parent/'modal-budget.json', cumulative)
            write_json(ledger_path, record)
            write_json(slot_path, dict(state='accepted', provider='modal', experiment='A2-15', app_id=app.app_id))
            print('Accepted A2-15 diagnostic app '+app.app_id, flush=True)
            timer = threading.Timer(110, stop)
            timer.daemon = True
            timer.start()
            manifest = None
            chunks = []
            complete = False
            for item in fn.remote_gen():
                if item['kind'] == 'manifest':
                    assert manifest is None and item['packed_bytes'] <= MAX_PACKED
                    manifest = item
                    record['transport_manifest'] = item
                    write_json(ledger_path, record)
                elif item['kind'] == 'chunk':
                    assert manifest and item['index'] == len(chunks) and len(item['data']) <= CHUNK
                    chunks.append(item['data'])
                    assert sum(map(len, chunks)) <= MAX_PACKED
                elif item['kind'] == 'complete':
                    assert manifest and item['chunks'] == len(chunks) == manifest['chunks']
                    complete = True
                else:
                    raise RuntimeError('Worker reported artifact guard failure')
            assert complete and manifest
            packed = b''.join(chunks)
            assert len(packed) == manifest['packed_bytes'] and hashlib.sha256(packed).hexdigest() == manifest['sha256']
            decoder = zlib.decompressobj()
            raw = decoder.decompress(packed, 24*1024*1024+1)
            assert len(raw) == manifest['raw_bytes'] <= 24*1024*1024 and decoder.eof and not decoder.unconsumed_tail
            decoded = json.loads(raw)
            assert set(decoded) == set(manifest['artifact_sha256'])
            dest = here/'native_artifacts'
            dest.mkdir(exist_ok=False)
            for name, b64 in decoded.items():
                assert name == 'summary.json' or re.fullmatch(r'a21[35]\.(producer|merge)\.(ttir|ttgir|llir|ptx|cubin|sass)', name)
                content = base64.b64decode(b64, validate=True)
                assert hashlib.sha256(content).hexdigest() == manifest['artifact_sha256'][name]
                (dest/name).write_bytes(content)
            record.update(state='function-terminal', artifact_retrieval_complete=True,
                          exit_code=manifest['exit_code'])
            write_json(ledger_path, record)
            stop()
    except Exception as exc:
        record.update(state='needs-reconciliation', app_id=app.app_id, error_type=type(exc).__name__)
        print('Diagnostic stopped at '+type(exc).__name__+'; inspect this exact app, do not relaunch.', flush=True)
    finally:
        stop()
        if timer: timer.cancel()
        try:
            states = cli('app', 'list', '--env', 'main', '--json')
            exact = next((x for x in states if x['app_id'] == app.app_id), None)
            record['shutdown_observation'] = exact
            record['shutdown_verified'] = bool(exact and exact['state'] == 'stopped' and int(exact['tasks']) == 0)
            record['billing_after'] = cli('billing', 'summary', '--json')
        except Exception as exc:
            record.update(shutdown_verified=False, shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stop_result, client_finished_utc=now(), client_wall_seconds=time.monotonic()-started)
        write_json(ledger_path, record)
        write_json(slot_path, dict(state='terminal' if record.get('shutdown_verified') else 'uncertain',
                                  provider='modal', experiment='A2-15', app_id=app.app_id))
        print(json.dumps({k: record.get(k) for k in ['app_id', 'state', 'exit_code', 'shutdown_verified', 'client_wall_seconds']}), flush=True)


def self_test():
    import ast
    tree = ast.parse(Path(__file__).read_text())
    worker_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) for n in ast.walk(worker_node))
    assert specialization(False)['SPLITS'] == 11
    assert specialization(True)['PAGE_LOOKAHEAD'] is True and 'PAGE_LOOKAHEAD' not in specialization(False)
    assert set(specialization(True)) - set(specialization(False)) == {'PAGE_LOOKAHEAD'}
    assert cost_bound(125) == Decimal('.196609280') and cost_bound(125) < LIMIT
    assert (86001+8191)//8192 == 11 and (2*16+2*2)*64 == 2304
    assert len(SOURCES) == 2 and all(len(v) == 64 for v in SOURCES.values())
    class MockServerWarning(UserWarning):
        pass
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        suppress_credential_identifier_warning(MockServerWarning)
        warnings.warn('Your Modal token TEST_IDENTIFIER expires soon', MockServerWarning)
        warnings.warn('Unrelated server warning', MockServerWarning)
        warnings.warn('Unrelated Python warning', UserWarning)
        assert [str(w.message) for w in caught] == ['Unrelated server warning', 'Unrelated Python warning']
    print('7 local compile-protocol/privacy guards passed; no Modal or CUDA call.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['worker', 'launch', 'self-test'])
    args = p.parse_args()
    if args.action == 'worker': raise SystemExit(worker())
    elif args.action == 'launch': launch()
    else: self_test()
