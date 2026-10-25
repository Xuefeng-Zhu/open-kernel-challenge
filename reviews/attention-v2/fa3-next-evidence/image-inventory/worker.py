"""Bounded CPU-only vendor-source inventory; never import CUDA/operator packages."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import time

MAX_FILES = 64
MAX_BYTES = 8 * 1024 * 1024
MAX_FILE = 2 * 1024 * 1024
MAX_SNIPPET = 4096
MAX_SNIPPETS_TOTAL = 64 * 1024
MAX_SIGNATURES_TOTAL = 24 * 1024
MAX_JSON = 256 * 1024
MAX_WALK_DEPTH, MAX_WALK_DIRS, MAX_WALK_ENTRIES = 4, 32, 256
MODULES = ('triton', 'flashinfer', 'sglang', 'sgl_kernel', 'cutlass')
DISTRIBUTIONS = ('torch', 'triton', 'sglang', 'sgl-kernel', 'flashinfer',
                 'flashinfer-python', 'flashinfer-jit-cache', 'cutlass-dsl')
API_NAMES = {'warpgroup_mma', 'warpgroup_mma_wait', 'warpgroup_mma_init',
             'fence_async_shared', 'async_load', 'async_gather', 'commit_group',
             'wait_group', 'init', 'expect', 'wait', 'arrive', 'invalidate',
             'flash_attn_with_kvcache', 'get_scheduler_metadata', 'xqa',
             'trtllm_batch_decode_with_kv_cache', 'trtllm_batch_context_with_kv_cache',
             'single_decode_with_kv_cache'}
METHOD_CLASSES = {'BatchDecodeWithPagedKVCacheWrapper', 'BatchPrefillWithPagedKVCacheWrapper'}
RELATIVE_SOURCES = {
    'triton': ('__init__.py', 'experimental/gluon/__init__.py',
        'experimental/gluon/language/nvidia/hopper/__init__.py',
        'experimental/gluon/language/nvidia/ampere/async_copy.py',
        'experimental/gluon/language/nvidia/hopper/mbarrier.py',
        'experimental/gluon/language/nvidia/hopper/tma.py'),
    'flashinfer': ('__init__.py', 'xqa.py', 'jit/xqa.py', 'decode.py', 'prefill.py'),
    'sgl_kernel': ('__init__.py', 'flash_attn.py'),
    'sglang': ('__init__.py', 'srt/layers/attention/flashattention_backend.py',
               'srt/layers/attention/flashinfer_backend.py'),
    'cutlass': ('__init__.py',),
}
FIXED_VENDOR_DIRS = (
    '/opt/sglang/sgl-kernel/csrc/attention',
    '/opt/sglang/sgl-kernel/3rdparty/flash-attention/hopper',
    '/opt/sglang/third_party/flash-attention/hopper',
    '/opt/sglang/3rdparty/flash-attention/hopper',
    '/opt/sglang/csrc/attention',
)
HEADER_NAMES = {'mainloop_fwd_sm90_tma_gmma_ws.hpp', 'mainloop_fwd_sm90.hpp',
    'flash_fwd_kernel_sm90.h', 'tile_scheduler.hpp', 'tile_size.h', 'flash_api.cpp',
    'flash.h', 'flash_fwd_launch_template.h', 'paged_kv.h', 'softmax.h',
    'flash_prepare_scheduler.cu', 'decode.cuh', 'prefill.cuh', 'page.cuh'}
SNIPPET_PATTERNS = (
    re.compile(r'page_(?:size|block_size).*(?:not in|supported|arbitrary|%|==)'),
    re.compile(r'CUDAGraph|CUDA graph|cuda graph|capture'),
    re.compile(r'\.cpu\(|\.to\(\s*[\'"]cpu[\'"]'),
    re.compile(r'warpgroup_mma|wait_group|async_load|async_gather|mbarrier'),
    re.compile(r'PagedKVNonTMA|VarlenDynamicPersistent|prepare_varlen|Pipeline|NumMma'),
)


def clipped(text, size):
    return str(text).encode('utf-8')[:max(0, size)].decode('utf-8', errors='ignore')


def encoded(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode('utf-8') + b'\n'


def source_summary(raw, suffix):
    """AST/text only: no eval, import, annotation resolution or operator calls."""
    text = raw.decode('utf-8', errors='replace')
    result = dict(api_signatures=[], page_size_guards=[])
    if suffix == '.py':
        try:
            tree = ast.parse(text)
            class Visitor(ast.NodeVisitor):
                def __init__(self):
                    self.classes = []

                def visit_ClassDef(self, node):
                    self.classes.append(node.name)
                    self.generic_visit(node)
                    self.classes.pop()

                def visit_FunctionDef(self, node):
                    if node.name in API_NAMES or (self.classes and self.classes[-1] in METHOD_CLASSES and node.name in {'plan', 'run'}):
                        result['api_signatures'].append(dict(name='.'.join(self.classes+[node.name]),
                            line=node.lineno, signature=clipped('('+ast.unparse(node.args)+')', 240)))
                    self.generic_visit(node)

                visit_AsyncFunctionDef = visit_FunctionDef

            Visitor().visit(tree)
            result['api_signatures'] = result['api_signatures'][:16]
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'page_size':
                    if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and isinstance(node.comparators[0], (ast.List, ast.Tuple)):
                        values = node.comparators[0].elts
                        if all(isinstance(x, ast.Constant) and type(x.value) is int for x in values):
                            result['page_size_guards'].append(dict(line=node.lineno,
                                operator=type(node.ops[0]).__name__, values=[x.value for x in values][:32]))
            result['page_size_guards'] = result['page_size_guards'][:8]
        except (SyntaxError, ValueError, RecursionError) as exc:
            result['ast_error_type'] = type(exc).__name__
    lines, indices = text.splitlines(), set()
    for pattern in SNIPPET_PATTERNS:
        matches = [i for i, line in enumerate(lines) if pattern.search(line)][:3]
        for index in matches:
            indices.update(range(max(0, index-1), min(len(lines), index+2)))
    result['source_excerpt'] = clipped('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(indices)), MAX_SNIPPET)
    result['excerpt_qualification'] = 'Short selected source lines only; not a capture/precision/performance proof'
    return result


def safe_vendor_location(path):
    """Only module vendor origins, never user/config/cache directories."""
    parts = path.parts
    if any(part in {'.aws', '.ssh', '.config', '.cache', 'local-notes', 'credentials'} for part in parts):
        return False
    if len(parts) > 1 and parts[1] in {'Users', 'home', 'root', 'etc', 'proc', 'sys', 'dev'}:
        return False
    return path == Path('/opt/sglang') or str(path).startswith('/opt/sglang/') or 'site-packages' in parts or 'dist-packages' in parts


def resolve_allowroot(root):
    allowed = root.resolve(strict=True)
    if not safe_vendor_location(allowed):
        raise ValueError('Resolved allowroot is not a permitted vendor path')
    if root == Path('/opt/sglang') and allowed != root:
        raise ValueError('Fixed vendor allowroot resolves outside /opt/sglang')
    return allowed


class Inventory:
    def __init__(self, report, deadline_seconds):
        self.report, self.started = report, time.monotonic()
        self.deadline_seconds = deadline_seconds
        self.seen, self.snippet_bytes, self.signature_bytes = set(), 0, 0

    def check(self):
        if time.monotonic()-self.started > self.deadline_seconds:
            raise TimeoutError('CPU inventory deadline')

    def limit(self, reason):
        if reason not in self.report['limits_reached']:
            self.report['limits_reached'].append(reason)

    def missing(self, path):
        if len(self.report['missing_candidates']) < 64:
            self.report['missing_candidates'].append(clipped(path, 384))

    def skip(self, path, reason):
        if len(self.report['skips']) < 64:
            self.report['skips'].append(dict(path=clipped(path, 384), reason=reason))

    def inspect(self, path, root, *, stat_only=False):
        self.check()
        try:
            resolved, allowed = path.resolve(strict=True), resolve_allowroot(root)
            if not resolved.is_relative_to(allowed):
                self.skip(path, 'symlink_outside_allowroot')
                return
            if not safe_vendor_location(resolved):
                self.skip(path, 'protected_or_nonvendor_source_path')
                return
            if str(resolved) in self.seen:
                return
            if self.report['files_examined'] >= MAX_FILES:
                self.limit('max_files')
                return
            info = resolved.stat()
            if not stat.S_ISREG(info.st_mode):
                return
            self.seen.add(str(resolved))
            self.report['files_examined'] += 1
            row = dict(path=clipped(resolved, 512), size=info.st_size, read=False)
            if stat_only:
                row['skip_reason'] = 'stat_only_binary_no_hash'
            elif info.st_size > MAX_FILE:
                row['skip_reason'] = 'file_exceeds_2MiB'
            elif self.report['hashed_file_bytes']+info.st_size > MAX_BYTES:
                row['skip_reason'] = 'total_read_budget'
                self.limit('max_bytes')
            else:
                flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                descriptor = os.open(resolved, flags)
                with os.fdopen(descriptor, 'rb') as stream:
                    current = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(current.st_mode) or current.st_size != info.st_size
                            or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)):
                        raise ValueError('Vendor file changed before bounded read')
                    raw = stream.read(info.st_size)
                    after = os.fstat(stream.fileno())
                if len(raw) != info.st_size or after.st_size != info.st_size:
                    raise ValueError('Vendor file size changed during bounded read')
                self.report['files_read'] += 1
                self.report['hashed_file_bytes'] += len(raw)
                row.update(read=True, sha256=hashlib.sha256(raw).hexdigest(), **source_summary(raw, resolved.suffix))
                remaining = MAX_SNIPPETS_TOTAL-self.snippet_bytes
                row['source_excerpt'] = clipped(row['source_excerpt'], min(MAX_SNIPPET, remaining))
                self.snippet_bytes += len(row['source_excerpt'].encode())
                signatures = []
                for signature in row['api_signatures']:
                    count = len(encoded(signature))
                    if self.signature_bytes+count > MAX_SIGNATURES_TOTAL:
                        self.limit('signature_export_budget')
                        break
                    signatures.append(signature)
                    self.signature_bytes += count
                row['api_signatures'] = signatures
            self.report['files'].append(row)
            if len(encoded(self.report)) > MAX_JSON-8192:
                self.report['files'].pop()
                self.limit('json_export_budget')
        except FileNotFoundError:
            self.missing(path)
        except TimeoutError:
            raise
        except (OSError, ValueError) as exc:
            self.skip(path, type(exc).__name__)

    def walk_headers(self, start, root):
        self.check()
        try:
            start = start.resolve(strict=True)
            if not start.is_relative_to(resolve_allowroot(root)) or not start.is_dir():
                return
        except TimeoutError:
            raise
        except (FileNotFoundError, OSError):
            self.missing(start)
            return
        except ValueError:
            self.skip(start, 'invalid_vendor_allowroot')
            return
        pending = [(start, 0)]
        while pending and self.report['files_examined'] < MAX_FILES:
            self.check()
            if self.report['walk_directories'] >= MAX_WALK_DIRS:
                self.limit('walk_directories')
                break
            directory, depth = pending.pop(0)
            self.report['walk_directories'] += 1
            remaining = MAX_WALK_ENTRIES-self.report['walk_entries']
            if remaining <= 0:
                self.limit('walk_entries')
                break
            try:
                entries = []
                with os.scandir(directory) as scan:
                    for entry in scan:
                        if len(entries) >= remaining:
                            self.limit('walk_entries')
                            break
                        entries.append(entry)
                for entry in sorted(entries, key=lambda item: item.name):
                    self.check()
                    self.report['walk_entries'] += 1
                    if entry.is_symlink():
                        continue
                    if not safe_vendor_location(Path(entry.path)):
                        continue
                    if entry.is_dir(follow_symlinks=False) and depth < MAX_WALK_DEPTH:
                        pending.append((Path(entry.path), depth+1))
                    elif entry.is_file(follow_symlinks=False) and entry.name in HEADER_NAMES:
                        self.inspect(Path(entry.path), root)
            except TimeoutError:
                raise
            except OSError:
                continue


def initial_report():
    return dict(kind='CPU-only bounded vendor API/source eligibility inventory', status='starting', passed=False,
        started_utc=datetime.now(timezone.utc).isoformat(), worker_sha256=None,
        gpu_requested=False, cuda_context_requested=False, candidate_kernel_launches=0,
        package_imports_executed=[], modules={}, distributions={}, files=[], skips=[],
        missing_candidates=[], limits_reached=[], files_examined=0, files_read=0,
        hashed_file_bytes=0, walk_directories=0, walk_entries=0,
        limits=dict(max_files=MAX_FILES, max_read_bytes=MAX_BYTES, max_file_bytes=MAX_FILE,
            max_excerpt_bytes=MAX_SNIPPET, max_json_bytes=MAX_JSON,
            max_total_excerpt_bytes=MAX_SNIPPETS_TOTAL, max_total_signature_bytes=MAX_SIGNATURES_TOTAL,
            max_walk_depth=MAX_WALK_DEPTH, max_walk_directories=MAX_WALK_DIRS,
            max_walk_entries=MAX_WALK_ENTRIES),
        qualification='Module presence differs from distribution metadata; source APIs do not prove GPU eligibility or performance')


def collect_inventory(deadline_seconds=30, report=None):
    """Read-only API. No vendor package import, CUDA call, subprocess or network."""
    if not 0 < deadline_seconds <= 30:
        raise ValueError('CPU inventory deadline must be at most30 seconds')
    report = initial_report() if report is None else report
    inventory = Inventory(report, deadline_seconds)
    roots = {}
    for name in DISTRIBUTIONS:
        inventory.check()
        try:
            report['distributions'][name] = clipped(importlib.metadata.version(name), 128)
        except importlib.metadata.PackageNotFoundError:
            report['distributions'][name] = None
        except TimeoutError:
            raise
        except Exception as exc:
            report['distributions'][name] = dict(error_type=type(exc).__name__)
    for name in MODULES:
        inventory.check()
        try:
            spec = importlib.util.find_spec(name)  # Top-level only: never dotted discovery/import.
            if spec is None:
                report['modules'][name] = dict(found=False)
                continue
            locations = []
            for location in spec.submodule_search_locations or []:
                locations.append(location)
                if len(locations) == 2:
                    break
            proposed = Path(locations[0]) if locations else Path(spec.origin).parent if spec.origin else None
            resolved = proposed.resolve(strict=True) if proposed else None
            if resolved is None or not safe_vendor_location(resolved):
                report['modules'][name] = dict(found=True, source_allowed=False,
                    reason='origin_not_a_permitted_vendor_path', origin_omitted=True)
                continue
            roots[name] = resolved
            row = dict(found=True, source_allowed=True, root=clipped(resolved, 512))
            if spec.origin and spec.origin not in {'built-in', 'frozen'}:
                origin = Path(spec.origin).resolve(strict=True)
                if origin.is_relative_to(resolved):
                    row['origin'] = clipped(origin, 512)
            report['modules'][name] = row
        except TimeoutError:
            raise
        except Exception as exc:
            report['modules'][name] = dict(found=None, error_type=type(exc).__name__)
    for name, root in roots.items():
        for relative in RELATIVE_SOURCES[name]:
            inventory.inspect(root/relative, root)
    if 'sgl_kernel' in roots:
        inventory.inspect(roots['sgl_kernel']/'flash_ops.abi3.so', roots['sgl_kernel'], stat_only=True)
    if 'flashinfer' in roots:
        for relative in ('data/include/flashinfer/attention', 'data/csrc/xqa'):
            inventory.walk_headers(roots['flashinfer']/relative, roots['flashinfer'])
    for path in FIXED_VENDOR_DIRS:
        inventory.walk_headers(Path(path), Path('/opt/sglang'))
    report.update(status='complete', passed=True, elapsed_seconds=time.monotonic()-inventory.started,
        interpretation='Bounded inventory completed; missing/unread/limited source is not an eligible-kernel claim')
    if len(encoded(report)) > MAX_JSON:
        raise ValueError('CPU inventory report exceeds256KiB')
    return report


def save(output, report):
    raw = encoded(report)
    if len(raw) > MAX_JSON:
        raise ValueError('CPU inventory report exceeds256KiB')
    temporary = output.with_suffix('.tmp')
    if temporary.is_symlink() or output.is_symlink():
        raise ValueError('Output must not be a symlink')
    temporary.write_bytes(raw)
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=30)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline) or not 0 < args.deadline <= 30:
        parser.error('--deadline must be finite and in (0,30] seconds')
    report = initial_report()
    # The worker itself is an explicitly uploaded allowlisted file, not user data.
    report['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def alarm(*_):
        raise TimeoutError('CPU inventory30-second watchdog')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(min(30, args.deadline)))
    try:
        collect_inventory(args.deadline, report)
        save(args.output, report)
        return 0
    except Exception as exc:
        report.update(status='incomplete', passed=False, error_type=type(exc).__name__)
        save(args.output, report)
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
