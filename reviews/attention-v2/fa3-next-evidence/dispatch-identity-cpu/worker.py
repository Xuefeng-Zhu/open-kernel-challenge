"""CPU-only vendor identity evidence. Never imports vendor packages or requests CUDA."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import signal
import stat
import struct
import time

EXPECTED = {
    'selected': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'reference': '386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',
}
MAX_REPORT = 256 * 1024
MAX_SOURCE = 128 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
MAX_BINARY = 1536 * 1024 * 1024
MAX_EXCERPTS = 16 * 1024
MAX_FILES, MAX_DIRS, MAX_ENTRIES, MAX_DEPTH = 96, 128, 2048, 5
MODULES = ('sgl_kernel', 'sglang', 'torch', 'triton')
DISTRIBUTIONS = ('sglang', 'sgl-kernel', 'torch', 'triton')
HEADERS = {'mainloop_fwd_sm90_tma_gmma_ws.hpp', 'mainloop_fwd_sm90.hpp',
    'flash_fwd_kernel_sm90.h', 'tile_scheduler.hpp', 'tile_size.h', 'flash_api.cpp',
    'flash.h', 'flash_fwd_launch_template.h', 'paged_kv.h', 'flash_prepare_scheduler.cu'}
FIXED_DIRS = (
    '/opt/sglang/sgl-kernel/csrc/attention',
    '/opt/sglang/sgl-kernel/3rdparty/flash-attention/hopper',
    '/opt/sglang/third_party/flash-attention/hopper',
    '/opt/sglang/3rdparty/flash-attention/hopper', '/opt/sglang/csrc/attention',
)
PATTERN = re.compile(r'PackGQA|pack_gqa|num_splits|CollectiveMainloopFwdSm90|'
    r'Element[PV]|ElementAccum|TORCH_LIBRARY|(?:\.def|m\.def)\s*\(|'
    r'PagedKVNonTMA|return_softmax_lse|fwd\.default|tile_size|TileShape')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode() + b'\n'


def vendor_path(path):
    path = Path(path)
    denied = {'.aws', '.ssh', '.config', '.cache', 'local-notes', 'credentials'}
    if any(part in denied for part in path.parts):
        return False
    if len(path.parts) > 1 and path.parts[1] in {'Users', 'home', 'root', 'etc', 'proc', 'sys', 'dev'}:
        return False
    return (path == Path('/opt/sglang') or str(path).startswith('/opt/sglang/')
            or 'site-packages' in path.parts or 'dist-packages' in path.parts)


def wrapper_ast(raw):
    tree = ast.parse(raw.decode('utf-8'))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == 'flash_attn_with_kvcache'), None)
    if fn is None:
        return {'status': 'wrapper_function_not_found'}
    args = fn.args.posonlyargs + fn.args.args
    defaults = dict(zip([a.arg for a in args][-len(fn.args.defaults):],
                        [ast.unparse(v) for v in fn.args.defaults])) if fn.args.defaults else {}
    defaults.update({a.arg: ast.unparse(v) if v is not None else None
                     for a, v in zip(fn.args.kwonlyargs, fn.args.kw_defaults)})
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'torch.ops.sgl_kernel.fwd.default']
    result = {'status': 'parsed_not_executed', 'defaults': {k: defaults.get(k)
              for k in ('num_splits', 'pack_gqa', 'sm_margin', 'ver', 'return_softmax_lse')},
              'native_call_count': len(calls)}
    if len(calls) == 1:
        native = [ast.unparse(v) for v in calls[0].args]
        result.update(native_positional_count=len(native), native_keywords=[v.arg for v in calls[0].keywords],
            native_positions_1based={k: native.index(k)+1 if k in native else None
                for k in ('scheduler_metadata', 'num_splits', 'pack_gqa')},
            return_flag_in_native_arguments=any('return_softmax_lse' in v for v in native)
                or any('return_softmax_lse' in ast.unparse(v.value) for v in calls[0].keywords))
    result['return_flag_name_lines'] = [n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id == 'return_softmax_lse']
    return result


def note_build_ids(raw, endian):
    """Parse GNU note records, bounded bytes supplied by caller."""
    found, pos = [], 0
    while pos + 12 <= len(raw):
        namesz, descsz, kind = struct.unpack_from(endian + 'III', raw, pos)
        pos += 12
        if namesz > len(raw)-pos:
            break
        name = raw[pos:pos+namesz].rstrip(b'\0'); pos += (namesz+3)&~3
        if descsz > len(raw)-pos:
            break
        desc = raw[pos:pos+descsz]; pos += (descsz+3)&~3
        if name == b'GNU' and kind == 3 and 0 < len(desc) <= 128:
            found.append(desc.hex())
    return sorted(set(found))


def elf_build_ids(read_at, size):
    """ELF program-note identity only; no loader, dlopen, or subprocess."""
    head = read_at(0, min(64, size))
    if len(head) < 16 or head[:4] != b'\x7fELF':
        return {'status': 'not_elf', 'build_ids': []}
    cls, data = head[4:6]
    if cls not in (1, 2) or data not in (1, 2):
        return {'status': 'unsupported_elf', 'build_ids': []}
    endian = '<' if data == 1 else '>'
    fmt = endian + ('HHIQQQIHHHHHH' if cls == 2 else 'HHIIIIIHHHHHH')
    need = 16 + struct.calcsize(fmt)
    if len(head) < need:
        return {'status': 'truncated_elf_header', 'build_ids': []}
    vals = struct.unpack_from(fmt, head, 16)
    phoff, phentsize, phnum = vals[4], vals[8], vals[9]
    expected = 56 if cls == 2 else 32
    if phnum > 2048 or phentsize < expected or phnum * phentsize > 128*1024:
        return {'status': 'program_header_bound', 'build_ids': []}
    found = []
    for index in range(phnum):
        offset = phoff + index*phentsize
        if offset > size or phentsize > size-offset:
            return {'status': 'invalid_program_header_range', 'build_ids': []}
        row = read_at(offset, phentsize)
        if cls == 2:
            kind, _, noteoff, _, _, count, _, _ = struct.unpack_from(endian+'IIQQQQQQ', row)
        else:
            kind, noteoff, _, _, count, _, _, _ = struct.unpack_from(endian+'IIIIIIII', row)
        if kind != 4:
            continue
        if count > 256*1024 or noteoff > size or count > size-noteoff:
            return {'status': 'note_segment_bound', 'build_ids': []}
        found.extend(note_build_ids(read_at(noteoff, count), endian))
    return {'status': 'program_notes_parsed', 'elf_class': 32 if cls == 1 else 64,
            'endianness': 'little' if data == 1 else 'big', 'build_ids': sorted(set(found)),
            'qualification': 'GNU build identity, not source revision or executed specialization'}


class Audit:
    def __init__(self, seconds=45):
        self.start, self.seconds = time.monotonic(), seconds
        self.source_bytes = self.binary_bytes = self.excerpt_bytes = 0
        self.seen = set()
        self.report = {'files': [], 'binary_files': [], 'skips': [], 'limits_reached': [],
                       'walk_directories': 0, 'walk_entries': 0}

    def check(self):
        if time.monotonic()-self.start > self.seconds:
            raise TimeoutError('CPU metadata deadline')

    def skip(self, path, reason):
        if len(self.report['skips']) < 128:
            self.report['skips'].append({'path': str(path)[:512], 'reason': reason})

    def limit(self, value):
        if value not in self.report['limits_reached']:
            self.report['limits_reached'].append(value)

    def allowed(self, path, root):
        self.check()
        root, path = Path(root).resolve(strict=True), Path(path).resolve(strict=True)
        if not vendor_path(root) or not vendor_path(path) or not path.is_relative_to(root):
            raise ValueError('nonvendor_or_outside_allowroot')
        return path

    def read_source(self, path, root):
        try:
            path = self.allowed(path, root)
            if str(path) in self.seen:
                return
            if len(self.report['files']) >= MAX_FILES:
                self.limit('source_file_count'); return
            observed = path.stat(); size = observed.st_size
            if size > MAX_FILE or self.source_bytes+size > MAX_SOURCE:
                self.limit('source_byte_bound'); return
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as f:
                before = os.fstat(f.fileno())
                if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (observed.st_dev,observed.st_ino,observed.st_size,observed.st_mtime_ns):
                    raise ValueError('source_changed_before_open')
                raw = f.read(size+1); after = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != size:
                raise ValueError('file_changed_or_not_regular')
            self.seen.add(str(path)); self.source_bytes += len(raw)
            text = raw.decode('utf-8', errors='replace')
            row = {'path': str(path), 'bytes': size, 'sha256': sha(raw), 'text_matches': []}
            if path.name == 'flash_attn.py':
                row['wrapper_ast'] = wrapper_ast(raw)
            for no, line in enumerate(text.splitlines(), 1):
                if PATTERN.search(line):
                    excerpt = f'{no}: {line}'[:800]
                    if self.excerpt_bytes+len(excerpt.encode()) > MAX_EXCERPTS:
                        self.limit('excerpt_bytes'); break
                    row['text_matches'].append(excerpt); self.excerpt_bytes += len(excerpt.encode())
                    if len(row['text_matches']) >= 24: break
            if path.name in {'METADATA', 'PKG-INFO'}:
                match = re.search(r'^Version:[ \t]*([^\r\n]{1,128})\r?$', text, re.MULTILINE)
                row['distribution_version'] = match.group(1) if match else None
            self.report['files'].append(row)
            return row
        except TimeoutError:
            raise
        except (OSError, ValueError, SyntaxError, StopIteration) as exc:
            self.skip(path, type(exc).__name__)

    def read_binary(self, path, root):
        try:
            path = self.allowed(path, root); observed = path.stat(); size = observed.st_size
            if len(self.report['binary_files']) >= 16 or self.binary_bytes+size > MAX_BINARY:
                self.limit('binary_hash_bound'); return
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as f:
                before = os.fstat(f.fileno())
                if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (observed.st_dev,observed.st_ino,observed.st_size,observed.st_mtime_ns):
                    raise ValueError('binary_changed_before_open')
                if not stat.S_ISREG(before.st_mode): raise ValueError('not_regular')
                digest = hashlib.sha256()
                while True:
                    self.check(); chunk = f.read(min(1024*1024, size-f.tell()))
                    if not chunk: break
                    digest.update(chunk); self.binary_bytes += len(chunk)
                def read_at(offset, count):
                    self.check()
                    if self.source_bytes+count > MAX_SOURCE: raise ValueError('ELF metadata byte bound')
                    self.source_bytes += count
                    f.seek(offset); raw = f.read(count)
                    if len(raw) != count: raise ValueError('short_elf_read')
                    return raw
                identity = elf_build_ids(read_at, size)
                after = os.fstat(f.fileno())
            if (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('binary_changed')
            self.report['binary_files'].append({'path':str(path),'bytes':size,'sha256':digest.hexdigest(),
                'elf':identity,'qualification':'File identity only; library not loaded or executed'})
        except TimeoutError:
            raise
        except (OSError, ValueError, struct.error) as exc:
            self.skip(path,type(exc).__name__)

    def walk(self, start, root):
        try: start = self.allowed(start, root)
        except TimeoutError: raise
        except (OSError, ValueError): self.skip(start,'missing_or_outside_vendor'); return
        pending = [(start,0)]
        while pending:
            self.check()
            if self.report['walk_directories'] >= MAX_DIRS:
                self.limit('walk_directories'); break
            directory, depth = pending.pop(0); self.report['walk_directories'] += 1
            remaining = MAX_ENTRIES-self.report['walk_entries']
            if remaining <= 0: self.limit('walk_entries'); return
            try: entries = sorted(itertools.islice(directory.iterdir(),remaining+1),key=lambda p:p.name)
            except TimeoutError: raise
            except OSError: self.skip(directory,'directory_unreadable'); continue
            for path in entries:
                self.report['walk_entries'] += 1
                if self.report['walk_entries'] > MAX_ENTRIES:
                    self.limit('walk_entries'); return
                if path.is_symlink(): continue
                if path.is_dir() and depth < MAX_DEPTH: pending.append((path,depth+1))
                elif path.is_file() and path.name in HEADERS: self.read_source(path,root)


def bind_sources(selected, reference):
    result = {}
    for label, path in [('selected',selected),('reference',reference)]:
        path = Path(path)
        if path.is_symlink() or path.stat().st_size > 256*1024:
            raise ValueError('Source binding refused')
        digest = sha(path.read_bytes())
        if digest != EXPECTED[label]: raise ValueError('Source binding mismatch: '+label)
        result[label] = digest
    return result


def run(selected, reference, seconds=45):
    a = Audit(seconds)
    report = {'kind':'CPU-only installed FA3 file identity audit','started_utc':datetime.now(timezone.utc).isoformat(),
        'source_bindings':bind_sources(selected,reference),'gpu_requested':False,'cuda_context_requested':False,
        'vendor_packages_imported':[],'kernel_launches':0,'modules':{},'distributions':{},
        'runtime_schema':'UNKNOWN: extension not loaded','resolved_pack_gqa':'UNKNOWN: no native call',
        'resolved_split_counts':'UNKNOWN: no native call','executed_tile':'UNKNOWN: no kernel execution',
        'internal_QK_PV_dtype':'UNKNOWN: no matched runtime binary disassembly/dataflow'}
    roots = []
    for name in MODULES:
        a.check()
        try:
            spec = importlib.util.find_spec(name)
            origin = Path(spec.origin).resolve() if spec and spec.origin else None
            if origin is None or not vendor_path(origin):
                report['modules'][name]={'found':bool(spec),'readable_vendor_origin':False};continue
            root = origin.parent; roots.append(root)
            report['modules'][name]={'found':True,'origin':str(origin),'readable_vendor_origin':True}
            a.read_source(origin,root)
            if name == 'sgl_kernel':
                a.read_source(root/'flash_attn.py',root)
                binaries = sorted(itertools.islice(root.glob('flash_ops*.so'),17))
                if len(binaries) > 16: a.limit('binary_candidate_count')
                for binary in binaries[:16]: a.read_binary(binary,root)
                for sub in ('include','csrc','third_party','3rdparty'): a.walk(root/sub,root)
        except TimeoutError:
            raise
        except (OSError,ValueError,ImportError) as exc:
            report['modules'][name]={'status':type(exc).__name__}
    for name in DISTRIBUTIONS:
        a.check()
        try:
            dist = importlib.metadata.distribution(name)
            location = getattr(dist, '_path', None)
            if location is None:
                report['distributions'][name] = {'version':None,'status':'unsupported_metadata_provider'}
                continue
            location = Path(location)
            row = a.read_source(location/'METADATA', location)
            report['distributions'][name] = {'version':row.get('distribution_version') if row else None,
                'metadata_sha256':row.get('sha256') if row else None,
                'status':'bounded_metadata_file' if row else 'metadata_not_read'}
        except TimeoutError: raise
        except importlib.metadata.PackageNotFoundError:report['distributions'][name]={'version':None,'status':'metadata_not_found'}
        except (OSError,ValueError,TypeError) as exc:report['distributions'][name]={'version':None,'status':type(exc).__name__}
    for path in FIXED_DIRS:a.walk(Path(path),Path('/opt/sglang'))
    a.check()
    report.update(a.report,status='complete',source_bytes_read=a.source_bytes,binary_bytes_hashed=a.binary_bytes,
        source_excerpt_bytes=a.excerpt_bytes,elapsed_seconds=time.monotonic()-a.start,
        qualification='Bounded source and file identity only; missing source is not universal absence or a runtime dispatch claim')
    if len(encoded(report)) > MAX_REPORT:raise ValueError('Report exceeds256KiB; no truncation')
    return report


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-source',type=Path,default=root/'reviews/attention-v2/fusion-lookahead-probe/candidate.py')
    parser.add_argument('--reference-source',type=Path,default=root/'reviews/attention-v2/official/reference.py')
    parser.add_argument('--deadline',type=int,default=45)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1<=args.deadline<=45:parser.error('deadline must be1..45seconds')
    def alarm(signum,frame):raise TimeoutError('45second CPU-only watchdog')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.deadline)
    try:report=run(args.selected_source,args.reference_source,args.deadline)
    except Exception as exc:
        report={'status':'failed','reason':type(exc).__name__,'gpu_requested':False,'kernel_launches':0}
    finally:signal.alarm(0)
    raw=encoded(report)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(raw)
    else:print(raw.decode(),end='')
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
