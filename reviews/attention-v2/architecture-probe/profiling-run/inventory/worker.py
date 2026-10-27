"""CPU-only existing-tool inventory. No CUDA import, query, context or kernel.

Only path discovery, file metadata/hashes and existing executable version/help
commands are permitted. This cannot establish H200 counter permissions.
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

DEADLINE = 20
MAX_RESULT = 128_000
MAX_COMMAND_OUTPUT = 8_000
HASH_FILE_LIMIT = 32*1024*1024
HASH_TOTAL_LIMIT = 96*1024*1024
TOOL_PATTERNS = {
    'ncu': ['/opt/nvidia/nsight-compute*/ncu', '/opt/nvidia/nsight-compute*/*/ncu',
            '/usr/local/cuda*/nsight-compute*/ncu', '/usr/local/cuda*/bin/ncu'],
    'nsys': ['/opt/nvidia/nsight-systems*/bin/nsys', '/opt/nvidia/nsight-systems*/*/bin/nsys',
             '/opt/nvidia/nsight-systems*/*/target-linux-x64/nsys', '/usr/local/cuda*/bin/nsys'],
}
LIBRARY_PATTERNS = [
    '/usr/local/cuda*/extras/CUPTI/lib64/libcupti.so*',
    '/usr/local/cuda*/targets/x86_64-linux/lib/libcupti.so*',
    '/opt/sglang/lib/python*/site-packages/nvidia/cuda_cupti/lib/libcupti.so*',
    '/opt/sglang/lib/python*/site-packages/torch/lib/libcupti.so*',
    '/usr/lib/x86_64-linux-gnu/libcupti.so*',
]
COMMANDS = {'ncu': [('--version',), ('--help',), ('--list-sections',)],
            'nsys': [('--version',), ('profile', '--help')]}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def discover_tool(name):
    candidates = []
    found = shutil.which(name)
    if found:
        candidates.append(found)
    for pattern in TOOL_PATTERNS[name]:
        candidates.extend(sorted(glob.glob(pattern))[:16])
    result = []
    seen = set()
    for raw in candidates[:64]:
        path = Path(raw)
        try:
            resolved = path.resolve(strict=True)
            if resolved in seen or not resolved.is_file() or not os.access(resolved, os.X_OK):
                continue
            seen.add(resolved)
            result.append(dict(path=str(path), resolved=str(resolved), bytes=resolved.stat().st_size))
        except OSError:
            continue
    return result[:8]


def main(output):
    started = time.monotonic()
    report = dict(kind='CPU-only existing image tool inventory', status='starting',
        started_utc=datetime.now(timezone.utc).isoformat(), gpu_requested=False,
        cuda_context_requested=False, candidate_kernel_launches=0,
        hardware_counter_permission='unknown_not_tested_no_gpu',
        worker_sha256=digest(Path(__file__)), tools={}, cupti_libraries=[],
        distributions={}, limits=dict(worker_seconds=DEADLINE, command_timeout_seconds=4,
            result_bytes=MAX_RESULT, output_bytes_per_stream=MAX_COMMAND_OUTPUT),
        search_patterns=dict(tools=TOOL_PATTERNS, libraries=LIBRARY_PATTERNS))
    hashed_bytes = 0

    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        raw = json.dumps(report, separators=(',', ':'))+'\n'
        if len(raw.encode()) > MAX_RESULT:
            raise RuntimeError('Bounded inventory result exceeded limit')
        temp = output.with_suffix('.tmp')
        temp.write_text(raw)
        temp.replace(output)

    def gate():
        save()
        if time.monotonic()-started >= DEADLINE-0.25:
            raise TimeoutError('Inventory deadline')

    def alarm(*_):
        raise TimeoutError('Inventory alarm')

    def file_record(path):
        nonlocal hashed_bytes
        size = path.stat().st_size
        row = dict(path=str(path), bytes=size, sha256=None)
        if size <= HASH_FILE_LIMIT and hashed_bytes+size <= HASH_TOTAL_LIMIT:
            row['sha256'] = digest(path)
            hashed_bytes += size
        else:
            row['hash_omitted_reason'] = 'Bounded file/total hash-byte limit'
        return row

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(DEADLINE)
    save()
    try:
        for name in ('torch', 'triton', 'sglang', 'sgl-kernel', 'nvidia-cuda-cupti-cu13', 'nvidia-cuda-cupti-cu12'):
            gate()
            try:
                dist = importlib.metadata.distribution(name)
                meta = dist.read_text('METADATA')
                report['distributions'][name] = dict(version=dist.version,
                    metadata_sha256=hashlib.sha256(meta.encode()).hexdigest() if meta else None)
            except importlib.metadata.PackageNotFoundError:
                report['distributions'][name] = None
        for name in ('ncu', 'nsys'):
            gate()
            paths = discover_tool(name)
            row = dict(status='found' if paths else 'not_found_in_bounded_search',
                       candidates=paths, commands=[])
            report['tools'][name] = row
            if not paths:
                continue
            selected = Path(paths[0]['resolved'])
            row['selected_executable'] = file_record(selected)
            for args in COMMANDS[name]:
                gate()
                before = time.monotonic()
                timeout = min(4, max(0.1, DEADLINE-(before-started)-0.5))
                item = dict(argv=[str(selected), *args])
                row['commands'].append(item)
                try:
                    child = subprocess.run(item['argv'], capture_output=True, timeout=timeout,
                                           stdin=subprocess.DEVNULL)
                    for stream in ('stdout', 'stderr'):
                        raw = getattr(child, stream)
                        text = raw[:MAX_COMMAND_OUTPUT].decode(errors='replace')
                        clipped = len(raw)>MAX_COMMAND_OUTPUT
                        # Bound encoded JSON too (escapes/Unicode can expand).
                        while len(json.dumps(text).encode()) > MAX_COMMAND_OUTPUT:
                            text = text[:len(text)//2]
                            clipped = True
                        item[stream] = text
                        item[stream+'_bytes'] = len(raw)
                        item[stream+'_truncated'] = clipped
                    item['exit_code'] = child.returncode
                except subprocess.TimeoutExpired:
                    item.update(timed_out=True, timeout_seconds=timeout)
                item['wall_seconds'] = time.monotonic()-before
                save()
        seen = set()
        for pattern in LIBRARY_PATTERNS:
            gate()
            for raw in sorted(glob.glob(pattern))[:16]:
                path = Path(raw).resolve()
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                report['cupti_libraries'].append(file_record(path))
                if len(report['cupti_libraries']) >= 16:
                    break
            if len(report['cupti_libraries']) >= 16:
                break
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='partial_or_failed', error_type=type(exc).__name__)
    finally:
        signal.alarm(0)
        report['hashed_file_bytes'] = hashed_bytes
        save()
    return 0 if report['status']=='complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().output))
