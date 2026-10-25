"""Existing CUDA-driver metadata queries and conservative static resource gates.

No driver library is loaded at import. Occupancy is a compiled legal maximum,
not an observed resident-CTA trace or a performance claim.
"""
import ctypes
import hashlib
from pathlib import Path
import re

MAX_SHARED_BYTES = 64 * 1024
DEVICE_ATTRIBUTES = {
    'warp_size': 'CU_DEVICE_ATTRIBUTE_WARP_SIZE',
    'max_threads_block': 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'max_threads_sm': 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR',
    'max_registers_block': 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK',
    'max_registers_sm': 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR',
    'max_shared_sm': 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
    'max_shared_block_optin': 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
    'max_blocks_sm': 'CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR',
}
FUNCTION_ATTRIBUTES = {
    'static_shared_bytes': 'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    'local_bytes': 'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES',
    'registers_per_thread': 'CU_FUNC_ATTRIBUTE_NUM_REGS',
}


def header_enums(text):
    result = {}
    for name in [*DEVICE_ATTRIBUTES.values(), *FUNCTION_ATTRIBUTES.values()]:
        match = re.search(r'\b'+re.escape(name)+r'\s*=\s*(\d+)\b', text)
        if match:
            result[name] = int(match.group(1))
    return result


def inspect_resources(kernel, num_warps, triton_root):
    """Query already-installed driver APIs after warmup/module load; never launch."""
    result = dict(status='inconclusive', kernel_launched=False,
                  qualification='CUDA occupancy legal maximum, not achieved residency')
    try:
        headers = [Path('/usr/local/cuda/include/cuda.h'),
            Path('/usr/local/cuda/targets/x86_64-linux/include/cuda.h'),
            Path(triton_root)/'backends/nvidia/include/cuda.h']
        selected = next((path for path in headers if path.is_file() and path.stat().st_size <= 4*1024*1024), None)
        if selected is None:
            raise ValueError('Existing CUDA enum header unavailable')
        raw = selected.read_bytes()
        enums = header_enums(raw.decode('utf-8'))
        required = set(DEVICE_ATTRIBUTES.values()) | set(FUNCTION_ATTRIBUTES.values())
        if set(enums) != required:
            raise ValueError('Exact existing CUDA enum values unavailable')
        result.update(header=str(selected), header_sha256=hashlib.sha256(raw).hexdigest())
        driver = ctypes.CDLL('libcuda.so.1')
        current = driver.cuCtxGetDevice
        current.argtypes, current.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        device = ctypes.c_int()
        if current(ctypes.byref(device)) != 0:
            raise ValueError('Current CUDA device metadata unavailable')
        get_device = driver.cuDeviceGetAttribute
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_device.restype = ctypes.c_int
        limits = {}
        for key, name in DEVICE_ATTRIBUTES.items():
            value = ctypes.c_int()
            if get_device(ctypes.byref(value), enums[name], device.value) != 0 or value.value <= 0:
                raise ValueError('Actual CUDA device limit unavailable: '+key)
            limits[key] = value.value
        handle = ctypes.c_void_p(int(kernel.function))
        get_function = driver.cuFuncGetAttribute
        get_function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        get_function.restype = ctypes.c_int
        attributes = {}
        for key, name in FUNCTION_ATTRIBUTES.items():
            value = ctypes.c_int()
            if get_function(ctypes.byref(value), enums[name], handle) != 0 or value.value < 0:
                raise ValueError('Actual CUDA function attribute unavailable: '+key)
            attributes[key] = value.value
        dynamic_shared = int(kernel.metadata.shared)
        threads = num_warps*limits['warp_size']
        occupancy = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
        occupancy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        occupancy.restype = ctypes.c_int
        blocks = ctypes.c_int()
        code = occupancy(ctypes.byref(blocks), handle, threads, dynamic_shared)
        if code != 0 or blocks.value <= 0 or blocks.value > limits['max_blocks_sm']:
            raise ValueError('Compiled maximum CTA/SM unavailable')
        result.update(status='complete', device_limits=limits, function_attributes=attributes,
            threads_per_block=threads, dynamic_shared_bytes=dynamic_shared,
            total_shared_bytes=attributes['static_shared_bytes']+dynamic_shared,
            static_max_cta_per_sm=blocks.value,
            dynamic_shared_basis='Triton launch metadata.shared; CUDA API accounts for static function shared allocation')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, reason=str(exc)[:500])
    return result


def instruction_inventory(ptx, sass):
    """Presence/counts only; next-tile identity and wait ownership need review."""
    return dict(
        ptx_wgmma_async=len(re.findall(r'\bwgmma\.mma_async\.', ptx)),
        ptx_wgmma_commit=len(re.findall(r'\bwgmma\.commit_group\.', ptx)),
        ptx_wgmma_wait=len(re.findall(r'\bwgmma\.wait_group\.', ptx)),
        ptx_local_access=len(re.findall(r'\b(?:ld|st)\.local\b', ptx)),
        sass_hgmma=len(re.findall(r'\bHGMMA(?:\.|\s)', sass)),
        sass_ldg=len(re.findall(r'\bLDG(?:\.|\s)', sass)),
        sass_cvt=len(re.findall(r'\bF2F(?:\.|\s)|\bF2FP(?:\.|\s)|\bCVT(?:\.|\s)', sass)),
        sass_stsm=len(re.findall(r'\bSTSM(?:\.|\s)', sass)),
        sass_shared_store=len(re.findall(r'\b(?:STS|STSM)(?:\.|\s)', sass)),
        sass_local_access=len(re.findall(r'\b(?:LDL|STL)(?:\.|\s)', sass)))


def resource_gate(rows):
    """Predeclared resource gate; unknown observations never pass."""
    reasons, unknown = [], []
    by_arm = {row.get('arm'): row for row in rows}
    if len(rows) != 2 or set(by_arm) != {'serial', 'overlap'}:
        return dict(status='inconclusive', reasons=['Exactly two matched arms required'], execution_admitted=False)
    for arm, row in by_arm.items():
        observed = row.get('cuda_resources', {})
        if observed.get('status') != 'complete':
            unknown.append(arm+': exact CUDA resource metadata unavailable')
        if row.get('driver_spills_raw') != 0:
            (unknown if row.get('driver_spills_raw') is None else reasons).append(arm+': spill metadata nonzero or unavailable')
        if row.get('registers') is None:
            unknown.append(arm+': registers unavailable')
        if not row.get('sass_available'):
            unknown.append(arm+': readable exact SASS unavailable')
        if observed.get('status') == 'complete':
            if observed['total_shared_bytes'] > MAX_SHARED_BYTES:
                reasons.append(arm+': total shared memory exceeds64KiB')
            if observed['function_attributes']['local_bytes'] != 0:
                reasons.append(arm+': CUDA local-memory bytes nonzero')
            if observed['function_attributes']['registers_per_thread'] != row.get('registers'):
                unknown.append(arm+': compiler/driver register metadata differ')
        inventory = row.get('instruction_inventory', {})
        if inventory.get('ptx_local_access', 0) or inventory.get('sass_local_access', 0):
            reasons.append(arm+': local-memory instructions present')
    serial, overlap = by_arm['serial'].get('cuda_resources', {}), by_arm['overlap'].get('cuda_resources', {})
    if serial.get('status') == overlap.get('status') == 'complete':
        if serial['device_limits'] != overlap['device_limits'] or serial['threads_per_block'] != overlap['threads_per_block']:
            unknown.append('Matched actual device/block limits differ')
        elif overlap['static_max_cta_per_sm'] < serial['static_max_cta_per_sm']:
            reasons.append('Overlap reduces statically permitted CTA/SM')
    return dict(status='reject' if reasons else 'inconclusive' if unknown else 'pass',
        reasons=reasons+unknown, execution_admitted=False,
        qualification='Resource pass alone is not instruction-order, ownership, correctness or performance admission')
