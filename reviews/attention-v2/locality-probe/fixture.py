"""Pure-stdlib page1 locality fixture plan; never imports a vendor or executes CUDA.

This is diagnostic input construction, not a candidate kernel or an optimization.
A full-capacity row permutation changes physical locality while preserving every
encoded K/V byte and every logical attention input, including unused pages.
"""
from array import array
from dataclasses import dataclass
import hashlib
import sys
from typing import Sequence

FRESH_SEEDS = (135831, 135832)
MAX_PLAN_PAGES = 4_000_000
MAX_SMALL_BYTES = 16 * 1024 * 1024
DIAGNOSTIC_CASES = (
    dict(name='sliding_B1_L1', batch=1, q_len=1, ctx=80000, heads='tp8',
         layer='sliding', page=1, kv_scale=1.0, spread='fixed', dist='soft'),
    dict(name='full_B16_L8', batch=16, q_len=8, ctx=80000, heads='tp8',
         layer='full', page=1, kv_scale=1.0, spread='agentic', dist='coherent'),
    dict(name='full_B32_L1', batch=32, q_len=1, ctx=80000, heads='tp8',
         layer='full', page=1, kv_scale=1.0, spread='agentic', dist='coherent'),
)


@dataclass(frozen=True)
class Page1LocalityPlan:
    batch: int
    width: int
    capacity_pages: int
    q_len: int
    cache_seqlens: tuple[int, ...]
    page_table_a: tuple[tuple[int, ...], ...]
    page_table_b: tuple[tuple[int, ...], ...]
    cache_loc_a: tuple[int, ...]
    cache_loc_b: tuple[int, ...]
    new_to_old: array
    old_to_new: array
    valid_pages: int


def _integer(value, label):
    if type(value) is not int:
        raise ValueError(label + ' must be a Python integer, not a coerced value')
    return value


def build_plan(page_table_a: Sequence[Sequence[int]], cache_seqlens: Sequence[int],
               cache_loc_a: Sequence[int], q_len: int,
               capacity_pages: int) -> Page1LocalityPlan:
    """Construct a full physical-row bijection for a page1 fixture.

    ``K_B[new] = K_A[new_to_old[new]]`` and analogously for V. Its inverse is
    ``K_A[old] = K_B[old_to_new[old]]``. Valid logical token j in request b maps
    to canonical physical row b*width+j. Remaining physical rows are matched in
    ascending order, so bytes of unused pages are preserved too.

    Inputs must already be CPU Python integer sequences (e.g. diagnostic
    tensor.tolist() outside captured calls). The plan copies table/location
    containers. Its two compact array('q') mappings are newly allocated,
    separate buffers and are read-only by caller convention.
    """
    q_len = _integer(q_len, 'q_len')
    capacity_pages = _integer(capacity_pages, 'capacity_pages')
    if q_len < 1:
        raise ValueError('q_len must be positive')
    if not 1 <= capacity_pages <= MAX_PLAN_PAGES:
        raise ValueError('capacity_pages outside bounded diagnostic plan')
    batch = len(page_table_a)
    if batch == 0 or len(cache_seqlens) != batch:
        raise ValueError('Nonempty batch and matching cache lengths required')
    width = len(page_table_a[0])
    if width < 1 or any(len(row) != width for row in page_table_a):
        raise ValueError('Page table must be nonempty and rectangular')
    lengths = tuple(_integer(v, 'cache_seqlens') for v in cache_seqlens)
    if any(n < q_len or n > width for n in lengths):
        raise ValueError('Each total length must include all new tokens and fit table width')
    if sum(lengths) > capacity_pages:
        raise ValueError('Live page prefixes exceed physical capacity')
    table_a = tuple(tuple(_integer(v, 'page_table') for v in row) for row in page_table_a)
    locations_a = tuple(_integer(v, 'cache_loc') for v in cache_loc_a)
    if len(locations_a) != batch * q_len:
        raise ValueError('cache_loc shape must be batch times q_len')

    new_to_old = array('q', [-1]) * capacity_pages
    old_to_new = array('q', [-1]) * capacity_pages
    seen_old = bytearray(capacity_pages)
    table_b = []
    for b, (row, total) in enumerate(zip(table_a, lengths)):
        target = []
        for j, old in enumerate(row):
            if j >= total:
                if old != -1:
                    raise ValueError('Unused page-table suffix must remain -1')
                target.append(-1)
                continue
            if not 0 <= old < capacity_pages:
                raise ValueError('Live page-table entry outside physical capacity')
            if seen_old[old]:
                raise ValueError('Duplicate live physical page, including cross-request alias')
            new = b * width + j
            if new >= capacity_pages:
                raise ValueError('Canonical live row outside preserved physical capacity')
            seen_old[old] = 1
            new_to_old[new] = old
            old_to_new[old] = new
            target.append(new)
        expected = tuple(row[total-q_len:total])
        if locations_a[b*q_len:(b+1)*q_len] != expected:
            raise ValueError('cache_loc does not match live new-token page-table positions')
        table_b.append(tuple(target))

    next_old = 0
    for new in range(capacity_pages):
        if new_to_old[new] != -1:
            continue
        while next_old < capacity_pages and seen_old[next_old]:
            next_old += 1
        if next_old == capacity_pages:
            raise ValueError('Internal unused-row permutation mismatch')
        new_to_old[new] = next_old
        old_to_new[next_old] = new
        next_old += 1
    # Validate the inverse itself, rather than inferring it from live prefixes.
    for new, old in enumerate(new_to_old):
        if old < 0 or old_to_new[old] != new:
            raise ValueError('Full-capacity row mappings are not inverse bijections')
    locations_b = tuple(old_to_new[old] for old in locations_a)
    return Page1LocalityPlan(batch, width, capacity_pages, q_len, lengths,
                             table_a, tuple(table_b), locations_a, locations_b,
                             new_to_old, old_to_new, sum(lengths))


def permute_rows_bytes(encoded: bytes | bytearray, row_bytes: int,
                       plan: Page1LocalityPlan, inverse: bool = False) -> bytes:
    """Apply opaque byte rows for SMALL CPU validation only, never full GPU caches."""
    row_bytes = _integer(row_bytes, 'row_bytes')
    if row_bytes < 1 or not isinstance(encoded, (bytes, bytearray)):
        raise ValueError('Positive row_bytes and an encoded byte buffer required')
    if len(encoded) != plan.capacity_pages * row_bytes:
        raise ValueError('Encoded capacity/row size mismatch')
    if len(encoded) > MAX_SMALL_BYTES:
        raise ValueError('Large caches require the runtime GPU permutation adapter')
    mapping = plan.old_to_new if inverse else plan.new_to_old
    result = bytearray(len(encoded))
    source = memoryview(encoded)
    for out_row, in_row in enumerate(mapping):
        result[out_row*row_bytes:(out_row+1)*row_bytes] = source[in_row*row_bytes:(in_row+1)*row_bytes]
    return bytes(result)


def causal_window_rows(total: int, q_len: int, query_index: int, window: int = 0) -> range:
    """Absolute logical keys for one current query; window0 means full attention."""
    for value, label in [(total, 'total'), (q_len, 'q_len'),
                         (query_index, 'query_index'), (window, 'window')]:
        _integer(value, label)
    if not (q_len >= 1 and total >= q_len and 0 <= query_index < q_len and window >= 0):
        raise ValueError('Invalid query/window geometry')
    last = total - q_len + query_index
    first = max(0, last - window + 1) if window else 0
    return range(first, last + 1)


def summarize_plan(plan: Page1LocalityPlan) -> dict:
    """Small provenance, not arrays, bytes, resource measurements or a benchmark."""
    return dict(batch=plan.batch, page_table_shape=[plan.batch, plan.width],
        capacity_pages=plan.capacity_pages, q_len=plan.q_len,
        cache_seqlens=list(plan.cache_seqlens), valid_pages=plan.valid_pages,
        unused_pages=plan.capacity_pages-plan.valid_pages,
        permutation_encoding=f'native-endian signed{plan.new_to_old.itemsize*8}; {sys.byteorder}',
        new_to_old_sha256=hashlib.sha256(plan.new_to_old.tobytes()).hexdigest(),
        old_to_new_sha256=hashlib.sha256(plan.old_to_new.tobytes()).hexdigest(),
        full_capacity_permutation=True, candidate_source_changed=False,
        runtime_same_addresses_verified=False, actual_hardware_locality_measured=False)
