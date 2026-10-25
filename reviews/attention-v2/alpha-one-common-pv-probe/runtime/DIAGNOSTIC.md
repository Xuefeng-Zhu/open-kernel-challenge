# Separate historical alpha-frequency diagnostic

`diagnostic.py` is a generated, independent copy of exact candidate
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.
Its SHA256 is
`ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60`.
It does not mutate a loaded module or modify the uninstrumented candidate.
It is neither a timing candidate nor an official-check source.

The five controlled source patches add only:

1. A `stats_ptr` parameter after `out_ptr` in `_fused`.
2. Three scalar `tl.int32` zero counters before the historical loop.
3. Integer increments after the unchanged valid-row `all_alpha_one` predicate:
   total+=1, fast+=predicate, slow+=1−predicate.
4. Three non-atomic stores to the CTA's own row after the historical loop and
   before the unchanged new-KV tail, including zero stores for empty splits.
5. Passing `data['alpha_stats']` from the instrumented `_e2_kernel_fn` call.

`test_diagnostic.py` reverses those patches to recover every candidate byte
and its complete AST. Original floating-point expressions, predicate, K/V
loads/stores, tail, dispatch, geometry, precision and merge are unchanged.
Five offline tests also check counter shape/type, noninterference with float
expressions, ownership and an independently enumerated CPU tile-count model.
These tests do not establish native numerical equivalence of instrumentation.

## API and ownership

The entrypoint remains `kernel_fn(data)`. The harness supplies one extra,
fresh, private contiguous CUDA `alpha_stats` tensor with dtype `torch.int32` and shape
`[batch * num_kv_heads * SPLITS, 3]`. It is a designated writable diagnostic
output, not an immutable attention input. Column order is `[total, fast, slow]`.

CTA `(b, kvh, split)` owns row
`((b * HKV + kvh) * SPLITS + split)` and writes exactly its three scalar slots.
No atomics, persistent cache, inter-CTA reduction or shared global counter is
added. Every launched CTA overwrites its own row on every replay; empty history
splits write `[0,0,0]`. Counters describe historical tiles only, excluding the
last-split new-token tail.

The bounded diagnostic cases are full B32/L1 without RoPE, so the existing
`PAGE_LOOKAHEAD` flag is true, CHUNK_SIZE8192 and BLOCK_N64. Other dispatch
configurations are not part of this diagnostic count contract.

For historical length `seq` and split `s`, the CPU expected total is

```python
span = max(0, min(seq, (s + 1) * 8192) - s * 8192)
expected_total = (span + 63) // 64
```

For every CTA require `0 <= fast <= total`, `0 <= slow <= total`,
`fast + slow == total == expected_total`, with matching rows across KV heads
for **total** only. Fast/slow may differ by KV head because Q/K values differ.
The sum across splits for one request/KV head is `ceil(seq/64)`. Maximum total
per chunk is128, far below int32 overflow. The host model covers tile/chunk
boundaries and empty/partial split tails by explicit Python range enumeration.
Prefill/reset every slot with a negative sentinel before a diagnostic launch
and verify every slot becomes nonnegative. This detects unwritten empty-CTA
rows, which an all-zero initialization alone would not distinguish from valid
zero statistics. Kernel counters initialize internally and never read the buffer.

## Required use before interpreting frequency

Run this module only outside every benchmark interval. First compare each
complete diagnostic output/new-KV and whole K/V-cache bytes against the exact
uninstrumented candidate on the same fixture already checked by the original
oracle. Preserve fixed addresses, immutable inputs and untouched cache rows.
Only `alpha_stats` may be excluded from the immutable-input view, and only for
the diagnostic runner; no exclusion applies to uninstrumented benchmark arms.
Allocation, reset/restore and statistics retrieval remain outside timed calls.

If instrumentation changes any observed output/cache byte or violates the
original oracle/replay contract, reject those statistics and stop the timing
branch. Do not substitute relaxed tolerances or infer correctness from source
identity. The worker/controller own these native gates and source/transport
bindings; this file alone neither launches nor admits a run.

Reported frequency is the fraction of visited historical CTA tiles whose
computed alpha is exactly1 for **all valid query rows**. It is not a row-wise
frequency, probability-mass estimate, skipped-key ratio or count of instruction
issue cycles saved. Fresh compile artifacts showed predicated FMULs, so even
a high fast fraction cannot prove a speedup. Only the separate uninstrumented
complete-call paired experiment can test benefit.
