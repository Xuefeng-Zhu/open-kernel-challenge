# Per-call compact KV for sliding verify: local source preparation

This probe preserves the exact selected fusion-plus-lookahead source as control,
SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The candidate adds a representation change for L8, sliding2048, RoPE500000,
page1, HQ16/HKV2/D64, with any batch size. Every other configuration reaches the
unchanged baseline dispatch. The old entrypoint is renamed without changing
its body; all original imports, functions, comments and license are retained.

The unchanged fused preparation first writes original live K/V and FP8 Q.
One new Triton kernel then copies exact uint8 K/V representations through the
live page table to per-call contiguous FP8 scratch. It uses fixed 32-token by
128-byte tiles, four warps, with no sweep. It derives start and valid length
on device from `cache_seqlens`, so CUDA graph replay consumes live metadata.

For total length h+8 and capacity2055, start=max(h+8-2055,0) and
packed length=min(h+8,2055). This is the union of the eight query windows;
**FA3 must still retain causal=True and window_size=(2047,0)**. Disabling
windowing would incorrectly include early keys for later queries. Absolute
RoPE is already applied before copying, and relative query/key differences
are preserved by the common offset. Scratch capacity is written completely
on each invocation, including zero padding, with no cross-call cache.

The new FA3 adapter differs in exactly four keyword arguments: K scratch,
V scratch, page_table=None, and live packed cache lengths. Scale handling,
Q, cu_seqlens_q, max query length, softmax factor, causal/window arguments,
scheduler_metadata, automatic split/packing choices and output norm/gate
remain the original source expressions. Source code preserves choices supplied
to FA3; its internal dispatch can change with representation and remains untested.

Extra scratch is two [B,2055,2,64] FP8 buffers plus B int32 lengths:
16,834,560+128 bytes at B32. The gather adds one launch and65 CTAs per request
(2080 at B32). Allocation, preparation, original cache updates, copying,
scheduling, attention and gate remain inside the complete timed call.
No contiguous-loader/TMA eligibility, resource usage or benefit is assumed.

Seventeen CPU tests pass: six AST/shape-dispatch/metadata checks and eleven independent
geometry/opaque-byte/address/ABA checks. They prove source preservation,
intended argument changes, exact mathematical key-set correspondence and
copy-address models. They do not compile/import candidate/Triton/Torch or
validate device execution, API acceptance, numerical accuracy or speed.

Run from the repository root:

    python3 -B -m unittest discover -s reviews/attention-v2/sliding-compact-kv-probe -p 'test_*.py' -q

One import-time helper records bounded scalar Torch/CUDA versions and the native
operator's schema, without invoking that operator or inspecting inputs. Schema
UTF8 exceeding4096 bytes produces a constant rather than a truncated record;
version fields are bounded to128 bytes. Failures print a fixed unavailable
marker, never exception contents. CPU tests exercise these guards with a fake
operator; no actual installed runtime metadata has been obtained locally.

No GPU, network, authentication, official job or Git operation was performed
by this local source preparation. Existing selected and rollback records are
unchanged. Original oracle checks and an actual-best whole-call comparison
remain mandatory before any promotion; the independent experiment card records
prospective scoring and acceptance limits.
