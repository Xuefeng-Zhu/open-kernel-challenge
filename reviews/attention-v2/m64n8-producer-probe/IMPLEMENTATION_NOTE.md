# Producer integration source note

This is an unexecuted source integration. Offline compilation establishes code
generation and resource use only; it does not establish numerical GPU correctness,
replay safety, performance, or official-judge binary identity.

The control is `fusion-lookahead-probe/candidate.py`, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The new function keeps its 21 runtime/formal scalar-pointer parameters and 16
constexpr parameters. The host must restrict dispatch to L1/G8/HQ16/HKV2/D64,
page size 1, M16/N64/new-token N16, separate merge and `SPLITS > 1`; the selected
producer remains the fallback. Page specializes to constexpr 1 in the offline
harness. No selected source or merge implementation is changed here.

History keeps Q's FP32 RMS/RoPE followed by BF16 → E4M3 → BF16. Cached K/V
encoded E4M3 values widen to BF16. QK is K[64,64] × Qᵀ[64,8]; FP32 max/sum
reduce axis 0. P rounds BF16, and PV is Vᵀ[64,64] × P[64,8], with FP32 online
carry. QK and PV request `max_num_imprecise_acc=0`. Scalar K scale enters
`(sm_scale * qk_factor) * ks * 1.4426950408889634`; V scale remains in the
unchanged merge.

Two raw shared slots each hold one FP8 K tile and one FP8 V tile. Both copies
are committed together for a real tile. Priming issues at most two real tiles;
the loop waits for the oldest tile, crosses a CTA barrier, loads and masks its
raw payload, publishes converted operands, and crosses a barrier before reusing
the consumed raw slot for tile i+2. `PAGE_LOOKAHEAD` retains one additional page
ID vector ahead of payload issue. A zero-outstanding copy drain and CTA barrier
run unconditionally before the tail, including empty history. QK completes
before score access; synchronous PV completes before next iteration's converted
operand overwrite. These are source invariants requiring native artifact review.

Logical shared operands occupy 34 KiB: two raw K and V slots (16 KiB total),
converted K/V (16 KiB), and Qᵀ/P (2 KiB). Compiler scratch and alignment must be
included in actual resource counts; this ledger is not an admission result.
Raw FP8 rows use an eight-element, eight-phase swizzle: the nominal XOR span
fits their 64-byte row. The Python API itself has no shape-aware swizzle
validation, so only native artifacts can establish actual physical allocation.

The last split always owns the new-token cache write and N16 tail. New K is
FP32 RMS/RoPE divided by K scale and directly encoded E4M3, with no extra BF16
rounding. New V is loaded BF16, widened FP32, divided by V scale and encoded
E4M3. N16 QK/PV use Ampere MMA with BF16 operands and FP32 accumulators. This
pinned `mma_v2` API has no explicit imprecise-accumulator argument; its BF16
semantic dot default is zero. The N dimension is not enlarged to 64. Only Q
rows and P's query rows are expanded to the original M16 shape. All gather
indices stay in bounds before padding is zeroed.

Valid G8 partial rows retain the selected split-major M16 allocation stride.
Ignored rows 8–15 are deliberately neutral; the selected control can have
nonneutral ignored history rows, and the unchanged merge never reads them.

API audit used the cached Triton 3.8.0 ARM64 wheel SHA256
`a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50`.
Its Gluon `gather` accepts equal-rank source/index tensors with matching
non-gather dimensions; its `expand_dims` requires the sliced layout axis to
match the expanded axis. `async_load` exposes no `other=0`, so invalid shared
rows are explicitly zero-selected after the copy wait. No driver APIs, inline
assembly, host tool installation, service calls, compilation or GPU execution
were performed by the source implementation agent.

CPU validation run by the source agent: the existing 12 contract/model tests and
15 then-current offline-harness tests passed. A separate symbolic FIFO/slot
check covered real history tile counts 0–257: each tile was consumed once, no
unread raw slot was overwritten, and the final drain left no pending group.
This model assumes the documented per-thread committed-group ordering and
cannot establish generated GPU memory ordering or cross-warp publication.
