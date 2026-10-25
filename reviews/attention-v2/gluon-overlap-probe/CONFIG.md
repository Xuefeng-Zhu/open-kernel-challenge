# Standalone matched scheduling probe

This is local, uncompiled source for an attention-core mechanism test. It has
no production `kernel_fn`, launch wrapper, dispatch edit, split/merge or formal
entry. The working fusion+lookahead source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
No kernel import, native compilation or GPU execution occurred on this Mac.

## Exact boundary and configuration

The kernel takes eight pointers in this order:

| Argument | Contiguous fixture contract |
|---|---|
| Q | BF16 `[B,8,HQ,64]`, **unscaled** normalized/position-processed Q, before applying `qk_factor`; the kernel rounds BF16→E4M3→BF16 before scale use |
| KCache, VCache | E4M3 `[physical_rows,HKV,64]`, already containing historical and eight new quantized tokens; each tile is gathered and widened to BF16 inside the kernel |
| PageTable | Live integer page1 mapping `[B,PT_STRIDE]`; valid logical keys map to valid physical rows; unused entries may be invalid and are masked |
| HistoryLengths | Live integer `[B]`, nonnegative historical key counts, excluding the eight query/new-key tokens; `history+8 <= PT_STRIDE` |
| KScale, VScale | GPU FP32 scalars; original key/value descales; finite inputs are required |
| Out | Separate writable BF16 `[B,8,HQ,64]` raw attention result |

The six constexprs are `HKV,HQ,PT_STRIDE,SM_SCALE,QK_FACTOR,OVERLAP`.
`config.py` defaults to B32,HKV2,HQ16,PT_STRIDE86008,SM_SCALE0.125 and
QK_FACTOR5.4730064863875 (the scale in `../official/reference.py`). Both arms use
grid`(32,2)`, four warps and one compiler stage. M=N=D64 and L=G8 are fixed in
source. There is no geometry sweep. Nonunit fixture K/V scales remain live
GPU inputs rather than compile constants.

This boundary intentionally isolates the attention core. RMS/RoPE preparation,
FP8 cache append, output normalization/gating and any production split/merge
are excluded from this microcore. They must be present inside captured/timed
whole calls before any eventual production performance claim. Prepopulated
fixture caches are inputs, not a proposed outside-call data cache. One CTA per
request/KV head scans the full history; this is not a proven competitive
parallel decomposition versus FA3.

## Single isolated change and source proof

Each CTA has two BF16 K slots and two BF16 V slots, nominally32768 bytes total.
Both arms first stage tile0, issue current asynchronous BF16 QK, stage the
next tile into the other slot, execute identical FP32 online softmax and
synchronous BF16 PV, then flip slots. Serial waits for QK before next staging;
overlap waits after staging. Ordinary global gathers/casts/shared stores are
used; this source makes no cp.async, TMA gather or runtime-overlap claim.

All M64 rows are valid: row`j*8+g` is token`j` and grouped head`g`. Logical
keys cover`[0,history+8)` and causal visibility is`key <= history+j`. Every
masked K/V payload load has zero fallback; every staged shared tile is fully
overwritten, including padding. The final next-tile stage is fully masked and
still zeroes the unused slot in both arms. Current K is not overwritten while
QK is pending; current V is drained by synchronous PV before slot reuse.

QK and PV both set `max_num_imprecise_acc=0`. QK uses BF16 operands after
required E4M3 rounding/widening; PV rounds P to BF16 and widens cached V to
BF16; rowmax, exponentials, denominator and numerator state are FP32. QK scale
is`(SM_SCALE*QK_FACTOR)*KScale*log2(e)`, and VScale is applied once after the
normalized numerator. Both arms compute PV from zero C, then
`acc*alpha+pv`. This matches the two probe arms' source operation order; it is
**not** a claim of bitwise equivalence to a different carry-C implementation,
FP8 FA3 arithmetic or the full reference. Native compiler fusion still needs
inspection and numerical execution tests.

Nine offline tests parse kernel AST without importing Triton. They specialize
both constexpr arms and prove identical loop AST after removing the sole
QK-wait placement difference; check exact APIs, casts, zero-padding/write
scope, fixed configuration and waits; exercise causal/tile boundaries and a
symbolic two-slot lifetime model. They pass. These are source contracts, not
compiled synchronization or GPU correctness certificates.

## Installed and primary-source binding

Actual CPU inventory result SHA256 is
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`;
image`im-cPTbGpNiX3RarKResPgX7h`, Triton3.8.0. Installed Hopper source SHA256
`c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc`
provides the inspected MMA/wait signatures. Inventory used no vendor imports.

The public [v3.8 WGMMA tutorial](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py#L478)
provides precedent for a leading two-buffer dimension with a 2D shared layout
and dynamic `.index()` selection. The public
[Hopper API](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py#L138)
returns a tensor for one wait dependency. These are API precedents, not proof
that this particular source compiles in the installed package.

## Exact compile-only gate

Later compilation must be separately serialized and source/hash bound. Compile
exactly the two fixed arms through supported warmup, without invoking either
kernel. Preserve TTGIR/PTX/cubin/SASS and actual register/shared/spill metadata.

1. Both arms must lower QK/PV to strict BF16→FP32 WGMMA, with the same logical
   MMA work and common arithmetic ordering. Lower precision is not a rescue.
2. The overlap arm must retain current QK outstanding across actual next-tile
   page/payload load, conversion and other-slot shared-store work. Serial must
   wait first. A hidden earlier wait, barrier serialization or identical native
   schedule falsifies this source schedule, even if compilation succeeds.
3. Prove two distinct live K/V slots, shared publication before MMA use, token
   dependency before score access and PV drain before reuse. Missing evidence
   is inconclusive; source lifetime comments are insufficient.
4. Report actual resources in both arms. Reject spills/local-memory accesses,
   lost buffering or unusable resource allocation. A register increase is a
   cost to report, not evidence of achieved occupancy or a runtime stall.

Passing this gate admits only a future original-tolerance numerical/replay
check. It does not admit timing or official quota. FA3's recoverable gap is
unknown and already includes asynchronous/persistent machinery; there is no
latency or whole-score forecast from this prototype.
