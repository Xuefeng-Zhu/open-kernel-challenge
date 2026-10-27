# Revised V-stage native gate: publication contrast survives

**PASS for the prospective remaining-stage code-generation gate.** Both arms
prefetch raw K, and native shared publication remains on opposite sides of
the corresponding QK drain. They have not collapsed to identical relevant
schedules. This pass is narrower than the source labels: **V loads and some
conversion are hoisted before QK/wait in both binaries**, so there is no clean
V-gather-before-versus-after contrast. The old experiment's fully-serial gate
remains rejected. No kernel has executed; no correctness, achieved overlap,
latency or benefit over FA3 is established.

This offline report reviews TTGIR/PTX/SASS and verifies all ten artifact
sizes/hashes against[`artifacts/manifest.json`](artifacts/manifest.json).
There are five actual artifacts per arm and no TTIR. No source edit, GPU,
service or additional compilation was performed by this review.

## Binding and resource gate

Result SHA256:
`258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240`.
Commit`048bd62de599831edbb40b118df9a6047f6f1043`, app
`ap-XzBXIhHnYx5BgeBYIoyAkL`, image`im-cPTbGpNiX3RarKResPgX7h`.
Kernel SHA256`6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`;
config SHA256`686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0`.
These are the reviewed fixed M64/N64/D64,L8/G8,B32/HKV2/HQ16, four-warp,
one-stage arms. The actual page-table and history pointers are int32, not the
earlier harness's corrected-before-launch type. Result records zero launches,
no correctness/performance tests and `execution_admitted=false`.

| Actual property | v_after_wait | v_before_wait |
|---|---:|---:|
| Registers/thread |166|168|
| Shared bytes |32768|32768|
| Driver spill report / CUDA local bytes |0 /0|0 /0|
| PTX/SASS local load/store instructions |0|0|
| CUDA legal maximum CTA/SM,128 threads |3|3|

The before arm costs two more registers. This did not change the reported
legal CTA maximum; neither number measures achieved occupancy/residency.
There is no lost shared-buffer bank or observed spill/resource rejection.

## Common work and sole PTX scheduling difference

Line numbers refer to full artifact files. In both TTGIRs, QK is the BF16
asynchronous dot at129, next int32 page load135 feeds the same physical
offsets, and raw FP8 K loads at146. In`v_after_wait`, the dependent QK wait
at147 precedes V load152, K widening/store153–154, V widening/store155–156
and proxy fence157. In`v_before_wait`, V load151, K widening/store152–153,
V widening/store154–155 and fence156 precede the dependent wait157.

In both PTXs QK uses four
`wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16` operations at
1986/1993/1999/2005 and commit2008. Next-page int32 loads2032–2050 feed
raw-K loads2092–2134 **before the first wait in both arms**. After-arm wait
is2142, followed by V loads2169 onward and shared stores2526 onward. Before-
arm V loads2162 onward and all publication stores2519–2945 precede wait2955.

Remove debug `.loc`/`.file` lines, comments, blank lines and `$L__tmpN`
debug labels, take kernel text through its final`ret`, then remove the first
QK wait from each arm. The remaining2180 lines are identical, SHA256
`d87f98dd398fdf1b70e6d0e51150e2c0aa9c763a08de1b8ebea2edd96e1288d8`.
The PV wait is retained. Thus PTX logical loads, work, masks and arithmetic
match; a mnemonic count alone is not the basis for this conclusion.

## Actual native dataflow: remaining publication contrast

SASS scheduling differs from PTX/source order, as allowed by the new card.
The K/V global bases are separately identified by argument constants:
K`c[0][0x218]` and V`c[0][0x220]`, loaded to UR14/UR16. Both native arms
have next-page and payload prefetch before their QK drains. In particular,
the after arm's V load and part of its BF16 widening are already early.

| Native event | v_after_wait SASS | v_before_wait SASS |
|---|---|---|
| QK HGMMA PCs |`0x2a60/0x2be0/0x2c40/0x2c50`|`0x2d10/0x2d80/0x2da0/0x2e60`|
| Corresponding QK drain |line1531,`0x2fb0`|line1813,`0x3880`|
| First next K shared store |line1677,`0x3440`, after drain|line1565,`0x30c0`, before drain|
| Final next V shared store |line2331,`0x48b0`, after drain|line1801,`0x3820`, before drain|
| Publication fence |line2335,`0x48d0`|line1809,`0x3860`|

An actual V value demonstrates this is a producer dataflow difference:

* After: offset+Vbase at`0x2800`→LDG`R83` at`0x2830`→E4M3 unpack`0x2cd0`
  →BF16`R80/R81` at`0x2d20/0x2d30`, all before drain`0x2fb0`; packing
  at`0x4740`→V-bank store`0x4760` occurs **after** the drain.
* Before: offset+Vbase at`0x2890`→LDG`R84` at`0x28e0`→unpack`0x2de0`
  →BF16`R140/R139` at`0x2ec0/0x2f80`→pack`0x34f0`→V-bank store`0x3520`,
  all **before** drain`0x3880`.

The placement of complete shared publication therefore survives. Do not
relabel this as all V loading newly overlapped: much of that work is common
native prefetch. The compiler's other legal instruction motion/register
allocation is part of the two compiled schedules. See the independent
[`SASS_REVIEW.md`](SASS_REVIEW.md) for exact PC/line/register chains.

## Slots, precision and drain ordering

TTGIR retains two separate `2×64×64` BF16 K/V descriptors at82–83. Before-
arm PTX768 binds shared base`%r169`; current K is`base+(%r738<<13)` at
1894–1895 and current V adds16384 at1897. Current QK descriptors derive
from saved K base1900–1901/1983 onward. XOR2139 changes the slot; next base
2141–2142 feeds K stores2514–2519 and V adds16384 at2884/2916 onward.
PV descriptors instead derive from saved **current V**`%r397` at3327–3328
and3426–3447. Thus0/8192-byte slots and the separate V bank are retained;
publication addresses are not the current operands while QK is pending.
The after arm has the same address arithmetic and bank sizes.

TTGIR78–81 preserves BF16 Q→E4M3 round-to-nearest-even→BF16 widening.
The waited scores feed FP32 rowmax/exponentials/denominator state; P rounds
to BF16 at194 before BF16 PV196. The PTX QK/PV instructions explicitly use
BF16 inputs and FP32 accumulators, not an FP8 imprecise-accumulation rescue.
Both start each QK/PV result from zero C. PV commit3450 and wait3454 precede
common FP32 numerator FMAs3458–3489 and loopback3496 in both PTXs. FMA
contraction and usual approximate/FTZ exponential lowering are present,
identically in the normalized PTX; no bitwise-equivalence or IEEE-exact-exp
claim is made.

Native after-arm PV`0x4db0..0x5000` drains at line2607/`0x5150`, before
loopback2675/`0x5370`. Before-arm PV`0x4f80..0x50d0` drains at2593/`0x50e0`,
before loopback2667/`0x5330`. Current V is therefore drained before the next
iteration's slot reuse in these static paths. Full masked-zero stores and
publication fences remain. This is static ownership evidence, not a runtime
race test or measurement of concurrent instruction execution.

**Disposition:** the new prospective publication contrast/resource/dataflow
gate passes. Only a separately reviewed bounded correctness/replay experiment
can establish numerical and runtime safety; timing is a subsequent gate.
Neither a schedule difference nor CUDA legal occupancy establishes useful
latency overlap. Original tolerances, current full-call comparator3da and the
old rejected strict-control result remain unchanged. No official admission
or production promotion follows from this artifact review.

## Inspected identities

| File | SHA256 |
|---|---|
| v_after_wait.core.ttgir |21140684f37e8e47800a8e7ec39cbdba0cbb41f61ddc30bc23fe0d8a0be053e1|
| v_before_wait.core.ttgir |23fe6770de098234d86e3b443e49447039549fb9c8df5f39daf36eb0a4e95ae8|
| v_after_wait.core.ptx |5abb19407019c849938ac06852f64540d7df1b9e6901e377adaa11dfa42603a7|
| v_before_wait.core.ptx |8896430c4539d95a4f17d76791d0028dd0e45ad497e7a3d2631fb52fa90946d1|
| v_after_wait.core.sass |19341152847cfea894358c0c59c5ac5b202128bb4c9c08664e9ab5a7bfd70c90|
| v_before_wait.core.sass |4e32ac70a6b8fe76e28322eb5a889e22169edabc442daa05b017908b43fc58d6|
