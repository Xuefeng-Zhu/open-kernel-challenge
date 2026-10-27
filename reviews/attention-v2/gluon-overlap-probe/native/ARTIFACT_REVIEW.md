# Native artifact review: surviving schedule, imperfect serial control

**A native K/V conversion/shared-store scheduling difference survives, with
strict BF16/FP32 MMA and both shared slots intact. The literal predeclared
serial-before-all-gather gate is not met:** ptxas hoists the serial arm's next
page-table and K payload loads before its QK wait. Treat this as a qualified
primitive viability result, not a clean fully serialized control or automatic
admission to execution/timing. No kernel ran; correctness, achieved overlap,
runtime races, latency and benefit over FA3 remain untested.

This offline review reads the ten actual artifacts (five per arm: TTGIR,
LLVM IR, PTX, cubin, SASS). No TTIR was emitted; the twelve-file transport cap
is not the artifact count. All ten sizes and SHA256s match
[`artifacts/manifest.json`](artifacts/manifest.json). No service, GPU operation
or algorithm/source edit was performed in this review.

## Binding and resources

Result SHA256:
`8b5f8717c56de7a6ea6971840bd4f2252aced16c86e1fb3aca26ed6af1a859ea`.
Frozen commit`605e815462b56768fca9c84c68a8aada639bbb0b`, app
`ap-AECevTBOrJ7vVRE4EWgk3R`, image`im-cPTbGpNiX3RarKResPgX7h`.
Kernel SHA256`3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c`;
configuration SHA256`ecaf6588893ad4ce6bc57944a9932018f511113aeb13a6ee266edcf536d701ec`.
Both compile the fixed B32/HKV2/HQ16/M64/N64/D64, four-warp, one-stage pair,
PT_STRIDE86008,SM_SCALE0.125,QK_FACTOR5.4730064863875. OVERLAP is the sole
constexpr difference. Driver module loading/resource queries are not attention
kernel execution; result explicitly records zero launches and no correctness
or performance tests.

| Reported native property | Serial | Overlap |
|---|---:|---:|
| Registers/thread |167|168|
| Shared bytes |32768|32768|
| Driver spill report / CUDA local bytes |0 /0|0 /0|
| PTX/SASS local load/store instructions |0|0|
| CUDA legal maximum CTA/SM at128 threads |3|3|

The one-register increase is a real cost. The unchanged legal CTA maximum
is not achieved residency, occupancy or a bottleneck diagnosis. No resource
collapse or spill was found; no register threshold was silently relaxed.

## IR and PTX dataflow, rather than instruction counts

Line numbers below refer to complete files under`artifacts/`.

* `overlap.core.ttgir:77–80` retains BF16 Q load, round-to-nearest-even E4M3
  conversion, then BF16 widening. Lines81–82 allocate separate
  `2×64×64` BF16 K and V shared descriptors. Lines124–127 use current slot K
  for an asynchronous BF16×BF16→FP32 QK result.
* In overlap, the exact QK result`%score_pending_68` is waited at154, after
  next-page load136, K load146→widen147→other-slot store148, V load150→
  widen151→other-slot store152 and publication fence153. In serial, the same
  result is waited at`serial.core.ttgir:128`, before those operations137–154.
  This proves the intended high-level dependency and other-slot writes.
* PTX overlap lines1989/1996/2002/2008 issue four
  `m64n64k16.f32.bf16.bf16` QK operations, with zero C in the first and the
  same FP32 accumulator in later operations; commit2011 groups them. Next
  page loads2040–2058 feed physical-address construction2062–2095, K payload
  loads2104 onward and E4M3→F16→BF16 conversion; next K stores begin2460.
  V uses the same physical offsets with the separate cache base; its final
  next-slot store is2947, proxy fence2951, and corresponding QK wait2957.
  Serial PTX instead places that wait at2016, before next page/payload loads.
* Buffer addresses are explicit in overlap PTX: shared base`%r165` is loaded
  at768; current K is`base+(%r723<<13)` at1897–1898, and current V adds16384
  at1900. The current K MMA descriptors derive from that saved base via
 1903–1904/1986–2008. Slot XOR2013 precedes a **different** next base2017–2018;
  next K writes derive from it at2455–2460 and next V adds16384 at2886 onward.
  Thus slot offsets are0/8192 bytes in separate K/V banks. PV derives its
  descriptor from saved **current** V (`%r386`) at3329–3330/3428–3449, not from
  the toggled next base.

The native SASS corroborates the remaining schedule difference but also shows
why PTX order alone is insufficient:

| Dataflow boundary | Serial SASS line / PC | Overlap SASS line / PC |
|---|---|---|
| Current QK HGMMA group |1299/`0x2870` through1387/`0x2b30`|1301/`0x2880` through1323/`0x2930`|
| Example next-K payload LDG |1333/`0x2980`, before wait|1351/`0x2a10`, before wait|
| Example next-V payload LDG |1467/`0x2db0`, after wait|1461/`0x2d80`, before wait|
| First next BF16 shared store |1703/`0x3510`, after wait|1641/`0x3320`, before wait|
| Final next V shared store |2365/`0x49c0`, after wait|1803/`0x3830`, before wait|
| Publication fence |2369/`0x49e0`|1807/`0x3850`|
| Corresponding QK drain |1455/`0x2d50`|1833/`0x3920`|

Two actual overlap value chains bind load, conversion and store. K:
LDG`R132` at`0x2a10`→E4M3 unpack`0x2f30`→BF16`R133/R76` at
`0x2f40/0x2f60`→pack`0x32d0`→STS`0x3320`. V: LDG`R80` at
`0x2d80`→unpack`0x3170`→BF16`R139/R140` at`0x3210/0x3230`→
pack`0x35a0`→V-bank STS`0x35d0`. Both chains precede QK drain`0x3920`.
Overlap's slot ULEA at`0x27f0`, bit flip`0x2f10` and new ULEA`0x2f20`
match the8192-byte other-slot address model; V stores add`0x4000`.
See the independent [`SASS_REVIEW.md`](SASS_REVIEW.md) for exact register
chains and line evidence. SASS's printed descriptor operands are not treated
as an independently complete descriptor decode; the IR/PTX binding above is
part of this conclusion.

Page-table loads are hoisted before QK in **both** binaries. Serial's K loads
also execute before its QK drain, while its widening and shared stores occur
afterward. Its native schedule is therefore already partially prefetched.
The useful observed contrast is the remaining V-load/conversion/shared-store
placement; it is not "all prefetch versus none." No posthoc threshold change
or replacement label turns the strict original control criterion into a pass.

## Arithmetic, synchronization and limits of proof

A stronger comparison than mnemonic counts is available at PTX: remove only
debug `.loc`/`.file` lines, comments, empty lines and `$L__tmpN` debug labels,
take the kernel text through its first final`ret`, then remove the **first QK
wait instruction only** from each arm. The resulting2189 stripped lines are
identical, SHA256
`3c9d56d78a167e20a7d0c033321b24686ce32e485e50cf4f208f9475fd9d3d05`.
The second PV wait remains in this comparison. Thus all remaining PTX math,
payload loads, masks, stores and logical work have matching instruction order;
ptxas scheduling/register allocation differs afterward.

TTGIR overlap154–190 completes QK before score scaling, causal mask, FP32
rowmax/exp2 and FP32 denominator reduction/update. Lines191–193 round P to
BF16 and perform BF16 PV with zero C. PTX PV3431–3449 uses the same explicit
BF16-input/FP32-accumulator instruction type, commits3452 and drains3456.
The common numerator update is then FMA3460–3491, and loopback3498 follows
that drain. Contraction to FMA is present in both arms; this is not separate
multiply/add bitwise equivalence or a changed carry-C MMA claim. FP32 exp2
lowers through the usual approximate/FTZ operations, identically in this PTX
comparison; no IEEE-exact exponential or runtime tolerance pass is claimed.

SASS overlap PV`0x4ec0..0x5110` drains`0x5250`; its output numerator FFMAs
begin`0x5260`, then loopback`0x54a0`. Serial PV`0x4ef0..0x5150` drains
`0x52a0`, before its corresponding numerator use and loopback`0x54c0`.
Both current-buffer reuse paths therefore follow PV completion in the static
control flow. Shared stores, proxy fences and CTA barriers remain visible.
The [PTX completion rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)
distinguish proxy publication from WGMMA group completion; this review does
not infer dynamic overlap duration from the absence of an earlier group wait.
The MMA may naturally finish before later staging work, and intervening
barriers/scheduling still have runtime costs.

**Disposition:** compilation, explicit datatypes, matched PTX work, resource
feasibility and safe static slot addressing are supported. A meaningful native
stage-placement difference is supported. The strict fully serial native
control is not retained; the original gate must receive an explicit parent
disposition before any onward phase. Neither a performance pass nor an FA3
feature gap follows. All original numerical/replay checks remain pending;
production candidate/dispatch and selected baseline are unchanged.

## Artifact identities used

| File | SHA256 |
|---|---|
| serial.core.ttgir |80a0a6fe838e5551098b1f285667dfca23ddf78f97b259ac099b53d30e2089f5|
| overlap.core.ttgir |e3763bc15b64897a192a8cdc9b7498c61a5ffecf7ecb17d988bd0ade8c677ebc|
| serial.core.ptx |f11faa6ad3d8c14248e61811b7c991ed95899ff473f06f057a154e284f45ede9|
| overlap.core.ptx |7767e6065cc191c3aabffc1865ee13e9a82e778a69105cf490df57176410565a|
| serial.core.sass |63ef626f4e871b35d49d57bfa43740237383c850f1d87f3952fa903c1f2baadf|
| overlap.core.sass |d334f93fc988fcb5f6fc657cdffaa470e6502efef66b779c2f7dd7961156de06|

LLVM IR and cubin hashes/sizes are preserved and verified in the manifest;
this report's conclusions above are tied to the inspected TTGIR/PTX/SASS.
