# Independent native V-stage mechanism review

**PASS for the revised card's remaining conversion/shared-publication contrast.**
Actual V gathering is hoisted ahead of the QK drain in both arms; a V-load
serialization claim therefore fails. The native schedules nevertheless retain
a substantial publication-placement difference and do not collapse into one
relevant schedule. This conclusion uses the new prospective card's allowance
for partial hoisting; it does not change the earlier probe's failed strict
serial-control gate.

Only this local review was written. No kernel execution, GPU/service request,
network, compiler, profiler, credentials, checker edit or Git action occurred.
Numerical correctness, runtime races, performance and production eligibility
remain untested. The compile result itself records zero kernel launches and
`execution_admitted=false`.

## Source and artifact binding

- Frozen commit: `048bd62de599831edbb40b118df9a6047f6f1043`.
- Kernel SHA256: `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`.
- Config SHA256: `686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0`.
- Native result SHA256: `258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240`.
- Artifact manifest SHA256: `6f9db6cec8ff3c6293c48e8889732cf3c3ca0e6dfffee0b7cd5d243bdf66a97a`.
- Predeclared `../EXPERIMENT_CARD.md` SHA256:
  `209082108de413aeefb114c963096519451f60908934e12ea9aac15c30fefc1d`.

Exactly10 artifacts are present: TTGIR, LLIR, PTX, CUBIN and SASS for each arm;
there is no TTIR. Their sizes/hashes match the saved manifest, result inventory
and decoded bundle. A second independent local review verified compressed/raw
bundle hashes, payload pins, resources and precision. Both use the same
H200/SM90 image, Triton3.8.0, HQ16/HKV2, M=N=D64/L8/G8/page1, stride86008,
four warps/one compiler stage. `V_BEFORE_WAIT` is the only option delta.

## Native wait placement and actual V dataflow

The eight pointer parameter order binds Q/K/V/PageTable to constant-bank
offsets0x210/0x218/0x220/0x228. In both hot loops, K base is loaded into
UR14 from0x218 and V base into UR16 from0x220. Page-table entries produce
physical-row offsets; separate 64-bit additions of the K/V bases feed payload
LDGs. Thus the V loads below are payload, not merely page-table/address loads.

| Native stage | `v_after_wait` SASS offsets | `v_before_wait` SASS offsets |
|---|---|---|
| Current QK BF16/FP32 HGMMA group | 0x2a60–0x2c50 | 0x2d10–0x2e60 |
| Actual next V payload LDGs | 0x2830,0x28a0,0x2980,0x2ba0,0x2ca0,0x2d80,0x2f00,0x2f60 | 0x28e0,0x2990,0x2a20,0x2ae0,0x2b50,0x2c00,0x2c60,0x2ce0 |
| Corresponding QK `WARPGROUP.DEPBAR.LE ...0` | 0x2fb0 | 0x3880 |
| Next-slot K/V shared stores | 0x3440–0x48b0 | 0x30c0–0x3820 |
| Next-slot publication fence | 0x48d0 | 0x3860 |
| Current PV group | 0x4db0–0x5000 | 0x4f80–0x50d0 |
| PV drain, before loop backedge | 0x5150; backedge0x5370 | 0x50e0; backedge0x5330 |

All V payload loads precede the QK drain in **both** arms; in the before arm
they also precede QK issue. Some conversion work also hoists in the after arm:
V load0x2830 writes R83, and its E4M3 unpack at0x2cd0 plus BF16 conversions
at0x2d20/0x2d30 precede drain0x2fb0. It would be incorrect to label the
after arm a fully serialized V-stage control.

However conversion is not identically placed. For example after-arm V base
UR16 additions0x2920/0x2950 feed LDG R86 at0x2980; E4M3 unpacks from that
V payload remain at0x3210/0x32f0, **after** the QK drain. Before-arm matching
V fragment LDG R87 at0x2a20 unpacks at0x3270/0x32d0, **before** drain0x3880.
Most decisively, every next-slot STS and the publication fence remain on
opposite sides of the respective drain. The contrast is conversion/publication
scheduling with already-prefetched V, not a proof of hidden V memory latency.

## Matched work, precision and ownership gates

**Matched-work PASS.** Removing blank/comment/debug-location lines from PTX
yields2,513 lines in each arm. Removing only the first QK wait makes the entire
remaining normalized PTX **exactly identical**. Its sole delta is moving that
wait from after the common K prefetch to after the shared-publication fence.
There are51 global loads,48 shared stores,8 BF16 WGMMA instructions,2 commits
and2 waits per static kernel in both arms. This is an instruction-volume match,
not an execution-count/timing measurement. Native instruction scheduling and
register allocation differ, as expected; no artificial dependency or barrier
was inserted to force a slower arm.

**Precision PASS.** Both source dots request zero imprecise accumulation.
TTGIR and PTX/SASS show BF16 inputs and FP32 QK/PV accumulation (four QK and
four PV instructions); no FP8 MMA or TF32 lowering is present. Metadata listing
TF32 as a default/allowed option is not the actual instruction precision.
Q still rounds BF16→E4M3→BF16, K/V widen within the kernel, probabilities
round BF16, and FP32 softmax/max/sum/numerator updates match between arms.
PV starts with zero C, then common FP32 `acc*alpha+PV`; this is not a bitwise
claim versus the production implementation or a carry-C variant.

**Correct pointer ABI PASS.** Actual warmup pointer3/4 are int32. TTGIR declares
PageTable and HistoryLengths as `ptr<i32>`; page loads are signed-extended to
i64 only after reading. This differs from the old preserved int64-page-table
compile specialization and matches the current official generator contract.

**Static buffer/lifetime PASS; runtime race status UNKNOWN.** TTGIR retains
distinct mutable K and V allocations `[2,64,64]` BF16. Current QK/PV descriptors
index `%slot`; next stores index `%slot xor 1`, with V storage16KiB beyond K.
SASS retains the slot xor and8KiB per-slot offset. Masked gathers supply zero;
full shared stores overwrite every next tile, including final speculative
padding. The publication fence follows those writes. PV remains synchronous,
with the native zero-outstanding drain before accumulation consumption and
loop backedge, so the prior slot is drained before its later reuse. Current
QK descriptors are formed before the slot-state update; publication does not
overwrite current K/V. No third live slot or cross-iteration undrained PV is
introduced. Static inspection is not an observed concurrency/race certificate.

## Resource gate and execution status

| Resource | after | before |
|---|---:|---:|
| Registers/thread, driver and Triton agree |166|168|
| Dynamic shared bytes (static shared0) |32768|32768|
| Raw reported spills / driver local bytes |0 /0|0 /0|
| PTX local loads/stores; SASS LDL/STL |0|0|
| Threads/CTA |128|128|
| CUDA-permitted maximum CTA/SM |3|3|

**Resource PASS.** The register increase does not reduce the queried legal
maximum. This maximum is not achieved residency, occupancy or tensor utilization.
Both modules loaded and compile completed; neither kernel executed.

Saved reconciliation records app`ap-XzBXIhHnYx5BgeBYIoyAkL` stopped/tasks0 at
2026-10-26T00:10:53.666428UTC, superseding the initial stopping observation.
This is the parent's saved resource receipt, not a new service query here.

## Minimum onward evidence

This pass admits **review of a bounded numerical harness**, not automatic
execution or timing. The already prepared `../runtime/CARD.md` uses legal
page1/int32 fixtures, zero/partial histories, nonunit scales, two fresh mappings,
original complete-call oracle, same-address ABA, private graph outputs/caches,
and exact bytes between the two matched-math prototype arms. Its adapters keep
preparation, KV append, FP8→BF16 Q conversion and norm/gate inside each captured
call; the actual selected FA3 call remains a separate comparator. The harness
still requires final independent source/hash review and parent admission.

Any oracle, cache, replay, byte or isolation failure stops before timing.
Only after those gates may the predeclared uninstrumented nearby4096 A/B pairs
answer mechanism benefit and FA3 competitiveness separately. No scored80K,
B1/decode/sliding, official-score, novelty, award-gap or production replacement
claim follows from this compilation. No old failure or gate was rewritten.
