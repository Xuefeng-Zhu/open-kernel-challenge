# Matched Gluon serial/overlap compiler experiment

**Question.** Can explicit Gluon asynchronous QK leave real next-tile gather,
conversion and shared-store work between the QK issue/commit and its dependency
wait, while retaining safe double-buffer ownership and the serial arm's legal
maximum CTA/SM? This is a compiler mechanism test, not a timing claim.

**Exact comparison.** One source, two `OVERLAP` specializations. Both use B32,
HKV2/HQ16, L8/G8, natural M64/N64/D64, page1, page-table stride86008, four
warps/one stage and grid(32,2). Both allocate the same two BF16 K slots and two
BF16 V slots, about32KiB total. Both stage the next tile identically; only its
position relative to the dependency-bearing current QK wait changes. No
threshold, split, stage, warp, precision or register-cap sweep is allowed.

**Math scope.** Q is unscaled post-RMS/RoPE BF16. The kernel rounds it to E4M3
then widens to BF16; cached E4M3 K/V widen to BF16. QK factor5.4730064863875
comes from the frozen official reference and is applied once after QK with
SM scale0.125 and live K scale. QK/PV use strict FP32 accumulation/state.
Both arms use zero-PV followed by `acc*alpha+PV`; this is not claimed bitwise
identical to the old TL carry-C lowering. Original-oracle validation is a later
mandatory gate. The standalone kernel omits production preparation, cache
append, output normalization/gating and FA3 dispatch; the working3da source
remains unchanged.

**Before import.** Native compilation must first verify exact installed
Triton3.8.0 and the CPU-inventoried Gluon init, Hopper and async-copy source
hashes. The installed Hopper hash is c427a1…a99dc, not assumed semantically
identical to a public tag. Any mismatch stops without discovery or fallback.

**Bounded compile only.** The worker defaults to CPU source preflight.
Explicit `--compile` permits exactly two JIT warmups and module-handle loads,
using deliberately undersized typed one-element pointers that must never be
executed. There is no kernel invocation, correctness run, timing, profiler,
network, credential or Modal code. The native worker deadline is90 seconds;
the root's separately reviewed controller governs actual resource admission.
No GPU job, reserve or submission follows merely from this card.

**Required mechanism gates, declared before artifacts.**

1. Actual asynchronous WGMMA QK, commit and dependency wait must survive lowering;
   readable exact SASS and TTGIR/PTX must identify the matching operations.
2. In the overlap arm, actual next-tile LDG/CVT/STSM must occur before the
   corresponding current QK wait; the serial arm supplies the matched control.
   Mnemonic counts alone do not prove tile identity or scheduling.
3. Actual shared addresses, slot alternation, fences and completed prior PV
   consumption must prove no buffer reuse while an outstanding consumer still
   owns it. QK/PV order and FP32 online-softmax state must remain matched.
4. Both arms must have zero spills/local-memory access and total shared memory
   at most64KiB. The overlap arm must not reduce the compiled legal maximum
   CTA/SM versus serial, using actual H200 device/function limits, block size,
   register/shared footprint and the supported CUDA occupancy metadata API.
   This maximum is not achieved residency. The old96-register cap does not
   apply to this independent M64 design.

Missing exact ISA, CUDA limits, occupancy metadata or unambiguous ownership
proof means **inconclusive, no execution admission**. Any known gate failure
rejects this exact pair without a precision, register-cap or geometry rescue.
No compile pass or static instruction inventory is reported as speedup.

**Artifacts.** Exactly two units, `serial.core` and `overlap.core`, retain TTGIR,
PTX and CUBIN; TTIR/LLIR are saved when emitted by Gluon. Existing cuobjdump
adds SASS when available, without installation. Source/harness/installed-source
hashes, register/shared/spill/local metadata and maximum CTA/SM queries are
retained. The complete result stays bounded2MiB, gzip artifact JSON8MiB and
each file4MiB. No readable SASS leaves the mechanism inconclusive.

Only after independent source/ISA/ownership/resource review could a separate
original-oracle standalone validation and matched uninstrumented timings be
considered. A future complete production comparison would have to include all
preparation, stores, compute, merge and output work. No such result is claimed.
