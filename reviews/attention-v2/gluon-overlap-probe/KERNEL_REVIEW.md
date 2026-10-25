# Independent matched Gluon attention-core source review

**PASS for local source/configuration and the isolated two-arm contract.**
No concrete source-level causal, scale, buffer-lifetime or arm-isolation blocker
was found. Nine CPU tests passed independently without importing Triton or the
kernel. Native API/layout acceptance, synchronization lowering, resource use
and numerical behavior remain untested; those are explicit compile/execution
gates, not properties established here. Only this report and the separate
architecture-gate clarification were written by this reviewer. No GPU, service,
network, credentials or Git operation was performed.

| Final reviewed artifact | SHA256 |
|---|---|
| `kernel.py` | `3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c` |
| `config.py` | `ecaf6588893ad4ce6bc57944a9932018f511113aeb13a6ee266edcf536d701ec` |
| `test_source.py` | `143e80c007c311ff3d89e807d4b3b455c4fe4fabe24b6c73b97ee0092d4a57d2` |
| `CONFIG.md` | `e0061148b2b77cb278282c56baac27930c7e5be3d0e9348dff4ce5142ca01db1` |

## Narrow input/output boundary

The14-argument API consumes unscaled, already-normalized BF16 Q, a live page1
map and historical lengths, E4M3 K/V caches already populated with history plus
eight new tokens, GPU FP32 K/V scales, and a separate raw BF16 output buffer.
It does not perform Q normalization, cache append or output norm/gate, and is
not a production `kernel_fn`. Those operations cannot be omitted from any later
captured/timed complete-call comparison. No production dispatch is edited.

Final default configuration is B32,HKV2,HQ16,PT_STRIDE86008,SM_SCALE0.125,
QK_FACTOR5.4730064863875, grid(32,2), four warps and one compiler stage.
Both arms use fixed M=N=D64 and L=G8. This is natural verify geometry;
there is no M8-decode padding claim or split/warp/stage search.

The Q input must exclude QK_FACTOR. The kernel explicitly rounds
BF16→E4M3→BF16 before the factor enters the logit scale. Historical and new
cached K/V are gathered from live physical page mappings and widened to BF16
inside the kernel; no whole-cache repack or prior-call plan is introduced.
KScale multiplies logits exactly once, with log2(e) for `exp2`; VScale is applied
once to the final normalized numerator.

## Causal mapping and arithmetic order

All64 logical query rows are valid: `m=j*8+g`, token`j=m//8`, grouped query
head`kvh*8+g`. Q/Out offsets map each `(request,token,head,dimension)` once.
The key loop covers `[0,history+8)` once in64-token tiles. Visibility requires
`key<=history+j`, so every query sees all historical keys and exactly its causal
new-token prefix, including zero-history and partial-tile cases. CPU models
check endpoints around55/56/63/64/65 and8191/8192.

Both arms use the same FP32 rowmax, alpha, probability, denominator and numerator
recurrence; P is rounded to BF16 for PV, and QK/PV explicitly request
`max_num_imprecise_acc=0`. Both QK and PV start their matrix result from zero C.
The common numerator update is then `acc*alpha+pv`. This is matched between the
two probe arms; it is not a bitwise-equivalence claim to another carry-C MMA
implementation, native FP8 FA3, or the original complete reference. FP32
association and compiler fusion still require native inspection and original-
tolerance numerical validation before a production claim.

## Sole source difference and shared-buffer lifecycle

Specializing OVERLAP=False/True and removing the one dependency-bearing QK
wait yields identical loop ASTs. Serial issues current QK, waits, then performs
the next gather/cast/store. Overlap issues the identical QK, performs precisely
that next gather/cast/store into the other slot, then waits. Both read scores
only after waiting for `score_pending`; score math, synchronous PV and slot flip
remain in the same order. There is no QK/PV iteration reordering.

There are two BF16 K slots and two BF16 V slots,32768 nominal source bytes.
Initial tile0 is staged before its first issue. While current QK reads current
K, next staging writes the other K/V slot. The previous iteration's PV was
synchronous before that slot became reusable. Current PV likewise drains
before the next flip. This symbolic lifetime holds for one, two, several and
partial final tiles; there is no source overwrite of a consumed current slot.

Every masked page/K/V load uses a zero fallback, and both shared tile stores
write the full64×64 tile without a store mask. Padding therefore does not retain
stale V values that could contaminate PV. The last speculative next stage is
fully masked and zeroes the otherwise unused slot in both arms. Source uses
`fence_async_shared` after stores; actual shared publication, compiler-inserted
barriers and async proxy ordering must be demonstrated by emitted artifacts.

The source owner checked pinned Triton v3.8 tutorial/API precedent for a leading
two-buffer shared allocation with dynamic `.index()`, and for one-dependency
wait returning one tensor. The installed CPU inventory confirms the Hopper API
signatures. These are appropriate source precedents, not proof this exact
layout, permutation or wait expression compiles in the installed package.

## Required native falsifier and interpretation

The first native step remains the separately reviewed **compile-only** pair:
no attention invocation or timing. Both exact arms must compile without a
precision or geometry rescue. Inspect TTGIR/PTX/SASS for strict BF16/FP32 MMA,
the same matrix/arithmetic work, the delayed dependency-bearing QK wait across
actual next-tile gather/conversion/other-slot stores, correct publication and
safe PV drain. Record actual registers/shared/spills/local traffic; nominal
source buffers are not an occupancy measurement. Hidden waits, serializing
lowering, lost buffers, spills or unusable allocation can falsify viability.

Even favorable assembly establishes only an implementable independent schedule,
not observed runtime overlap or a missing FA3 feature. Subsequent numerical/
replay checks need source-bound legal fixtures and original precision/bounds.
Only later uninstrumented complete-call comparisons against both this serial
arm and the current working FA3 path can support production benefit. The
serial arm is a causal control, not a representation of FA3, which already has
staged/asynchronous/persistent machinery.

Exact FA3 C++ source remaining unknown does not block this local primitive
viability probe. It does prevent inferring novelty or a production advantage
from the implementation alone. No official quota, promotion, full-family gain
or decode applicability follows from this source review.
