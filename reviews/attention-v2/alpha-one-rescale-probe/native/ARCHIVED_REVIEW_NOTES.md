# HISTORICAL ONLY: A2-15 native comparison checklist

These notes inspect the previously saved A2-15 native artifacts, not the new
alpha-one candidate or its pending compile. They establish what to trace in
fresh control/candidate artifacts. They do not establish fresh compilation,
correctness, skip frequency, occupancy, dynamic local traffic or performance.
No source/harness/service/GPU action was performed for this review.

All filenames and line numbers below refer to
`reviews/attention-v2/a215/native_artifacts/`. Line numbers and register names
are historical anchors; fresh artifacts need semantic register/dataflow
mapping rather than literal name matching.

## Historical specialization and hashes

The historical producer is full decode L1/G8/HQ16/HKV2/D64, BLOCK_M16,
BLOCK_N64, BLOCK_L16, SPLITS11, CHUNK_SIZE8192, no RoPE/window,
SEPARATE_MERGE=true and PAGE_LOOKAHEAD=true; four warps, three stages,
Triton3.8.0, SM90 target. PTX declares version8.8/targetsm_90a. The summary
records `enable_fp_fusion=true` and `enable_reflect_ftz=true`; these flags do
not alone describe every instruction's denormal behavior.

| File | SHA256 |
|---|---|
| `summary.json` | `0c523d393189997884ab0f4c99a8b137279f92d66af357d7fb9074e46d629b69` |
| `a215.producer.ttgir` | `345f1ede702bdf065cd7002bcd637af5519ec86721aee8344a54e6280d9ab32a` |
| `a215.producer.llir` | `da82162dfa21a5f04e2a6540b938e40ae5a1d10e4bd9cb37634d798a5b470768` |
| `a215.producer.ptx` | `df4a5b499b5a4116bc35d0864187c2339d3ac5b62d213955384da4174b1f19e2` |
| `a215.producer.sass` | `44dc14bbc9a1d2ab246146c7c76d76c9cb7f13b929ad4c900f18c3f3f021e8de` |

The archived A2-15 `_fused` source segment is identical to the working
composition control's segment; segment SHA256 is
`df3aff9cac1cd345a44b8b80f7b983d0a524a41250cb880315745f460d2a1ccc`.
That makes the artifacts relevant historical lowering evidence, not a substitute
for compiling the exact fresh control alongside the new candidate.

## Historical carry, denominator and PV chains

| Layer | Exact historical anchor | Meaning to preserve in a fresh comparison |
|---|---|---|
| TTGIR | 338–343 | `alpha=exp2(old_m-safe_new_m)` and `P=exp2(scores-safe_new_m)` are distinct, unchanged computations |
| TTGIR | 344–350 | Denominator old_l×alpha plus the same row sum of P |
| TTGIR | 354–356 | Expand/broadcast alpha, then multiply old numerator carry |
| TTGIR | 357–360 | P rounds to BF16; BF16 PV `tt.dot` receives the rescaled old carry as **C**, not zero |
| PTX | 1151–1152 | Denominator `fma.rn.f32 %r716,%r716,%r385,%r420` and the corresponding `%r717/%r386/%r424` operation |
| PTX | 1504–1512 | Eight `mul.f32` old-carry components; alpha in `%r385/%r386` |
| PTX | 1539–1574 | BF16 PV `mma.sync...f32.bf16.bf16.f32`, with rescaled carry registers reused as C |
| SASS | `0x2590/0x25a0` | `MUFU.EX2 R53/R52`: the two per-thread alpha values |
| SASS | `0x25f0/0x2610/0x2630/0x2640` | `R28*=R53`, `R29*=R53`, `R30*=R52`, `R31*=R52` |
| SASS | `0x2b40/0x2b90/0x2bb0/0x2bd0` | Remaining carry maps `R35×R52→R27`, `R34×R52→R26`, `R33×R53→R25`, `R32×R53→R24` |
| SASS | `0x2c30/0x2c40`, then through `0x35e0` | BF16 PV HMMAs begin with C blocks R24/R28; the old carry participates throughout the existing matrix accumulation sequence |
| SASS | `0x3840/0x3850` | Denominator `FFMA R78,R53,R78,R36` and `FFMA R77,R52,R77,R41` |

For the strict numerator-only experiment, the intended fast C operand is the
unscaled old carry when every valid query row's computed alpha equals1. The
fallback C remains old carry×alpha. A separately accumulated PV-from-zero plus
final carry addition is **not** the historical operation order, even if its
real-number expression is identical. Trace C dependencies; opcode counts alone
cannot prove this property.

## Historical lookahead and new-token tail

* TTGIR223–224 allocates two K and two V tiles. At319, async wait permits two
  groups outstanding. The loop's QK dot is325, PV dot360, next page-ID loads
 371–372. The compiler stages tile i+2 rather than preserving literal source
  placement of page-ID loads before current QK.
* SASS current QK spans`0x1d90–0x20c0`. PV starts`0x2c30/0x2c40`; next page
  IDs load into R49/R48/R47/R46/R89/R62 at
  `0x2db0/0x2de0/0x2e30/0x2e50/0x2f10/0x2f60`.
  First dependent clamps occur at
  `0x35b0/0x35d0/0x35f0/0x3630/0x3600/0x3660` respectively.
  This is static load/use separation, not a cycle or overlap measurement.
* Tail denominator FMA remains at PTX2175–2176; its carry rescale is2178–2185
  followed by BF16 PV at2272/2277. Corresponding SASS tail PV ends at
  `0x4d60/0x4d70`, denominator FFMAs at`0x4dc0/0x4dd0`.
  The new experiment does not conditionally skip the new-token tail rescale.

A new branch may disrupt shared double-buffering or page-load scheduling even
without changing source lookahead logic. Fresh artifacts must retain or report
that change explicitly; equal source directives do not prove equal pipelines.

## Historical resource evidence only

The raw historical summary reports A2-15 producer96 registers/thread,
20,480 shared bytes and driver spill field0; merge27 registers,1,024 shared
bytes, raw spill field0. A2-13 producer was96 registers/12,544 shared bytes.
The old A2-15 producer PTX contains no `.local` allocation or `ld.local/st.local`
instruction, and its SASS contains no `LDL/STL` opcode under direct static
search. These observations do not measure dynamic traffic or achieved occupancy.
Global/profile scratch sizes in old metadata are0; that is not a statement
about the candidate's explicit attention partial tensors.

Fresh result review must quote its own resource metadata and static local-memory
searches, not carry these numbers forward as though remeasured. Register
increase, extra shared buffers or new local loads/stores are resource costs
that can invalidate the small numerator-rescale saving.

## Strict-gate violations and review order

1. Reject an unmasked all-row alpha predicate or replacing computed-alpha
   equality with rowmax equality, an epsilon, logit threshold, or deferred
   rescaling. Valid-row NaN alpha must not take the exact-one fast path.
2. Require an actual executable path that avoids carry multiplication. A
   reduction/equality check with all old multiplies still executed fails the
   mechanism. Distinguish mutually exclusive static copies from executing PV
   twice: trace control flow and per-path matrix work.
3. Reject PV-from-zero plus final add, altered carry-C arithmetic, changed BF16
   P/V conversion, lowered accumulation precision or changed matrix reduction
   order. Signed zero/subnormal and NaN behavior must not be waved away by
   claiming source algebra proves universal bitwise equality.
4. Denominator reduction/FMA, max/P/alpha computation, FP8 Q/K encoding,
   new-token normalization/store/tail, split ownership and merge remain control
   computations. New denominator reassociation or additional approximation is
   outside this experiment even if the original oracle might tolerate it.
5. Confirm PAGE_LOOKAHEAD-false paths stay unchanged; compare merge artifacts
   separately. Hash equality is strong evidence when exact; normalized register
   naming or debug-location differences require semantic inspection.
6. Only after numerical/dataflow scope passes, account for comparison/reduction,
   shuffles, barriers, branch/predication, instruction footprint, pipeline and
   registers/shared/local-memory costs. A compile pass is not timing admission.

No fresh `ARTIFACT_REVIEW.md` has been written. The exact pending artifacts,
when supplied by the parent, remain to be inspected against this checklist.
