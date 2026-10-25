# Independent source review: conditional numerator rescale, common PV

**Source review passes for a separately compile-gated probe.** The exact current
candidate changes only the historical-loop numerator update on the existing
PAGE_LOOKAHEAD route. It does not establish native lowering, numerical results,
resource admission or performance. No GPU, network, service, credential, harness
or candidate edit was performed by this reviewer; this is the only file written.

| Exact source | SHA256 |
|---|---|
| Control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Common-PV candidate | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |
| Earlier failed split-PV candidate, preserved separately | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |

## Exact isolation proved locally

The independent AST check locates the new historical PAGE_LOOKAHEAD branch by
its `all_alpha_one` assignment, validates its three statements and exact
predicate, and replaces only that branch with its literal else statement.
The resulting **entire module AST** equals the control AST, ignoring source
locations. Thus the denominator, max/P/alpha calculations, new-token tail,
address lookahead, precision, partial/merge, host dispatch and every other
function remain control computations. This is stronger than comparing only a
few selected expressions. The PAGE_LOOKAHEAD-false source path is the literal
original numerator expression.

For PAGE_LOOKAHEAD=true, the exact new source sequence is:

1. Sum `(row_ok & (alpha != 1.0)).to(int32)` across query rows, requiring0.
2. Only when that condition fails, multiply old `acc` by broadcast alpha.
3. Execute one common `acc + tl.dot(P_BF16,V)` after the nested conditional.

There is no PV call inside the nested rescale condition, no separate PV-result
temporary, no zero-C PV assignment, and no changed denominator expression.
The common source call alone does not prove the compiler retains one common
PV machine block or the required carry-C dataflow.

## Why the strict predicate is defensible

It compares the **already computed FP32 alpha** exactly with1, not a new
approximation, epsilon, max-equality condition or deferred-rescale threshold.
If the existing approximate exponential rounds a small nonzero exponent to1,
the control would multiply by that same computed1. No additional softmax
approximation is introduced by this test.

`row_ok` explicitly excludes padded query rows. The existing host guard is
unchanged: full B32/L1 without RoPE, including the admitted generic head/page/
scale scope. In the TP8 specialization L1/G8/BLOCK_M16, eight query rows are
valid. Input/history-column masking remains unchanged. Empty historical splits
do not execute this loop. On an ordinary finite first tile, the initial−Inf
row maximum gives alpha0, so original rescaling occurs. Any valid-row alpha
that is not exactly1, including NaN under unordered inequality lowering, must
take the original rescale path. The new compiler artifacts must confirm that
this predicate survives lowering.

For finite FP32 carry, multiplication by positive1 is exact and preserves the
sign of zero under the control instruction's behavior. Skipping only that
multiplication can therefore supply identical carry bits to unchanged PV MMA.
Padded-row carries can differ, but the separate merge reads only valid G rows;
the unchanged per-row softmax/dot operations do not mix padded query rows into
valid rows. The new across-row predicate is the place where masking matters.

No universal bitwise claim is made for NaN payload/signaling behavior, differing
denormal modes, or a compiler that reassociates the PV accumulation. The strict
carry-C compiler gate and unchanged native oracle remain necessary. In
particular, PV accumulated from zero and followed by a final carry addition
has a different intermediate rounding sequence and is outside this experiment.

## Relation to the actual failed variant

The earlier e66 source placed PV calls in both conditional branches. Its fresh
compile showed a real skipped-multiply path and correct logical carry-C
placement, but generated two static historical PV copies, lost the original
two-tile V async buffer, and raised producer registers96→128 while shared bytes
stayed20,480. The predeclared resource gate rejected it before candidate kernel
execution. These are actual artifact observations, not a measured performance
regression or numerical failure.

This separate source form removes one concrete structural difference: PV has
one source consumer after a conditional rescale. It tests whether that permits
the original V pipeline and resource use to survive. It does **not** prove
that duplicated PV consumers uniquely caused the earlier allocation increase,
or guarantee any compiler outcome. The predicate, branch and loop-carry phi
can still change scheduling, register pressure or pipeline formation. The
earlier failed source/artifacts must remain immutable.

## Unchanged native admission gates

Before any frequency instrumentation or timing, compile exact control3da and
candidate14a with identical installed compiler, precision, geometry and launch
options. The existing gates remain fixed:

* Producer registers must remain at or below96; no newly observed spill/local
  traffic is accepted. Quote fresh metadata and inspect PTX/SASS local-memory
  operations instead of transferring old raw spill0 as a new measurement.
* The original two-tile K **and** two-tile V async buffering/dataflow must be
  present, with no resource relaxation to excuse another pipeline change.
* A real fast executable path must bypass numerator rescale. Unconditionally
  computing `acc*alpha` and selecting old carry afterward fails this mechanism.
* Common PV must retain BF16 operands and the control matrix accumulation order.
  Fast C is old carry; fallback C is old carry×alpha. Reject zero-C PV followed
  by final add, extra executed PV work or changed matrix precision. Inspect
  TTGIR/LLIR/PTX/SASS rather than inferring this from source syntax.
* Denominator FMA/reduction, max/P/alpha and new-token tail remain unchanged.
  Separate merge SASS must be byte-identical to control. False-route source
  reconstruction is proved here; native false-route preservation is not yet
  recompiled by this review.

If any gate fails, reject this exact source form without frequency collection,
correctness/timing sweeps, threshold changes or official-check use. If all pass,
that is only compile admission: instrumented skip frequency must be kept
separate from uninstrumented times, and the predeclared original-oracle,
cache/replay/boundary tests must precede fresh-fixture paired whole-call timing.
No execution or success beyond the local source checks has occurred here.
