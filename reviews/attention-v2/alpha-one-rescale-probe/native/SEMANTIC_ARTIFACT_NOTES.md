# Fresh compile: exact-one branch exists, but pipeline/resource gate fails

Independent offline review of the returned compile-only artifacts confirms
that the candidate has a genuine numerator-rescale bypass with the intended
valid-row predicate and carry-C structure. It also confirms a substantially
different V pipeline and producer register growth from96 to128. The predeclared
resource gate has already failed. These notes do not admit frequency sampling,
correctness execution, timing, an official check or a changed working selection.

The returned result states zero kernel launches, `correctness_tested=false`
and `performance_tested=false`. All24 recorded compiler/disassembly artifacts
were independently verified for exact byte count and SHA256. Current source
bytes match the result's bindings:

* control: `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
* candidate: `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764`.

All line numbers below are for the fresh files in `native/artifacts/`. Registers
are compiler allocation identifiers, not source variables or hardware counters.

## Predicate and real bypass

Candidate TTGIR294–302 computes unordered-not-equal between computed FP32 alpha
and1, ANDs this with `row_ok`, extends the flags, sums across query rows and
tests that sum against0. Thus padded query rows do not control the predicate,
and a NaN alpha on a valid row forces the original path. This is exact equality
of the existing computed alpha, not a substituted max-equality or threshold.

The conditional is explicit in TTGIR338–354. Its fast PV dot at343 receives
old `%arg23` as C. Its fallback at348 first forms `%arg23*alpha`, then supplies
that result as C to the PV dot at352. Both retain BF16 P/V operands and FP32
matrix output. Neither branch computes PV from zero and then adds old carry.

The PTX contains distinct, mutually exclusive historical PV blocks:

| Path | Fresh candidate PTX | Observation |
|---|---|---|
| Predicate |1086–1110 | Float unordered-NE comparison, three shuffle reductions, scalar branch flag |
| Fallback |609–616;632–667 | Eight carry multiplications, then eight BF16 PV MMAs with that carry as C |
| Fast |1918–1953 | Eight BF16 PV MMAs with old carry as C and no intervening carry rescale |
| Selection/join |1897;1957;670–673 | Branch to fallback, otherwise fast block, common denominator FMA after join |

Machine code corroborates actual bypass rather than merely source-level
conditional arithmetic:

* SASS`0x23a0` compares the valid alpha lane value to1; the reduction includes
  shuffles at`0x2d40/0x2de0/0x2e70`.
* At`0x3090`, `@P2 BRA 0x3480` selects fallback. Fast execution proceeds through
  the eight BF16 HMMAs at`0x3210–0x3450`, then branches to`0x37e0`.
* Fallback executes the eight `FMUL`s at`0x34b0–0x3520`, then its eight BF16
  HMMAs at`0x35f0–0x37c0`, and joins at`0x37e0`.
* C-register dependencies are carried through each matrix chain. For example,
  fallback rescales R36–R39 and R24–R27 before its initial C blocks R36/R24;
  fast begins with those old C blocks without the rescale. Subsequent renamed
  output blocks continue those same chains.

There are two static historical PV copies, not evidence that both execute in
one iteration. Static PTX MMA count rises22→30 because one eight-instruction
historical PV block is duplicated. The CFG selects one such block per loop
iteration. This preserves the intended logical carry placement; it is not a
native numerical-equivalence test, and complete fragment-layout/rounding
equivalence should not be promoted to a universal bitwise claim from this
inspection alone.

## Denominator, QK and tail

Candidate TTGIR287–293 retains old_l×alpha plus the unchanged row reduction of
P, without moving that denominator rescale inside the new condition. PTX672–673
has `fma.rn.f32` at the common join; SASS`0x37f0/0x3810` has the corresponding
FFMAs. Scheduling relative to PV changes, but no new denominator approximation,
deferred rescale or changed row reduction is observed.

The historical QK remains an FP16 MMA over values already encoded as E4M3;
PV remains BF16 MMA with FP32 carry. Neither producer contains WGMMA. Static
counts of QK/PV types and the unchanged source settings do not justify a claim
of native FP8 QK instructions or lowered precision.

The source new-token-tail AST is identical in both exact bound files, SHA256
`a9154c3b7f8f04a9c4dab9fa3c53a21d80f8a6fe39eff8303bc701bb2c9eb9b7`.
Candidate TTGIR360–470 keeps its separate last-split path. Candidate PTX2409–2419
retains tail denominator FMA and eight old-carry multiplies; tail BF16 PV at
2506/2511 still receives that rescaled C. The new exact-one branch does not
skip tail rescaling or change new-KV ownership. Tail machine code need not be
byte-identical inside differently allocated producer functions.

The separate merge SASS is byte-identical across fresh control/candidate:
SHA256`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`.
Both merges report27 registers,1,024 shared bytes and raw driver spill field0.

## The V pipeline is materially different

Equal total shared bytes do not mean the original buffering survived:

| Fresh control | Fresh candidate |
|---|---|
| TTGIR223–224: separate2×K and2×V shared buffers | TTGIR207:2×K shared buffer; no2×V shared allocation |
| Loop async wait319 permits two outstanding groups over K/V | Loop async wait262 permits one outstanding K group |
| V is loaded from its buffered tile then converted to BF16 at351–353 | V is read by a direct `tt.load` at332, then materialized in FP8-dot and BF16-blocked layouts at333–337 before the condition |
| One common historical PV dot | Distinct fast/fallback PV operand-layout paths |

Candidate TTGIR308 loads future page IDs before PV; its SASS page loads at
`0x1ff0/0x20c0` feed clamps at`0x2170` and nearby address work. In the control,
page loads are interleaved with PV and separated from first dependent use over
the historical`0x2db0–0x3660` range. These static schedule changes are real;
they do not measure elapsed latency or demonstrate which pipeline is faster.

There is concrete changed liveness to report without pretending to know the
allocator's precise causal decomposition. Candidate V layout/conversion SSA
values are formed before the conditional and feed different branch operands.
SASS uses additional high-numbered temporaries: for example R110 is defined by
FP8 unpack at`0x23c0`, remains live through BF16 uses at`0x25f0/0x26a0`, then
is overwritten at`0x2700`. R111's unpack at`0x2520` feeds uses through`0x27c0`
before replacement. Multiple representations and altered CFG/scheduling are
plausible contributors to pressure. No allocation/liveness map was returned,
so these observations do **not** prove that any one feature accounts for the
exact32-register increase.

## Resource gate and limits

| Fresh unit | Registers | Shared bytes | Raw driver spill field |
|---|---:|---:|---:|
| Control producer |96|20,480|0|
| Candidate producer |128|20,480|0|
| Both merges |27|1,024|0|

Static local-memory checks find no PTX `.local` allocation or local load/store,
and no SASS `LDL/STL`, in either producer or merge. This is compatible with the
raw spill field0, not a measurement of dynamic spill traffic. Static producer
barrier inventory rises18→22 and shuffle count38→41; static branches rise4→7.
The duplicated conditional bodies make raw counts unsuitable as execution
frequency or causal bottleneck measurements.

The candidate therefore demonstrates the intended skipped-multiply mechanism,
but fails the predeclared register/resource admission and does not preserve
the control's V buffering. Stop this exact variant here. No alpha frequency,
correctness result, paired timing, whole-score gain or statistical superiority
has been measured. These notes do not authorize a different source form or
another compile/launch.

## Fresh artifact hashes

| Artifact | SHA256 |
|---|---|
| Control producer TTGIR | `cc45ea7c5fa8e37e194e2f5e50404a12ccd981c6a866ebbd03d411a2bb2d9c22` |
| Control producer PTX | `49eef4c01fce2a41284eb9680b89f572c3ae5690452ca939e4afa22649a14f90` |
| Control producer SASS | `44dc14bbc9a1d2ab246146c7c76d76c9cb7f13b929ad4c900f18c3f3f021e8de` |
| Candidate producer TTGIR | `72913c415c9c6b5b1c7bd158d83610bf038a75a69150dfd1f74ce52bbac4739b` |
| Candidate producer PTX | `08fa4d2269004a371e8c2dd1b3bf6014b22b5ec15189e2018a628eb24a5ebc56` |
| Candidate producer SASS | `a7ead74466088a62873705d05a96b9be805681a5daeb9a11f63a91e9a51b6b03` |
