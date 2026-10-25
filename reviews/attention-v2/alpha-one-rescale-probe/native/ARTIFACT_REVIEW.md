# Actual alpha-one compile artifacts: reject this source variant

**REJECT under the predeclared admission gates.** The exact alpha-one branch
does bypass historical numerator MUL and retains PV carry structure, but
producer register allocation rises **96→128** and the control's asynchronous
double-buffered V pipeline is lost. Either is sufficient to stop this variant
before frequency, correctness or performance execution. Compilation succeeded;
it is not a measured speedup or a correctness result.

## Identity and complete evidence

Frozen request/source commit is`99185609d7924d18bb07de950c3d689a9ec9e649`.
The returned raw`result.json` reports compile complete, four units, zero kernel
launches, correctness/performance both false. All24 artifact file sizes and
SHA256 values were independently verified against the returned unit records.
`artifact-review-evidence.json` retains this review's source/options/resource
and textual inventory checks; no service or GPU action was performed by this
reviewer.

| Item | Binding |
|---|---|
| Control source | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Candidate source | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |
| Worker | `9e0377dc696518225b5091f65fe7f72ac5e96e6c2b539c399f13bf337a80a585` |
| Environment | H200 sm90; Triton3.8.0; Torch2.14.1+cu130; CUDA13.0; driver610.57.04 |
| Same producer options | L1/G8/HQ16/HKV2/D64, M16/N64, CHUNK8192, SPLITS11, separate merge, PAGE_LOOKAHEAD true, four warps/three stages |

The fresh control producer SASS is exactly identical to the archived A2-15
SASS (SHA`44dc14bbc9a1d2ab246146c7c76d76c9cb7f13b929ad4c900f18c3f3f021e8de`).
Fresh paired outputs, rather than archived resource assumptions, remain the
primary evidence below. The constexpr-false specialization was not compiled.

## Predicate and actual MUL bypass: structural pass

Candidate TTGIR lines294–302 compare the **computed** alpha with1 using
unordered-not-equal, AND the result with`row_ok`, sum the integer valid-row
violations along axis0 and test zero. A valid NaN/non-unit alpha therefore
selects fallback; padded rows do not contribute. This is not an approximate
max-difference predicate. PTX1086 uses`setp.neu.f32` against0x3f800000 and
the following shuffle/integer reduction produces the branch predicate. In this
M16/G8 specialization the lowered predicate uses the valid-row alpha fragment,
while the second padded-row fragment is excluded by the specialized row mask.

The executable PTX CFG proves a real bypass:

- `@%p25 bra $L__BB0_4` at1897 selects the slow block.
- Slow block`$L__BB0_4` at607 performs eight carry`mul.f32` at609–616, then
  eight BF16 PV MMA instructions at632–667.
- The fall-through fast block performs its eight BF16 PV MMA instructions
  at1918–1953 without those carry multiplies, then branches to the join.
- Both paths join at`$L__BB0_5` near670; they do not execute both PV copies.

SASS corroborates the actual control flow: conditional BRA0x3090 targets slow
0x3480. Fast HMMA instructions run at0x3210–0x3450 and branch at0x3470 to
join0x37e0. Slow FMULs at0x34b0–0x3520 precede slow HMMA0x35f0–0x37c0.
Whole-file MUL counts happen to remain53 in both files; that static count
alone would miss the dynamic bypass and is not used as its proof.

## PV carry, denominator and tail: retained structure, not native correctness

Candidate TTGIR338–354 contains two mutually exclusive dots with BF16 operands.
Fast C is the existing`%arg23` old numerator. Slow C is precisely its alpha
product`%acc_222`. The PTX chains use old-accumulator tuples735–742 as both
the initial carry and destination; the four successive K16 MMA steps per
output tuple retain the expected logical PV accumulation chain. There is no
PV-from-zero followed by a final scalar add. Static BF16 MMA count grows10→18
because the historical eight-instruction PV sequence is duplicated in two
exclusive blocks, while the two-instruction new-KV tail remains one copy.
This is static duplication, not evidence of twice-executed PV.

Max/safe-max, alpha/P exponentials and denominator multiply/reduction/add retain
their mathematical dataflow. Candidate TTGIR287–293 contains the unchanged
denominator sequence; PTX672–673 keeps its two`fma.rn.f32` operations after
reconvergence. SASS0x37f0/0x3810 are the corresponding denominator FFMAs.
Both producers contain12 static FP32 FMAs and16`ex2.approx.ftz.f32` operations.
These counts support the localized review but are not a general equivalence
proof.

The last-split tail remains a separate split==10 branch with normalization,
E4M3 new-K/V stores, causal mask, original rescale and BF16 PV. Candidate
TTGIR360 onward retains this source dataflow; tail PTX2409–2410 retains the
denominator FMA and the tail numerator MUL/MMAs follow normally. Source tests
also prove the entire tail is byte-identical. Partial ownership and store
indexing remain the original separate-merge route.

Multiplication lacks an added FTZ modifier on the inspected slow carry MULs;
the existing exponentials keep their FTZ approximation form. Finite identity
algebra, IR structure and disassembly do not certify all signed-zero, subnormal
or NaN payload behavior. No native oracle, cache/replay comparison or numerical
execution was performed, and this review does not assert universal bitwise
equivalence.

## V buffering and register admission: two hard failures

| Same-job producer resource | Control | Candidate | Gate |
|---|---:|---:|---|
| Registers/thread | 96 | 128 | **FAIL: increased** |
| Shared bytes | 20,480 | 20,480 | Same total does not preserve buffering |
| Driver raw spill field | 0 | 0 | No new reported spills |
| PTX local accesses / SASS LDL or STL | 0 /0 | 0 /0 | No textual local-memory accesses |
| TTGIR async global→shared copies/commits | 6 /6 | 3 /3 | V async buffering removed |
| PTX barriers / butterfly shuffles | 18 /34 | 22 /37 | Added synchronization/reduction structure |

Fresh control TTGIR has both`2x64x64xf8E4M3FN` K and V shared allocations,
four loop-carried K/V async tokens and`async_wait` with num2. The candidate
has only the two-slot K allocation at207 and a K-only wait num1 at263.
Candidate332–337 instead loads the current V payload with synchronous`tt.load`,
materializes an FP8 dot-layout form and a BF16 blocked/shared form before the
conditional, and uses different V materializations in the two PV branches.
There is no two-slot V async memdesc or loop-carried V copy token. Thus
lookahead page addressing survives, but the established K/**V** payload
pipeline does not. Equal total shared bytes cannot be presented as preserved
buffering or an occupancy benefit.

The loop-carried state also changes materially. Control carries two64-element
address vectors plus K/V tokens; candidate252 and355 carry two64×64 i64 KV
address tensors and corresponding masks, with only K copy tokens. Together
with the two V layouts and exclusive PV blocks, this is a concrete increase
in live compiler state associated with the new branch. SASS names extend to
R125 instead of control R93, consistent with the recorded rounded allocations
of128 versus96. These observations explain what changed in lowering; they do
not provide a per-variable register-allocation map or prove a unique causal
allocation mechanism.

The32-register increase is33.3% per thread. This is a static resource regression
under the explicit admission policy, not a measurement of achieved occupancy,
stall reasons, HBM bandwidth or runtime slowdown. No counter or timing run is
needed to reject this first variant under the gate fixed before compilation.

## Merge identity and decision

Both merge units use the same options and27 registers/1,024 shared bytes/raw
spill0. Their SASS bytes are exactly equal, SHA256
`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`.
Source/debug records make some other formats differ; this does not weaken
the exact native merge-identity result.

The generated branch performs its intended narrow MUL bypass, but fails the
predeclared register and V-buffering gates. **Do not proceed to instrumentation,
correctness/performance execution, official check or selection promotion for
this source variant.** Keep its negative result separate from any later
lowering experiment; do not amend its source or relax the failed gates.

The saved independent reconciliation reports app`ap-uY5K1rBNFRR2K8rdo2F8tw`
stopped/tasks0 at2026-10-25T22:06:59.192653UTC, after the worker completed in
19.3083s. The parent already released the serialized slot. This report reads
those receipts and does not make a new live service-state or cost observation.
