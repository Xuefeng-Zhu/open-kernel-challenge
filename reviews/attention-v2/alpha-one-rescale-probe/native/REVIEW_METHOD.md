# Offline native-artifact review method

This document prepares the review of the forthcoming exact-source compile
probe. It does not claim fresh compilation, correctness, execution frequency
or performance. The archived A2-15 files below are historical references only;
fresh control and candidate outputs from the same job are the primary pair.

## Identity and completeness before interpretation

Validate both exact source hashes (control3da22a…0a7cd8, candidatee66f50…94f764)
against the frozen request and returned worker report. Check H200/sm90, installed
Triton/Torch/CUDA versions, exact producer/merge options, source runtime scalars,
pointer dtype/alignment and driver metadata. This job uses L1/G8/HQ16/HKV2/D64,
M16/N64, CHUNK8192/SPLITS11, PAGE_LOOKAHEAD true, four warps/three stages,
strict E4M3 QK encoding and BF16 PV. It does not exercise another dispatch path.

Verify exactly four units and24 artifact files, each size/hash matching both
the returned record and validated transport manifest. Check reported zero
kernel launches and no correctness/performance testing. Missing/partial files
or tool failures are blockers, never a mechanism pass.

## Map the historical loop before counting instructions

In the historical control TTGIR, the `scf.for` around the historical KV loop
starts near line308. Its carried state includes row maximum, row denominator,
the16×64 numerator and double-buffered K/V tokens/address state. The archived
mapping is:

| Dataflow | Archived A2-15 location / evidence |
|---|---|
| Historical max and safe max | TTGIR lines334–343 |
| Computed alpha and P | TTGIR lines343–349 |
| Denominator product/reduction/add | TTGIR lines349–355; PTX1151–1152 FMA |
| Old numerator product | TTGIR`%acc_245 = arith.mulf %arg23, %acc_244` near356 |
| PV accumulator input | TTGIR`tt.dot …, %acc_245` near360 |
| Numerator MUL in PTX | Eight`mul.f32` at1512–1519, registers294–305 |
| PV MMA chain | PTX1539 onward, same register tuples as both destination and C |
| K/V lookahead buffering | Two2×64×64 E4M3 shared memdesc objects, async tokens and loop-carried buffer indices |

Use operation/dataflow and source locations rather than expecting the fresh
compiler to retain historical SSA names, register numbers or exact line numbers.
Historical archived producer resources are96 registers,20,480 shared bytes,
driver spill field0. Merge resources are27 registers,1,024 shared bytes,
driver spill field0. Their source/signature/compiler options are evidence for
comparison, not a guarantee that fresh outputs are byte-identical.

## Predicate, branch and PV carry proof

Start in candidate TTIR/TTGIR: identify the **computed** alpha comparison with
1, query-lane validity (`row_ok`/index<8 in this specialization), integer
combination, axis0 reduction and scalar CTA branch. Prove valid non-unit or
NaN alpha cannot take the fast edge; padded rows cannot veto or admit it.
An inference from rowmax equality is not equivalent to this specified test.

Trace each branch to the PV dot. The fast edge must pass the old numerator
unchanged as C; the slow edge must pass exactly its alpha product. Both need
the same P BF16 conversion, V conversion and matrix accumulation sequence.
Mutually exclusive static code duplication is not itself double execution;
prove the CFG/reconvergence and actual executable MMA chains. A PV dot from
zero followed by a scalar add violates the strict carry-order gate.

In PTX, identify the conditional edge or predicate and show that an executable
fast path bypasses all historical numerator MULs. In SASS, corroborate that
the corresponding carry registers reach HMMA without those FMULs; distinguish
denominator rescale, qk scaling and tail products from historical numerator
products. Do not use whole-file MUL counts alone as proof of a fast path.

## Unchanged math, buffering and ownership

Compare max/reduction, safe-max, exponentials, denominator FMA/reduction,
QK/PV conversions and MMA opcodes/sequence, per branch where necessary.
Retain signed-zero/FTZ-related modifiers and distinguish FP32 scalar behavior
from tensor-core accumulation. Source identity and finite arithmetic algebra
are insufficient to certify universal bitwise equality.

Trace the last-split new-KV tail separately: normalization/quantization,
stores, causal mask, softmax rescale and tail PV remain the original algorithm.
Check partial stores and separate merge invocation/ownership without assuming
that register renaming or moved instructions imply an algorithm change.

Inspect loop-carried K/V buffers, depth, async global→shared operations,
commit/wait placement and dependencies, stage/buffer indices and source
address lookahead. A decrease in shared bytes can indicate lost buffering;
it is not automatically a resource improvement. Whole-file opcode inventories
help locate differences but cannot replace dataflow review.

## Resource falsifiers and unchanged merge

Against the **fresh same-job control**, reject any increased producer register
count, new driver spill field/local memory instructions or loss of established
buffering. Check both PTX`ld.local`/`st.local` and SASS LDL/STL; do not interpret
`ld.shared`/LDS as spills. Record shared memory, barriers/shuffles, code/cubin
sizes, async patterns and any changed static resource bound. These are static
observations, not achieved occupancy, actual HBM traffic or stall measurements.

For merge, require exact same-job SASS bytes/hash, plus matching options and
resource metadata. TTIR/PTX debug/source paths can differ while native merge
SASS stays identical. A semantic claim about merge is not a replacement for
the predeclared exact native-identity gate.

Report each admission gate independently with concrete artifact locations and
hashes. If any predeclared gate fails, reject this source variant without a
frequency/timing run or threshold sweep. If every gate passes, only mechanism
feasibility is admitted; original-oracle/replay/cache checks and uninstrumented
timing remain unperformed. Artifact review must never silently promote source.
