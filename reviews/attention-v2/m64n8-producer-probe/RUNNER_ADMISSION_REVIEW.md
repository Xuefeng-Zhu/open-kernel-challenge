# Bounded runner admission review: no known invalidity found

Reviewer: `/root/offline_transpose_review`, with an independent tail/ABI source
check by `/root/offline_transpose_review/fa3_mapping_source`.
This is a practical pre-run review, not formal verification or a GPU correctness
pass. No compiler, GPU, container, service, authentication or Git action was
performed. Only this new review document was written.

The user has now requested a direct runner check. Parent explicitly reassessed
the earlier self-imposed absolute zero-stack/local criterion: the common RoPE
slow-path stack is a performance/resource risk, not by itself a contract or
correctness violation. The old `ARTIFACT_REVIEW.md` STOP decision remains
unaltered at SHA256
`96c3d2703fad81a46d6cc5f764d3de26a9e25c28028a9d5c2190644b79564838`.
This review does not retrospectively change that declared-gate outcome.

**Finding:** no known source/contract or inspected native-ordering blocker was
found to one nonranking original-oracle runner check of the frozen candidate.
Actual numerical/cache correctness, replay and performance remain untested.
Admission to a diagnostic check is not production selection or proof of safety
for all nearby inputs.

## Exact reviewed identity and scope

| Artifact | SHA256 |
|---|---|
| `candidate.py` | `96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970` |
| `kernel.py` | `eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480` |
| Selected control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Existing native manifest | `2133c3ccccba2767b3513b5992b91ab342016ed52800b5c3ceb57bbf3f59fa68` |
| Existing native text archive | `edd799ac4acb474ac7166f6bd6f52640f3ad235304fa734cc141095a11e65ebd` |

Candidate319–323 limits replacement to L1/G8/HQ16/HKV2/D64/page1 with
SPLITS>1: sliding B1/B16/B32 with the original window/RoPE contract, or full
B32 without RoPE. S1 and other contracts retain selected behavior. Allocations,
grid, split/chunk policy, separate merge and FA3 routing remain unchanged.
The four scored split counts are16/9/5/11. Only candidate S11 full and S16
sliding have existing native artifacts; S9/S5 were not compiled. Their shared
source does not prove their native correctness or resources.

## History precision and reductions

Kernel107–112 retains FP32 norm/RoPE then Q BF16→E4M3→BF16, with the factor
outside the encoding and independent scalar K/V scales. Historical encoded
E4M3 K/V are widened to BF16 at163–166. Both history dots request FP32
accumulators and `max_num_imprecise_acc=0`; P rounds BF16 before PV. K scale
enters QK once; V scale remains in the unchanged final merge. This is a changed
QK implementation relative to selected FP8 tensor arithmetic, so algebra alone
does not prove numerical equivalence.

Existing D32full TTGIR308–345 retains64x8 FP32 QK, max/sum over key axis0,
FP32 online state and BF16 P/PV. PTX1034–1050 and1258–1273 are the separate
four-instruction native M64N8 QK/PV groups; D1slide has corresponding groups.
The D32full max reduction at1084–1144 and sum at1181–1241 contain intra-warp
shuffles, shared STMatrix/barrier/LDMatrix exchanges, then final FP32 reductions.
The shared exchange routes contributions from all four warps; it is not a
warp-local-only reduction or a BF16 downcast of denominator/max state.
Changed FP32 association, fused arithmetic and SFU behavior still require the
unchanged oracle and tolerances.

## Raw ring, shared publication and waits

Source137–175 issues at most two initial **real** tiles, one committed group
containing both K and V for each tile. No masked-only future group is issued.
Wait1 is used only while a next tile exists; the final tile uses wait0. Thus
the oldest tile is complete before consumption with at most one future group
pending. Single-tile, two-tile and longer histories preserve this FIFO rule.
All controlling conditions are CTA-uniform sequence/split expressions.

A CTA barrier160 precedes raw reads. Invalid final rows are explicitly zero
selected163–164. K/V widening and shared stores finish before barrier168,
then the consumed raw slot can be refilled for tile+2. The refill therefore
does not target an unread current tile. Address-only lookahead stays masked
and reads live page entries; it does not require an untimed cached plan.

D32full native PTX1297–1315 preserves the wait1/wait0 branches and CTA
barriers before raw loads; source204–205's unconditional drain survives at
2465–2467, including empty history and before tail scratch reuse. Native
LLVM shared bases are raw K0, raw V8192, BF16 K16384, V-transpose24576,
Q-transpose32768, P33792 and reduction scratch34816. These regions are
separate in the inspected full specialization; the raw slots stride4096.

QK's proxy fence and wait-to-zero precede score use (PTX1029/1058).
P store1249, CTA barrier1251 and proxy fence1253 precede PV. PV wait-to-zero
1280 precedes accumulator carry1284–1287. The next iteration's barrier
precedes converted K/V overwrite. Q-transpose remains live through its tail
reload; raw-copy drain precedes tail conversion/scratch use. No known pending
accumulator read or live-buffer overwrite was found in this bounded inspection.
This is not an exhaustive native CFG/race proof, and S9/S5 lowering is unknown.

## New-token tail, carry and partial ABI

Only the last-index split207–223 prepares, quantizes, stores and attends the
new token, even with no historical keys. New K remains FP32 norm/RoPE divided
by K scale then E4M3, without an added BF16 round; V scale placement also
matches the selected source. Only `nl==0` is active for L1.

The tail stays M16/N16: bounded gathers form padded Q/P rows then slice valid
G8 outputs at239/260. Separate BF16 MMA QK236 and PV256 have FP32 accumulators;
the existing native artifacts attribute eight MMA sites to each source line.
FP32 max/denominator/alpha carry and layout conversions243–265 return valid
query state to the64x8 history accumulator. Kernel267–278 preserves the
split-major M16 allocation stride and writes neutral ignored rows8–15;
the unchanged merge reads only valid G8 rows. No known tail mask, ownership,
scale-placement, gather-bounds or merge-consumed address violation was found.

## Remaining risks and decision

The32-byte stack/local sites are inlined RoPE range-reduction arrays also
present in selected sliding controls; ptxas reports zero register spill bytes.
Their runtime frequency/cost is unknown. Candidate4-byte cp.async lowering,
larger shared request, different association/layouts, official toolchain/API
behavior and uncompiled S9/S5 are additional risks, not correctness passes.
Cache mutation, graph replay, serving concurrency and native empty/tail/window
edge execution remain unobserved. Official prerequisites alone may exercise
fallbacks; the scored target calls must also pass their original reference.

No known invalidity was identified within this bounded review. One expressly
requested offboard check can test these unknowns without changing source or
tolerances. A compile/import/oracle failure must be reported as such, not
worked around or hidden. A pass would still require per-shape/control analysis
and separate replay/coverage assessment before promotion. Selected1.233633
and A2-11 rollback remain unchanged by this admission review.
