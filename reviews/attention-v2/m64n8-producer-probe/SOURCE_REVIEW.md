# Independent producer contract/model review: source PASS

No source-level scope, tail, indexing, ownership or valid-partial blocker was
found for the bounded **CPU compile-only gate**. This does not approve GPU
execution or an official check. Instruction geometry, physical resources,
generated memory ordering, native precision/oracle/replay and performance
remain untested by this review.

| Reviewed source | SHA256 |
|---|---|
| Selected baseline | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| `kernel.py` | `eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480` |
| `build_candidate.py` | `de1da53203a166d43d04c425ce81c4af855ca6d20453a406507b26d46bc5b93f` |
| Assembled `candidate.py` | `96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970` |
| `test_contract.py` | `e872828b1604a393062bebf4b4daea28b20a50f9cbab9ed2a45462740203a236` |

Sixteen independent pure-stdlib tests passed. They compare address/key-set
contracts and exactly representable mathematical fixtures, not hardware
FP32 rounding, a CUDA cast emulator, original-oracle GPU results or speed.
No Torch/Triton/candidate import, service, container, GPU, credentials or Git
operation was performed by the reviewer. Only the contract tests and this
review were written.

## Scope and preservation

The actual builder/kernel assembly round-trips to byte-exact selected3da after
removing its owned append and restoring the one launch replacement. The
builder refuses changed baseline bytes and global import/assignment/function/
class name collisions; original definitions, constants, public FA3 helpers,
merge and all allocations/grid/split policy remain intact.

The new selector is restricted to L1/G8/HQ16/HKV2/D64/page1, SPLITS>1,
sliding B1/B16/B32 with the existing RoPE/window contract, or full B32 without
RoPE. Everything else uses the selected producer or unchanged outer FA3
dispatch. In particular, nearby B32-full short-capacity S1 calls retain the
old direct-output/norm/gate path. No new S1 implementation is implied.
The four scored grids remain32/288/320/704 producer CTAs with S16/9/5/11.

The16 official prerequisites may exercise only fallbacks for this narrow
scope. The TASK explicitly says every timed call is also checked against the
reference, so later official scored calls would exercise their own original
oracle. That is still not targeted native empty-history, scale-boundary,
page-mutation or serving-concurrency coverage.

## Mathematical contract and quantization

The owned norm/RoPE helper has identical selected expressions after language
alias/name/docstring normalization. It remains FP32 through RMS and RoPE.
Only Q then rounds BF16→E4M3→BF16; no factor enters that cast. New K remains
FP32 norm/RoPE divided by ks then directly cache-E4M3, without an added BF16
round. New V remains BF16 input→FP32 division by vs→E4M3. History K/V widen
encoded E4M3 exactly to BF16. QK scale retains
`(sm_scale*qk_factor)*ks*LOG2E`; vs is applied by the unchanged merge.

History computes K×Q.T, max/exp2/sum across key axis0, BF16 P publication and
V.T×P, with FP32 online maxima, denominators and accumulator carry. Both
WGMMA calls explicitly request `max_num_imprecise_acc=0`. Mathematical
transpose/online/merge tests use D64/G8 dyadic inputs whose probabilities and
intermediate state are exactly representable. They do not prove hardware
reduction association, SFU/FMA accuracy or native bitwise identity.

The last-index split alone normalizes, quantizes, stores and attends the new
token, including empty history and an empty final historical chunk. The
new-token axis stays **N16**: separate BF16 MMA2 QK/PV, one valid L1 token,
original causal mask and no key-axis expansion to64. Only the small tail's
query side pads G8 to M16 with zeros. Its selected semantic BF16/FP32 precision
must still be checked in generated IR/PTX/SASS. Tail output is sliced back to
G8 and merged into the transposed FP32 state with the existing rescale order.

Window pruning and split/chunk ranges match the selected formulas, including
seq0/1/63/64/65,2047/2048/2049 and8191/8192/8193 boundaries. Historical key
sets are complete and nonoverlapping; lookahead masks cannot read new-token
positions. Page IDs stay live, masked and converted to int64 before physical
cache offsets; slot0 is not skipped. CPU A/B mapping mutation is an address
model, not observed fixed-address native replay.

## Buffering, tail and partial ABI

The source has two raw FP8 K/V slots with one committed group per real tile.
It preloads at most two tiles; wait1 is used while a next tile exists and
wait0 for the final tile. A CTA barrier follows completion before raw reads;
another follows copying current raw data into separate BF16 K/V scratch before
that raw slot is refilled for tile+2. The QK result is waited before softmax;
PV requests synchronous completion, with the next iteration's CTA barrier
preceding scratch reuse. A final wait0/barrier drains the raw pipeline before
the separate tail. An abstract FIFO model covers zero/even/odd tile counts
without unread-slot overwrite; it is not native memory-order evidence.

PAGE_LOOKAHEAD retains its selected full-B32 route guard. Its instruction
distance adapts to the explicit two-slot pipeline (pg0/pg1 preload and later
lookahead), so this is not a claim of bit-identical selected A2-15 scheduling
or orientation-only causal attribution. Native buffering/overlap must be
reviewed, not inferred from `num_stages=3`.

Valid partial addresses preserve
`((b*HKV+kvh)*SPLITS+split)*M16+g`, with g0..7 and D64. The unchanged merge
uses real SPLITS as its allocation stride and masks padded BS lanes. CPU
poison tests confirm ignored M16 rows8..15 and extra BS lanes are not consumed.
The original producer's invalid query rows can contain nonneutral historical
partials despite zero Q. Candidate neutral ignored rows are legal; they are
not falsely claimed equal to original padding. All merge-consumed rows and
the complete M16 allocation ABI must still be preserved natively.

## Remaining compile and admission gates

The final raw FP8 shared swizzle uses vec8/max_phase8 for64 one-byte columns.
That precompile layout correction is not posthoc resource tuning. The logical
buffer ledger is about34KiB; compiler physical shared usage, conversion
scratch and real lifetimes remain unknown. CPU compilation must establish
both history dots as native M64N8, identify the separate tail MMA, verify all
key reductions/publication/waits and meet the predeclared96-register,
zero-stack/spill/local, shared≤40KiB and static≥2CTA/SM gates. No count from
the earlier32-register single tile waives a full-producer gate.

Any generated API/layout, precision, resource, mask or lifetime failure stops
this branch before GPU/official quota without threshold/precision/flag rescue.
Source PASS and compile success cannot substitute for the original numerical
oracle, actual-best complete-call comparison, replay/ownership validation or
a separate production-selection decision. Selected1.233633/A2-11 are unchanged.
