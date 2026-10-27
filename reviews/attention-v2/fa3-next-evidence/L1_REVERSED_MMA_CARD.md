# L1 GQA8 matrix orientation: conditional compile falsifier

Status: proposed dataflow and local evidence review only. No source prototype,
compiler invocation, GPU, official check, credential action or service is
performed by this card. It is not an implementation-ready performance entry.
The specifically admitted first OFFLINE probe is now the smaller, unbuffered
one-tile gate in `../m64n8-transpose-probe/EXPERIMENT_CARD.md`. That card fixes
the first-probe gates before counts: native shape/precision, no spills/local
traffic and complete resource recording. The two-tile/full-producer resource
screen discussed below is a later integration proposal, not its admission
criterion. No count or outcome had been observed when this scope was separated.
Keep selected source SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
and1.233633 working selection unchanged.

## Actual baseline gap and single mechanism

The selected custom L1 producer has GQA8, M16, D64 and N64: eight query rows
are valid. Its known D32-full specialization uses FP8-encoded Q/K lowered
through F16 mma.sync, BF16 PV, FP32 softmax/partials,96 registers and20,480B
shared. It already has async copies, lookahead and FP32 accumulation. Padding
is a source/state fact, not proof of half the whole-call time being wasted.

Test whether changing BOTH matrix orientations can retain tensor arithmetic
while using only the eight valid query columns:

    scores[64,8] = K[64,64] @ transpose(Q)[64,8]
    acc[64,8]   += transpose(V)[64,64] @ P[64,8]

Reduce max/sum over the64-key axis; maintain one online-softmax state per
query column. The contemplated native target is BF16/F32-accumulation
WGMMA M64/N8, **only if the installed compiler/API actually supports it**.
No native N8 support is proven by the saved receipts. This differs from:

- parked G8 SIMT: it would retain matrix hardware instead of scalarizing dots;
- old L8 M64/N64 Gluon prototype: L1 has only eight queries, different state
  sizes, and must retain selected external split/grid/merge geometry;
- padding L1 query rows to WGMMA M64: the M64 axis here is keys or features,
  not64 padded queries;
- changing tile/split/stage counts, packing flags or accumulation precision.

The first question is a lowering/layout question, not a claim that WGMMA is
automatically faster. Extra cross-warp reductions and transpose publication
can erase the arithmetic/state saving.

## Coverage and required whole-call effect

Initial prospective scope is the four CURRENT CUSTOM scored decode routes:
D1/D16/D32 sliding and D32 full. Preserve existing dispatch; do not replace
full B1/B16 FA3 merely to increase reported coverage. L8 paths are unchanged.
Use the selected splits, window,8192 full chunks, page lookahead, four warps,
stage policy, partial layout, merge and last-split new-KV ownership.

| Uniform complete-call cut on four paths | Fixed-reference score | Score increase |
|---|---:|---:|
|10%|1.271018023|3.0305%|
|20%|1.313057556|6.4383%|
|50%|1.482796444|20.1975%|

Whole-score+3%/+5%/+10% requires uniform target cuts9.9049%/15.9296%/29.2188%.
These are conditional arithmetic, not predicted speedups. There is no measured
recoverable fraction. The group's zero-time limit is mathematically unbounded
because both decode32 paths would become zero; it is not a physical ceiling.
Matching the historical2.9097 point by these paths alone requires about98.48%
cut in this model, which is not a credible winning plan. Broader deployment
would require independent evidence against the actual FA3 routes.

## What the saved installed API evidence proves

The CPU inventory `image-inventory/result.json`, SHA256
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`,
records Triton3.8.0 and the installed Gluon Hopper module. Its6825-byte Hopper
source hash is
`c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc`.
Saved signatures expose `warpgroup_mma`, explicit async mode, accumulation
precision argument, `warpgroup_mma_wait` and shared-memory fences.

The existing `../gluon-vstage-probe/kernel.py`, SHA256
`6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`,
and its raw compile result SHA256
`258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240`
prove that installed explicit BF16/F32-accumulation M64/N64 operations,
shared layouts, fences and dependency waits compiled. Its later correctness
pass establishes that tested implementation, not this new layout.

Those receipts do NOT establish N8 admission, physical shared-A transpose
support, a correct64-key cross-warp softmax reduction, low-cost layout
conversion, or usable local offline compilation. The saved inventory contains
selected excerpts/signatures rather than the full semantic validator. No
matching compiler revision beyond distribution/version is known. Parent's
offline-tool investigation must establish an existing supported CPU route;
no installation, remote resource, credential or alternate environment is
authorized by this card. If no such route is available, report that blocker.

## Cheapest falsifier before a full producer

Prepare only a source/shape model and, if an already available offline SM90
compiler exists, a two-historical-tile primitive containing BOTH dots, FP32
max/sum/online carry, BF16 P publication and the real shared layouts. It must
not be an isolated empty MMA whose missing reductions hide the cost.

1. **Native N8:** generated IR/PTX/CUBIN must actually carry64x8 score and
   output tiles. Reject if the API rejects N8, secretly pads the query axis
   back to16/64, scalarizes the hot dots or emits only the old MMA family.
   No flag or tile sweep, precision relaxation or inline-assembly workaround.
2. **Reduction:** key-axis max and denominator must combine all64 keys for
   every query, including across all participating warps. Inspect shuffle,
   shared reduction and barriers; a per-warp answer is wrong. Record added
   synchronization; fewer elements do not imply a cheaper reduction.
3. **PV shared layout:** Q transpose as the QK B operand and P[64,8] as the
   PV B operand need valid native descriptors. V transpose must be a supported
   shared-A representation, not an unaccounted global transpose or full KV
   repack. Prove read/write ordering and that shared buffers are not recycled
   while WGMMA still consumes them. P must be BF16 before publication.
4. **Precision:** preserve Q BF16→E4M3 rounding and exact subsequent widening;
   K/V use the same encoded values and scales. Use strict FP32 accumulation
   and `max_num_imprecise_acc=0`, FP32 max/exp/denominator/carry and original
   BF16 P/V boundaries. Product representation equivalence is not reduction-
   order or bitwise equivalence; later original-oracle validation is mandatory.
5. **Declared engineering resource screen:** at most96 physical registers/
   thread, zero spills/local traffic, at most40KiB shared and at least two
   CUDA-legal CTAs/SM at128 threads. These are prospective rejection gates,
   not achieved occupancy. Four BF16 K/V ring tiles cost32KiB; Q and P require
   about2KiB more before layout padding/barriers. Virtual PTX registers alone
   cannot pass the physical-register gate. The later full producer must pass
   too; primitive resources do not establish complete-call feasibility.

Maintain the original partial stride/layout and merge first; do not mix a
compact-partial-store optimization into this test. The L1 new-token tail has
only one valid key and cannot silently inherit a64-key padding cost. Its
existing numerical/ownership behavior needs a separate explicit mapping,
possibly retaining the established small MMA sequence. Any tail conversion
and interface cost remains in the full call; no new global synchronization
or extra launch is assumed free.

If this compile gate fails or cannot be executed, stop before performance
implementation/official quota. Do not upload a baseline-returning unused
compile probe or metadata-only source variant to evade duplicate detection.

## Required next gates if lowering survives

Only then design a complete source-bound producer at unchanged selected
geometry. Reconstruct original key/window coverage, scales, Q/RoPE rounding,
per-tile online softmax, neutral empty splits, FP32 partials, last-split live
cache writes and final norm/gate. Validate page1/page64 fallback contracts,
nonunit scales, empty/tail/window/chunk edges, original caches/outputs,
same-address metadata replay and invocation-local ownership. All original
tolerances remain unchanged; no key skipping or cached live plan.

Require an uninstrumented, cold-L2, reversed-order whole-call comparison with
the ACTUAL selected3da control on fresh fixtures, including at least the
large full target and a sliding target. Native instruction presence or a gain
over the old slow prototype is insufficient. Prospective useful-effect gate:
fixed-reference modeled whole-score gain at least3%, no changed complete-call
regression above3% and no material unchanged-control drift; confirm fresh
fixtures before promotion. If same-session selected comparison is unavailable,
report that evidence gap rather than claim a performance winner from compile
output or a single cross-run official point score. Original full official
correctness is necessary before adoption, but no check is admitted here.

No current runtime or predicted benefit is reported. This is one bounded
lowering/dataflow uncertainty to resolve before a new architecture branch.
