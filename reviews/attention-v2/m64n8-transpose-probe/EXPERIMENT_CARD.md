# One-tile L1 GQA8 transpose: prospective offline mechanism gate

Status: card frozen before any native counts or compile outcome. No GPU,
numerical execution, timing, official check or production source change is
part of this gate. Root owns the already verified existing offline CPU
container route and its artifacts; the card author performs no container,
credential, service or Git action. Selected3da and1.233633 remain unchanged.

## Single question and actual baseline

Can the installed compiler family express both historical-tile QK and PV as
strict BF16/F32-accumulation native64x8 operations, without padding the eight
valid queries to16/64 or falling back to old MMA/scalar arithmetic?

Selected control source SHA256 is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Its L1 custom producer has M16/G8, D64 and N64; eight query rows are valid.
The exact known D32-full control compilation has96 registers,20,480B shared,
mma.sync F16 QK/BF16 PV, FP32 state and zero reported spills. It already uses
GQA reuse, strict FP8 encoding, async copies, selected split/chunk geometry
and lookahead. None is a missing capability claimed by this probe.

The proposed orientation is:

    QK: K[64,64] @ Q_transposed[64,8] -> scores[64,8]
    PV: V_transposed[64,64] @ P[64,8] -> attention[64,8]

The leading64 dimension is keys or features, not padded queries. This is
different from the old slow L8 M64/N64 prototype, padding L1 to M64, and the
parked SIMT proposal. It may retain matrix throughput while avoiding padded
query arithmetic/state, but extra cross-warp reductions, shared transpose
publication and synchronization may outweigh that benefit. No runtime gain
or hardware bottleneck is inferred from dimensions or instruction presence.

## Exact first-probe scope

One full64-key historical tile, G8, D64, four warps. Inputs are Q BF16[8,64],
K/V E4M3[64,64] and FP32 scalar K/V scales. Q represents the selected
post-normalization/RoPE BF16 boundary, then undergoes the same BF16→E4M3→BF16
rounding. K/V widen from their encoded values to BF16. Use the exact selected
QK factor5.4730064863875, softmax scale0.125, scales and log2 conversion.

FP32 scores reduce max and sum over all64 key rows for each query column.
Exponentials/state are FP32; P is rounded to BF16 before PV. PV accumulates
in FP32. Expose raw FP32 attention[8,64], row maximum[8] and row sum[8]; do
not pretend those are final production BF16 norm/gate outputs.

Four unbuffered shared operands are K[64,64], Q-transpose[64,8], V-transpose
[64,64] and P[64,8], with intended layouts matching their contiguous K/M
axes. Their logical BF16 storage is18KiB before padding/layout temporaries.
There is NO K/V ring, next-tile prefetch, online two-tile carry, norm/RoPE
implementation, new-token tail, global KV append, partial merge or dispatch.
No overlap/buffering or complete-producer resource claim is admissible.

## What existing source evidence can establish

The saved CPU inventory SHA256
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`
records Triton3.8.0 and the installed Hopper Gluon module, whose6825-byte
source hash is
`c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc`.
Saved signatures expose explicit warpgroup MMA, FP32 accumulation policy,
async/wait dependency APIs and shared fences. Existing native M64/N64 BF16
artifacts prove that tested shape compiled. They do not prove N8 admission,
shared-A transpose support or this reduction/layout chain.

The existing offline cached wheel is a separate compiler provenance source.
Record its exact file hashes/version/platform, target SM90, driver-free
compilation method, compiler options and tool versions. A version match alone
does not establish identical compiler bits or official judge binary identity.
No new package, network download or runtime may be introduced to rescue a
failure. Root's existing-container permission does not cover new credentials,
GPU attachment, engine changes or stopping other tasks' containers.

## Hard gates, declared before counts

1. **Both native N8 dots:** raw IR/PTX/CUBIN/SASS must show the intended64x8
   QK and PV operations, with eight-query score/attention state and no semantic
   query padding to16/64, scalarization or mma.sync fallback. Physical shared
   alignment padding must be recorded separately; it is not automatically
   padded-query arithmetic.
2. **Precision:** preserve E4M3 query rounding and exact BF16 widening, BF16
   operands/P and FP32 accumulation/max/exp/denominator/output. Explicitly
   retain `max_num_imprecise_acc=0`; do not lower precision to obtain an opcode.
   Different reduction order is not bitwise equivalence or a numerical pass.
3. **Complete reduction and PV publication:** all64 keys must participate in
   each query's max/sum across all participating warps. Inspect shared/shuffle
   reductions, barriers, layout conversions and P stores/fences before PV.
   Reject a per-warp partial result or a shared descriptor/layout mismatch.
4. **No spills/local traffic:** use ptxas/CUBIN metadata and native local-load/
   local-store inspection where available. Virtual PTX register declarations
   cannot alone establish physical register use. Missing evidence is unknown,
   not a pass.

Record ALL physical register/shared/resource counts and raw artifacts. The
prospective working reference is128 registers/thread and32KiB shared, allowing
layout/conversion overhead over18KiB operands. **These two values are reference
budgets, not hard admission thresholds or production parity.** Do not relabel
them after seeing counts. Report mechanism and resource observations separately.
Native instruction emission does not establish executable correctness, real
occupancy, utilization, overlap or speed.

A pass justifies reviewing integration cost. A failure/unsupported shape is
retained without another tile/flag/precision sweep or inline-assembly bypass.
Root may diagnose exact compiler error locally, but no official quota follows
for a baseline-returning unused compile probe or metadata-only dedupe variant.

## Coverage if a complete producer later becomes justified

Initial potential scope is ONLY four currently custom decode scored paths:
D1/D16/D32 sliding and D32 full. Keep selected S/grid/chunk/lookahead/warp/stage
policy and merge. No full B1/B16 FA3 or L8 route change is proposed here.

| Uniform complete-call reduction on those four paths | Fixed-reference score | Relative score gain |
|---|---:|---:|
|10%|1.271018023|3.0305%|
|20%|1.313057556|6.4383%|
|50%|1.482796444|20.1975%|

Whole-score+3%/+5%/+10% requires uniform cuts9.9049%/15.9296%/29.2188%.
This is conditional model arithmetic, not a forecast or measured recoverable
cost. The zero-time group limit is mathematically unbounded because decode32
would have zero total time, not physically unbounded headroom. Matching the
historical2.9097 point by this group alone requires about98.48% reduction;
this is not a complete credible winning plan.

## Separate later integration and performance gates

Before a production candidate, first prove the original full loop, FP32
online carry, scales/masks, last-split new-token ownership and partial/merge
interfaces. The one-key L1 new-token tail cannot silently inherit a64-key
padding cost; preserving its established small-MMA numerical behavior is a
separate source/layout task. Keep original partial storage/merge first rather
than mixing another storage optimization into the mapping experiment.

Full-producer compile review must separately establish buffering/lifetimes,
no spills and physical-register parity with the relevant96-register full
control, including all integration work. The unbuffered tile's count cannot
pass that gate by proxy. Cross-warp reduction cost remains an empirical risk.

Then require unchanged original-oracle output/cache tolerances, page/scale and
empty/tail/window/chunk boundaries, live metadata graph mutation and private
scratch ownership. Only uninstrumented complete-call comparisons against the
actual selected3da on fresh fixtures can establish performance. Predeclare a
meaningful fixed-reference modeled gain of at least3%, no changed-path
regression above3% and unchanged-control drift limits; no prototype-only win
or single historical normalized point score causes promotion. Existing native
measurement access is currently unavailable, so no such run is admitted.

No GPU or official resource cleanup is needed for this card. Root's offline
process must terminate and retain bounded read-only source/artifact provenance;
any running-container cleanup is limited to this probe's own child container.
