# Full L1 M64N8 producer: predeclared integration gate

Frozen before full producer compilation or native execution. This is one
integration hypothesis, not a flag/tile sweep or an XQA library port. Working
selected source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
official check `5090b83f1d63407b`, score 1.2336330588627955. A2-11 rollback is
preserved. No credential renewal, paid resource or formal submission is allowed.

## Evidence and causal hypothesis

Selected custom L1/G8 uses M16/D64/N64 with eight valid query rows. It already
has GQA reuse, FP8 query/cache encoding, page1 gather, FP32 online softmax,
asynchronous staging and B32-full page lookahead. None is claimed missing.
The reviewed one-tile offline probe emitted both reversed dots as native
M64N8, without query-axis padding or spill/local. Its 32 registers and 18 KiB
shared do not establish full-producer viability.

Hypothesis: placing eight queries on native N8 and keys/features on M64 can
reduce padded-query work/state in the complete producer while retaining tensor
arithmetic. Cross-warp reductions, transpose publication, FP8 widening, explicit
pipeline and tail/layout transitions can erase any gain. Geometry or logical
traffic is not a measured bottleneck. XQA already uses SWAP_AB; this mapping is
not a novelty claim. Pinned primary-source comparison precedes GPU admission.

## Exact isolated scope

Only four selected custom scored decode routes: sliding B1/B16/B32 and full
B32, L1/G8/D64/page1. Other shapes, DP/G variants, page64, FA3 and L8 routes
retain selected code. Preserve split/grid/chunk/lookahead policy, per-head
parallel merge and original M16 partial ABI with neutral unused rows.
Any S1 case also falls back unchanged; the integrated producer requires the
original separate merge. Neutral unused rows are ignored ABI padding, not a
claim that the selected producer's ignored padding already contained zeros.

Preserve scale placement, Q RMSNorm/RoPE and BF16 then E4M3 rounding; encoded
cache K/V, nonunit scales, strict FP32 QK/max/exp/denominator/online carry and
partials; BF16 P/V before PV; live page/length reads, masks, split ownership and
final norm/gate. Last split still prepares, quantizes, appends and attends to
the new key with empty history. Preserve N16 new-token tail and masks; never
silently pad it to 64 or omit it. No approximate softmax is part of this test.

Use explicit masked asynchronous K/V staging with safe ring-buffer lifetimes.
Gluon num_stages=3 alone does not inherit ordinary Triton's pipeline. No untimed
live plan/repack, output reuse, extra preparation launch or judge-input change.
The explicit two-slot staging primes two payload tiles and looks ahead to the
next refill address. It retains the selected B32-full lookahead guard and live
address policy, but not an identical instruction distance to the original
one-address lookahead. This integration couples orientation, layout conversion
and explicit buffering; any eventual complete-call effect cannot be attributed
to matrix orientation alone.

## Same-toolchain compile falsifier

Freeze source/helper hashes, compile exact selected `_fused` control and full
Gluon producer under cached Triton3.8, explicit SM90, four warps/three stages.
Extract only compiler-safe selected JIT source/dependencies. No Torch/SGlang
host import, driver binder, handle or device execution. Record extraction,
specializations, compiler/API/tool hashes and raw IR/PTX/cubin/SASS/resources.

Fixed geometries: B1 sliding S16, B16 sliding S9, B32 sliding S5 and B32 full
S11/8192 chunks/page lookahead. All HKV2/HQ16. Match runtime scalar ABI and
alignment where recoverable; unresolved specializations must remain explicit,
not an official binary identity claim.

Hard gates declared before counts:

1. Both historical dots emit native M64N8 without semantic query padding or
   MMA-sync/scalar fallback. Preserved small tail MMA is identified separately.
2. Precision, complete key reductions, FP32 online contract, shared publication
   and accumulator completion before consumption remain valid. Algebra and
   compilation cannot replace original-oracle numerical validation.
3. At most 96 physical registers/thread, zero stack/spills and zero native
   local-memory traffic in every eligible specialization. Record matched
   control resources. Do not relax this after seeing counts.
4. At most 40 KiB compiler dynamic shared launch request/CTA, safe explicit
   asynchronous K/V buffering and
   verified static SM90 capacity for at least two CTAs/SM. This is feasibility,
   not achieved residency. Missing lifetime/resource evidence does not pass.
5. Independent source/index/model and native-artifact reviews find no blocker
   in empty/window/chunk/tail/ownership/merge behavior.

Gate failure stops this branch before GPU/official quota; retain evidence,
without a configuration sweep to rescue it. CPU container is bounded to 240s,
2 CPUs/4 GiB, no network/GPU/credentials, owned payload/artifact mounts only.
Root may start the existing engine and remove this probe's own container;
never reconfigure/stop shared engine or other tracks' containers.

## Coverage and later official gate

Uniform whole-call cuts of 10%/20%/50% on four target paths, others fixed,
model overall score gains of 3.0305%/6.4383%/20.1975%. Whole-score +3% needs
roughly 9.9049% uniform target cuts. These are conditional model arithmetic,
not forecasts, measured recoverable loss or a complete winning plan.

Only after all gates pass, one authorized nonranking official check may run
the frozen substantive candidate: original 16 prerequisites, 12 timings and
unchanged tolerances. Follow terminal status, normalize targets and controls,
record evaluator/driver identity. A cross-run point score is not paired
evidence or certainty; no unused compile-only or synthetic-input check.

Predeclared useful-effect screen: target-only modeled score gain at least 3%,
no changed-path normalized regression above 3%, unchanged-control maximum
absolute normalized drift at most 3%. Failure rejects; substantial drift makes
comparison inconclusive. Keep selected 1.233633 until a separate evidence/risk
assessment supports promotion. No microprobe speed or FA3-internal claim.

Static capacity provenance: the [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
documents 65,536 registers, 64 warps, 32 blocks and 228 KiB shared per SM,
with 1 KiB reserved per block. At 128 threads, two blocks satisfying the
96-register/40-KiB dynamic limits plus at least the documented 1-KiB/block
reservation fit those architectural budgets, including ordinary register
allocation granularity. Cubin static SHARED and conservative dynamic+static
totals are recorded separately rather than relabeled as launch shared.
This conditional feasibility arithmetic is
not a measured H200 occupancy or runtime carveout observation.
