# Open Kernel Challenge: Hopper attention architecture research

Please conduct a critical, source-backed research review of this sanitized
engineering brief. We want evidence-led architectural improvements, not blind
parameter tuning. Use public primary sources (papers, NVIDIA PTX, versioned
Triton, FlashAttention, SGLang and FlashInfer source). Browse those sources and
cite precise links. Do not execute code, access private repositories, launch
compute, submit a challenge entry, or request credentials. We will independently
implement and validate promising ideas later.

The goal is competitive, legitimate, upstreamable attention acceleration on an
NVIDIA H200. We also have a separate projection kernel at 1.173239× reference;
it is preserved, outside this immediate research scope. Kernel scores do not
establish end-to-end Serving prize performance. Challenge rules and integration
requirements should be verified if relevant, not assumed from a component score.

## Official contract and scoring

Problem: `muse_attention_decode`, public site
https://openkernelchallenge.dev/problems/muse_attention_decode and rules
https://openkernelchallenge.dev/rules . Public machine-readable problem source:
https://openkernelchallenge.dev/api/v1/problems/muse_attention_decode .

One timed `kernel_fn(data)` computes Q/K RMSNorm, interleaved RoPE for sliding
layers, FP8 KV cache writeback, paged GQA attention and final RMSNorm×sigmoid gate.
The output is BF16 with shape [B, L, Hq * 64]. Scored TP8 cases use Hq = 16,
Hkv = 2, head_dim = 64, and a GQA ratio of 8. Batch B is 1/16/32, and q_len L
is 1 for decode or 8 for causal verify. Historical contexts are around 80,000
tokens, with uneven agentic request lengths. The sliding window is 2,048 tokens
and theta is 500,000; full attention has no RoPE. Page size is 1 in scored cases and 64 in most prerequisites;
KV scale is 1 in scored cases and 1/64 in some prerequisites. Physical cache pages
are scattered. Slot 0 is valid. Actual lengths/positions/cache locations change
on GPU under a fixed CUDA graph. Full-decode FA3 scheduler metadata is supplied
by the problem; sliding and verify metadata is None. Do not cache input-derived
data, move timed work outside the call, skip low-attention keys/values, exploit
fixtures or change allowed precision.

Conceptual computation per token:
```
q = rmsnorm(q) * 5.473; k = rmsnorm(k)
q,k = interleaved_rope(q,k,position)  # only sliding
cache[cache_loc] = E4M3(k/k_scale), E4M3(v/v_scale)
a = softmax(q @ K.T / 8) @ V        # every causal key in window/full context
out = rmsnorm(a, eps=1e-5) * sigmoid(gate)
```
Queries may be rounded to E4M3 as FA3 does, no coarser. BF16 activations remain
BF16 where the baseline does; cache stays FP8. Our custom PV uses BF16 operands,
FP32 softmax and partial accumulators. Do not silently propose lower precision.
The output relative-L2 bound is clamp(2.5 × the reference's own FP8-query error
+ 1.5%, 3%, 15%), versus FP32 reference; per-row bounds and nonfinite checks also
apply. New cache entries must be within 8%; the output oracle attends to our
new entries. Passing a tolerance is not permission to game the contract.
There are 16 prerequisites and 12 scored shapes, including a long context of 1M,
window-edge distributions, mixed lengths, DP head counts and nonunit scales.

Each of six loads is C = 46 * sliding_latency + 16 * full_latency, with S calculated similarly from the
same-run SGLang reference. Score = geometric_mean over six loads of S/C. It is NOT
the mean of 12 per-shape ratios. Timings are fresh-input, cold-L2 CUDA graph
replays; candidate/reference order alternates and paired ratios are combined.
The judge subtracts measured replay overhead. Reported 95% score intervals have
a minimum of ±1.5%. Standalone profile activity sums are not calibrated wall times.

## Current measured evidence

A2-11 is the selected complete attention file: score 1.187678756×,
95% interval [1.169863575, 1.210934250]. A2-13 introduces E4M3 QK on four custom
fallback paths and passed 16 prerequisites and 12 timed cases: score 1.210990803×,
interval [1.192825941, 1.229155666]. Intervals overlap, so A2-13 is promising,
not a statistically resolved new overall best. All numbers below are historical
A2-13 official adjusted whole-call times, with its own same-run SGLang reference:

| Load/layer | Candidate us | Reference us |
|---|---:|---:|
| decode 1/sliding |10.912|20.352|
| decode 1/full |27.183|31.264|
| decode 16/sliding |14.254|26.672|
| decode 16/full |200.209|199.936|
| decode 32/sliding |20.271|30.992|
| decode 32/full |316.412|384.832|
| verify 1/sliding |17.141|22.656|
| verify 1/full |29.802|35.952|
| verify 16/sliding |22.577|29.008|
| verify 16/full |204.454|204.208|
| verify 32/sliding |27.913|34.336|
| verify 32/full |389.167|388.992|

Four custom fallback scored paths: all sliding decode plus full decode B32.
Other eight scored paths use official image FA3. Sliding verify and full B1
decode/verify have fused preparation; large full FA3 paths still use ordinary
preparation. FA3 calls use causal=True, the appropriate window = (2047, 0) or (-1, -1),
page_table, cache_seqlens, cu_seqlens_q, supplied scheduler_metadata, FP8 Q/KV,
sm_scale*qk_scale_factor and proper descales; output rounds to BF16 before norm/gate.

Instrumented mainloop activities: full FA3 decode B16 = 185.9 µs, verify B16 =
189.8 µs, and verify B32 = 373.9 µs. The verify B16/B32 mainloops account for
91.8%/95.5% of their respective activity sums.
Their calibrated whole-call times are 204.454/389.167 µs. No bandwidth, stall or
occupancy counters were captured for these FA3 kernels; no exact compiled
register/shared-memory/spill data for them. Do not substitute custom-producer
resource measurements. Sliding verify FA3 combine activities are 3.9/3.9/4.2 µs,
and scheduler preparation takes 1.7/1.9/1.9 µs, with some PDL overlap; these cannot just be
subtracted from wall time.

## Custom producer's actual dataflow

There is one CTA per (request, KV head, split). L * G query rows share cache
tiles; decode has 8 useful rows padded to BM = 16, while verify naturally has
64 useful rows. The configuration uses BN = 64, D = 64, 4 warps and 3 launch
stages. The full B16/B32 fallback uses 8,192-token historical chunks with
shape-derived split capacity; actual lengths stay on the GPU. Other fallback
paths balance their span. Each split computes online softmax over its slice.
Only the last split computes/quantizes/stores/attends to new block tokens,
so no other split reads newly written cache rows. L1 has invocation-local FP32
partials and a separate parallel merge. L8 fallback retains invocation-local
zeroed completion tickets/last-finisher merge. No device-global mutable ticket
buffer remains. Independent overlapping graph/call safety is not fully tested.

A2-13 QK expressions (pseudocode, not entire submitted source):
```
q = norm_rope(q).to(bfloat16).to(float8e4nv)
qk_scale = sm_scale * qk_scale_factor * k_scale * log2(e)
for each historical key tile:
    k = load(FP8_cache, page_table_gather)  # E4M3 operand
    scores = dot(q, k.T, out_dtype=float32, max_num_imprecise_acc=0) * qk_scale
    scores = apply_causal_window_and_length_masks(scores)
    update_online_max_and_sum_in_FP32()
    p = exp2(scores - max)
    v = load(FP8_V_cache).to(bfloat16)
    acc = alpha * acc + dot(p.to(bfloat16), v)
```
Cache new-token QK uses the same E4M3 operand policy; scale applied exactly once.
This changes rounding from A2-11's BF16(factor*Q); it is legal E4M3 conversion
but consumes more error margin. A2-13 prerequisite output relative-L2 errors were 3.935–9.903%.
No new source changes should quietly erase a BF16 boundary or alter RMS epsilon.

## Successful and failed experiments; do not repeat blindly

- FA3 routing beat the original fused Triton full-verify path. Old E2 full
  verify B16/B32 took 339.220/588.574 µs across historical jobs. Reverting routing
  unchanged is not promising; present FA3 is about 204/389 µs.
- Fused QK normalization/FP8 query/cache preparation saved launches on sliding
  verify (~28% for target B1; ~23%/19% normalized cuts for B16/B32) and full B1 (~16% verify,
  ~13% decode). It is already used on those paths.
- A2-11 parallel L1 partial merge improved sliding decode; prior serial merge
  regressed. Whole selected score 1.18768×. Invocation-local scratch preserved.
- A2-12 standalone query hoist reduced producer activity but added a launch;
  target whole-call latency worsened 8.676%. Don't optimize only a subkernel.
- A2-13 legal E4M3 QK: four target normalized cuts of 5.584%, 3.533%, 2.243%, and 12.671%.
  The aggregate change was +1.963% with overlapping intervals, not proven native FP8 acceleration.
- A2-14 deferred denominator horizontal reduction kept a live FP32 [16, 64]
  per-column mass accumulator. It passed correctness but target worsened 33.926%,
  score 1.168709×. No A2-14 native resource/counter capture; extra live state and
  operations are source facts, register-pressure causality is not established.
- An earlier loop num_stages=3-only experiment was inconclusive (~0.25% score).
  No generated-code evidence then. Do not equate that flag with actual overlap.
- Avoid generic dual-head/wide-tile/materialized-QK-PV/rebalanced-span sweeps.
  Some of these were rejected during earlier investigations; lack of a receipt
  for a particular variant is not proof it failed on the GPU. Any revisit needs a
  distinct mechanism and explicit evidence, not a renamed repeat.

## A2-15 native code and paired follow-through

A2-15 changes only full B32/L1 historical page-ID lookahead relative to A2-13.
Native artifacts were saved for both exact sources on H200, capability 9.0,
with Triton 3.8.0, Torch 2.14.1+cu130 and CUDA 13.0. The exact Triton wheel Git
revision is unknown. Both producers use 96 registers and have a driver-reported
raw spill count of 0. Shared memory increases from 12,544 to 20,480 bytes.
Merge SASS is byte-identical. The parent uses one K/V buffer and wait_group 0;
A2-15 has two K/V buffers and wait_group 2. Source-level lookahead before QK
does not survive literally: emitted page-ID loads sit between PV matrix
instructions, 111–128 static instruction-address slots before first use.
Static distance does not prove dynamic latency hiding. Both scheduling and
buffering changed; their causal performance contributions are not isolated.

Important: both saved producers use FP16 MMA for QK and BF16 MMA for PV.
Neither uses native FP8 MMA or WGMMA. TTIR has an E4M3 dot; TTGIR has MMA
version 2 plus FP16 casts; PTX has
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. Triton v3.8 source explains
that supportMMA(v3) requires M % 64 == 0 and rejects FP8/F32 with
max_num_imprecise_acc < 32. Our M = 16 and bound = 0 independently block it.
The MMA2 path promotes FP8 to FP16 on sm90. The versioned source matches the
emitted behavior; we have not traced every pass in the actual compiler binary.
Native FP8 WGMMA can accumulate below single precision despite an F32
destination. Do not silently relax this precision constraint to obtain a
favorable instruction.

Separate paired execution then passed two B32/L1/full cases: a scored-family
case with page size 1 and scale 1, and an adversarial case with page size 64,
scale 1/64, and lengths
0/1/63/64/65/8191/8192/8193. Four graphs tested same-address A → B → A replays
with changed actual lengths and content. Output, new cache entries, immutable
inputs and untouched cache rows were checked. There were 44 validated replays,
including timing calls. This did not cover all 12 shapes, all prerequisites,
concurrent calls or independent sessions.

Raw whole-kernel graph medians were 312.032 → 295.904 µs. All 12 paired
comparisons favored A2-15. The median of 6 reverse-order pair ratios was
1.054731, equivalent to a 5.189108% latency cut. Four warmup pairs were
predefined, and all samples were retained. A separate empty-graph calibration
of 3.104 µs gives a ratio of 1.055312. Raw timing is primary; neither ratio is
an official score. The predeclared 10% goal was missed.

The historical full decode B32 weighted share is 0.844461647. Transferring
only the relative cut models
`(1 - 0.844461647 * 0.05189108)^(-1/6) = 1.007496`, or just +0.750% overall.
There has been no A2-15 official check. Selected A2-11 is preserved, and all
GPU apps are stopped.

## Our current tentative plan — challenge it, don't merely agree

The highest-weight remaining stages are full verify B16/B32. A hypothetical
10–20% whole-path cut to BOTH models +2.80–5.94% overall. Including full decode
B16 would model +4.29–9.19%. These are scenarios, not predicted achievable gains.

The tentative first feasibility probe is to compile the unchanged A2-13
custom producer at L8/B16/B32, which naturally has M = 64, versus a version
that changes ONLY QK operand widening to BF16 after the existing E4M3
quantization. Keep max_num_imprecise_acc = 0, BF16 PV, FP32 softmax,
partitioning, ownership and merge unchanged. All 254 finite E4M3 encodings
round-trip exactly through BF16 and FP16 in CPU tests. This might enable BF16
WGMMA without lower precision, but instruction choice, asynchronous placement,
register/shared-memory/layout costs and performance remain unknown. It would
not provide native FP8 peak throughput. Make no runtime dispatch change until
compiler evidence exists, then require full correctness and a paired whole-call
comparison with FA3. The old custom verify path's deficit is substantial.

An alternative is transposed QK: K[64, 64] @ Q.T[64, 16] could supply M = 64
without padding decode queries. Score layout, synchronization and accumulation
policy risks remain.

Another alternative is explicit FA3 num_splits = 1 for 2,048-token sliding
verify. This removes the combine stage but reduces parallelism to roughly
2 * B query tiles versus 132 SMs. No existing override has been tested.

A lower-risk alternative is to generalize proven fused preparation to large
full FA3 paths. Assumed savings of 5–7 µs each on decode B16, verify B16 and
verify B32 model only +0.84–1.19% overall.

## Primary sources already consulted; verify and extend

- https://github.com/triton-lang/triton/blob/v3.8.0/lib/Analysis/Utility.cpp#L1128-L1155
- https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L900-L937
- https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp#L314-L334
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
- https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_api.cpp#L330-L428
- https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
- https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/paged_kv.h
- https://github.com/flashinfer-ai/flashinfer/blob/a88349f9f43df74d31d1d52ad5aa20c28824a790/include/flashinfer/attention/decode.cuh

## Requested research output

1. Identify any mistakes or unsupported assumptions in our diagnosis or plan.
2. Rank at most 3 distinct high-value mechanisms. For each, specify the exact
   code or dataflow change, affected shapes, parent/control, numerical and
   synchronization risks, primary sources, conditional whole-score value, and
   falsifiable acceptance/rejection criteria.
3. Pick ONE next experiment. Distinguish compile-only evidence, CUDA
   correctness, actual paired timing and hardware profiling. Name the exact
   minimal missing artifacts or metrics needed before broad implementation,
   including whether the BF16 WGMMA probe is unlikely to improve on FA3.
   Avoid unsupported bandwidth or occupancy claims.
4. Discuss whether native FP8 arithmetic is worth changing the accumulation
   policy, or whether exact widening plus pipeline organization has more
   plausible value.
5. Propose a minimal validation matrix covering fresh cache contents, page size 64,
   non-unit scales, actual-length changes under fixed graphs, cache ownership
   and numerical edge cases.
6. Clearly label measured facts from this brief, sourced facts, inferences and
   hypotheses. Do not fabricate benchmarks, exploit the evaluator or execute
   public code. We will review licensing before reusing any implementation and
   test independently on H200.

Please give a concise decision memo with an evidence table and source links,
followed by enough technical detail to make the chosen experiment reproducible.
