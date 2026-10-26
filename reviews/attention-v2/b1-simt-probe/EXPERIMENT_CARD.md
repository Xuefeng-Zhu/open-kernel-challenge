# Implemented B1 G8 SIMT source falsifier

**Primary hypothesis:** eliminating the padded M16 attention computation and
unused partial writes with a bounded eight-head SIMT producer may outweigh
the extra reductions/layout work and loss of tensor-core throughput. GQA reuse,
window pruning and preparation fusion already exist. No G2 subgroup, split
sweep, counter rescue or additional preparation launch is included.

Control is the unchanged working source
`../fusion-lookahead-probe/candidate.py`, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The single-file candidate SHA256 is
`f5161be2e91b05fceb01e28564f96eecff471244704d036fa4cc7950dbe0ef34`.
It retains all14 baseline function ASTs, normalizing only the old dispatch
name to `_baseline_kernel_fn`; all imports/license/attribution are retained.
No candidate/Triton/Torch import, native compile or GPU run occurred locally.

## Exact change and route

The pure host guard selects only B1,L1,HQ16,HKV2,D64,window2048,rope500000,
page1 or64. All other calls use the unchanged baseline dispatch. The new
`_fused_b1_simt` uses query BLOCK_M8 and PARTIAL_M16, with the same split
formula giving S16, producer grid(1,2,16), BN64, new-token BLOCK_L16, four
warps/three stages, and unchanged `_merge` grid(1,16)/BLOCK_M16/BS16.
Historical width is128 for a full window; short histories retain the original
live-span partition formula. Query norm/RoPE and last-split cache append stay
inside the producer; output norm/gate stay in the original merge.

Two `@triton.jit` helpers replace only attention dot operations:

* `_simt_qk`: select one head with same-rank `tl.gather(q,full((1,D),g,int32),0)`,
  widen exact E4M3 values through BF16 to FP32, and reduce K[N,D]×q[D] over D.
* `_simt_pv`: gather one BF16 P row with index shape(1,N), widen BF16 P/V to
  FP32, and reduce P[N]×V[N,D] over N. Both static head loops run g0..7.

Largest source shapes are Q/acc[8,64], K/V[64,64] and logits/P[8,N], with a
temporary[N,D] per head. There is no[G,N,D] tensor. Payload global loads are
outside the head loops in the producer, not in the helpers. Compiler dataflow
must still prove actual payload reuse rather than eight reloads.

**Storage-plan clarification before native work:** this implementation uses
bounded distributed TL arrays; it does not manually allocate the shared
logits/P buffers sketched in `../fa3-next-evidence/B1_SINGLE_FALSIFIER.md`.
Nor does it assert a physical feature16×head8 lane layout. The installed
compiler chooses layouts/shared conversions. This choice keeps the proposed
execution-family question while making layout/resources explicit native gates.

## Precision, ownership and cheap source evidence

The producer reconstructs exactly the control `_fused` AST after replacing
the two SIMT helper calls with the original dot calls, PARTIAL_M with
BLOCK_M and removing added scope assertions. Thus norm/RoPE, Q BF16→E4M3,
scale placement, per-N64 rowmax/alpha/FP32 probabilities/denominator, BF16 P/V,
new masked N16 tail, cache stores and output processing remain the original
source expressions. The SIMT reductions change FP32 association; no bitwise
equivalence or native accuracy is claimed. Per-key online softmax is absent.

Partial allocation is still135168B. Only valid rows0..7 are written at old
BM16 addresses, totaling67584B; the unchanged merge reads those same67584B.
Padded rows8..15 are deliberately unwritten and must not be consumed. Every
split writes its valid FP32 state, including neutral empty-history states.
Only split15 writes the new K/V; it remains necessary even with empty history.
There is no active ticket/counter allocation or merge branch on this route.

Five AST/pure-host tests prove exact reconstruction, baseline preservation,
scope/fallback, helper rank/dtype contracts and fake-host allocations/launch
grids. Sixteen independent geometry tests cover history0/1/10/63/64/127/128/
2047/2048/2049/80000, exact key coverage, page1/64 shuffled mappings and slot0,
unique cache writers, query-head coverage and poisoned padded partial rows.
These are CPU source/model tests; they do not execute Triton or certify GPU
correctness. The source baseline remains unchanged.

The [pinned Triton3.8 gather contract](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/semantic.py#L1549)
requires equal-rank integer indices and matching nongather dimensions,
returning the index shape. The(1,D)/(1,N) gathers satisfy that public API;
installed native lowering still needs verification. Saved image inventory
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`
records installed Triton3.8.0. No new toolchain/install is needed or proposed.

## Prospective compile-only gate

After independent source/harness review and parent serialization, compile
exactly control producer, candidate producer and one unchanged merge, without
calling any kernel. Freeze page1/unit-scale B1/L1 configuration and official
int32 page/length and int64 cache-location pointer dtypes. Keep all original
four-warp/three-stage options; no fallback compiler flag or variant sweep.

The engineering resource envelope proposed **before seeing native counts** is
<=192 registers/thread, <=65536 shared bytes, zero spill/local-memory traffic
and CUDA legal maximum at least2 CTAs/SM at128 threads. This is a conservative
cost budget, not the old96-register cap or an achieved-occupancy prediction.
One balanced N64×D64 FP32 array alone corresponds to32 register words/thread;
several operands/products plus pipeline, gathers and scalar state can coexist.
Source shapes do not establish real liveness or a successful register budget.

Native evidence must additionally prove eight valid output rows, no hidden
rank3/padded-M16 computation, intended FP32 products/reductions, preserved
BF16 rounding and mask/ownership, no eightfold K/V payload reload, bounded
layout/reduction/shared synchronization, and unchanged merge. Reject a failed
API/resource/reuse/precision gate without changing geometry, stages or
precision to rescue it. Compilation is not numerical or timing admission.

## Opportunity, risk and onward gates

The saved complete-call target is11.477114us; instrumented producer7.8us and
merge1.7us are separate observations, not subtractable phase budgets. Current
padding is about2.016× useful shaped attention arithmetic, while ideal byte
movement is far below the actual time; neither proves scalar SIMT is faster.
Unrolled head reductions, gathers, product temporaries, shared conversions
and reduced tensor throughput can erase the gain or cause spills. The prior
query-hoist and serial-merge failures remain rejected.

Illustratively halving producer activity would save3.9us only under an
unmeasured one-for-one whole-call mapping, modeling1.035061× overall score
(1.233633→1.276886). The prospective useful-result target is>=15% complete
B1 sliding improvement on two fresh fixtures with unchanged-route controls;
that target-only scenario models1.014445× overall. These are scenarios, not
forecasts. This source/compile stage has zero direct official-score effect.

Only a native gate pass permits review of bounded original-oracle/cache,
unchanged-input, page/scales/window-edge and live graph replay checks. Paired
uninstrumented cold-L2 **whole-call** timing is later and must include all
preparation/cache append/merge/gate work. Current3da is the production
comparator. No official check, promotion, GPU admission or paid spending is
authorized by this local source report; selected3da remains the working choice.
