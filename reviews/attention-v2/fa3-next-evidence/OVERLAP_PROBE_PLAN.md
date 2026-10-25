# Gluon overlap probe: installed evidence and narrow falsifier

Local plan, 2026-10-25. No GPU job, runtime import, installation, service or commit. Projection owns the GPU slot. The parent subsequently requested local source implementation in a separate probe folder; production candidate remains untouched.

## Proven eligibility and remaining unknowns

CPU-only `image-inventory/result.json` SHA256 `91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78` finds Triton3.8.0 and Gluon/Hopper source. The installed Hopper file SHA256 `c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc` exposes `warpgroup_mma(a,b,acc,*,use_acc=True,precision=None,max_num_imprecise_acc=None,is_async=False)` and `warpgroup_mma_wait(num_outstanding=0,deps=None)`. Inventory executes zero package imports/contexts/kernels; complete bodies, layouts and dependency lowering remain untested.

The [v3.8 WGMMA example](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py) supplies public layout/API precedent. Presence is not native compilation proof. Prior M64 BF16 artifacts reached WGMMA but immediately waited; that rejects their schedule, not every async architecture. Current FA3 already has asynchronous/persistent machinery; its recoverable overlap gap is unknown.

## Exact isolated two-arm mechanism

First fixture: full B32/L8, TP8=16Q/2KV, GQA8, D64/page1; M64/N64, four warps. Verify naturally fills64 rows. Decode L1 fills8 and needs a separate reuse justification; padding alone is not an eligible decode improvement.

The local minimal source uses one CTA per request/KV head, the same full historical range and causal eight-token tail in both arms, two BF16 K/V shared slots, and identical merge-free geometry. This isolates scheduling before any split-production integration; it is not a replacement for current FA3. Preserve Q BF16→E4M3→BF16, historical KV E4M3→BF16 inside the kernel, BF16 QK/PV with explicit zero imprecise accumulation, FP32 softmax/partials and identical scale placement. Read lengths and page1 mapping on GPU every invocation. No host synchronization, cache, planner, repack, approximate rescale, warp specialization or QK/PV iteration reorder.

**A:** issue current async QK, immediately wait, then gather/cast/store the next K/V tile into the other shared slot. **B:** issue the same QK, perform precisely that next-tile gather/cast/store, then wait for QK. All score scaling, masks, rowmax, exp2, denominator, old-numerator rescale, PV and accumulator-add operations follow the same order. PV stays synchronous/common. Ordinary pointer gather is chosen first: no cp.async or Hopper TMA-gather claim. Regular affine TMA cannot express arbitrary page1 mapping; no Blackwell import.

## Gates before any measurement

1. CPU AST/source review proves arms differ only in wait placement; masks, payload, math, geometry, tail and live metadata are identical. No imports/compile on this Mac.
2. A later separately serialized compile-only job must prove BF16 WGMMA, dependency-bearing waits, useful gather/cast/shared-store work between issue/wait in B, and immediate wait in A. Reject hidden earlier waits, blocking lowering that removes the intended overlap, spills/local memory or untenable registers/shared usage. No numeric precision rescue or geometry sweep.
3. Prove copy/store completion and shared publication before MMA; never overwrite a slot still consumed by pending QK/PV. Drain before slot reuse/tail. Compile success does not prove execution overlap, correctness or speed.
4. Only after these gates: original oracle/tolerances, zero/partial/64-boundary lengths, causal tail, nonunit scales, independent output/scratch and replay mutations. Then fresh-input paired cold-L2 whole-call timings against both A and current fusion+lookahead FA3 (`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`). Include all preparation and live input work. Beating A alone is insufficient. No performance forecast or official admission follows from this plan.
