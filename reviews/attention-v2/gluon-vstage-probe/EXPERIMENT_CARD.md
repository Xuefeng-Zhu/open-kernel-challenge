# Prospective V-stage scheduling experiment

**Claim:** with the next tile's raw FP8 K prefetch common to both arms, moving
V gathering plus K/V BF16 conversion/shared publication across the current QK
wait may retain a useful native scheduling difference. This is a separate
local architecture-feasibility component. It makes no claim that FA3 lacks
overlap or that this prototype closes the competition gap.

## Evidence and isolated change

The earlier `../gluon-overlap-probe` compile result SHA256
`8b5f8717c56de7a6ea6971840bd4f2252aced16c86e1fb3aca26ed6af1a859ea`
showed BF16 WGMMA, two K/V shared slots, zero spills/local traffic and a real
conversion/publication placement difference. Its fully serialized native
control criterion failed: ptxas moved page/K loads ahead of that arm's wait.
That original criterion and all old code/artifacts remain unchanged. The new
design explicitly makes raw K prefetch common; it does not retroactively pass
the old experiment.

Both new arms issue QK, read live next-page mapping and raw FP8 K, then call
one identical publication helper. The helper gathers raw V, widens/stores K,
widens/stores V and executes the existing proxy-publication fence.

* `v_after_wait`: QK issue → common page/raw-K prefetch → QK wait → V-stage.
* `v_before_wait`: QK issue → common page/raw-K prefetch → V-stage → QK wait.

All score math, masks, scales, FP32 online state, pure-zero-C synchronous BF16
PV, numerator rescale/add, slot updates and output are unchanged between arms
and structurally match the prior probe. The only arm selector is
`V_BEFORE_WAIT`; the old labels "serial"/"overlap" are deliberately invalid in
the new configuration. The compiler is free to hoist legal work. There are no
manufactured dependencies, volatile loads, new barriers, precision changes,
ISA knobs or geometry knobs to force a slow control.

## Fixed contract and costs

Kernel API is `attention_vstage_kernel(Q,KCache,VCache,PageTable,HistoryLengths,
KScale,VScale,Out,HKV,HQ,PT_STRIDE,SM_SCALE,QK_FACTOR,V_BEFORE_WAIT)`.
The last six arguments are constexpr. Defaults are B32,HKV2,HQ16, stride86008,
SM_SCALE0.125,QK_FACTOR5.4730064863875; grid(32,2), four warps, one compiler
stage, fixed M=N=D64,L=G8. Q is **unscaled** post-RMS/position BF16
`[B,8,HQ,64]`; the kernel rounds BF16→E4M3→BF16 before scale use. FP8 caches
`[physical_rows,HKV,64]` already contain history plus eight new tokens.
`PageTable` is int32 `[B,PT_STRIDE]`; `HistoryLengths` is int32 `[B]` and
excludes these eight tokens, matching `../official/reference.py:188/239`.
Page1 mapping/lengths and FP32 K/V scales are live GPU inputs. Output is
separate raw-attention BF16 `[B,8,HQ,64]`.

RMS/RoPE, cache append, output norm/gate and production split/merge are outside
this isolated input/output boundary. Any later full-call candidate must include
their original captured/timed work. No production dispatch or source
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
is edited. Decode M8, B1 and sliding paths need separate evidence and reach
plans; natural-M64 verify is not a decode or complete-family solution.

Two K and two V BF16 slots require32768 nominal shared bytes. Raw K and its
mapping remain live across the wait in both arms and may increase registers;
actual register/shared/spill costs are unknown until native compilation. All
masked payload lanes read zero and full shared stores clear padding. The
current slot is never published over while QK/PV consumes it; synchronous PV
drains before subsequent reuse. This is a source lifetime model, not a runtime
race certificate.

## Opportunity ceiling and predeclared gates

This local/compile-only stage has **zero direct official-score effect**.
If QK duration is Q and movable stage duration is S, hiding opportunity cannot
exceed `min(Q,S)` per tile in a simple two-stage model; neither duration nor
the production exposed-stage fraction has been measured. There is no numeric
latency or weighted-score forecast. Even the parent's much broader hypothetical
50% cut across **all full paths** models score about1.889, below the22:57UTC
leader snapshot2.9097; this verify microcore alone has less reach. Those are
planning scenarios, not recoverable-loss measurements or score predictions.

1. **Cheap local gate:** AST specialization must show common raw-K prefetch
   before the wait in both arms and identical remaining code after removing
   the wait placement. Tests cover exact casts/masks, zero padding, causal
   keys at0/55/56/63/64/65/8191/8192 and a two-slot lifetime model. No GPU import
   or execution is performed locally.
2. **Future compile-only falsifier:** use the same installed image, fixed
   configuration and hash-bound pair, with no kernel execution. Verify matched
   PTX logical work/load volume and common K prefetch. Inspect actual dataflow
   for a remaining V-load/conversion/shared-publication contrast around the
   corresponding QK drain, documenting partial hoisting honestly. The revised
   gate does not require every address/K load to occur after QK issue.
3. **Stop conditions:** if both native schedules collapse to the same relevant
   work/wait placement, do not benchmark. Reject lost K/V buffers, unsafe reuse,
   changed precision, spills/local-memory traffic or unusable resources. Report
   register increases and legal resource limits without achieved-occupancy
   claims. Missing native dataflow evidence is inconclusive, not a pass. No
   forced slow control or knob rescue is allowed.
4. **Onward work is separate:** a surviving native contrast only admits review
   of source-bound original-tolerance numerical/replay cases, including zero
   history, tails, nonunit scales and fresh mappings. Timing acceptance and
   whole-call integration need a prospective plan after that evidence. Beating
   one arm alone is insufficient; current FA3 remains the full-call comparator.

Local cost is$0 with no remote job. There is no new controller, upload,
credential, install, official check or GPU admission in this preparation. A
later compile job requires its own reviewed resource/watchdog accounting and
the parent's serialized slot; actual paid limit remains$0.

## Source provenance

Installed source availability is bound to CPU inventory SHA256
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`,
image`im-cPTbGpNiX3RarKResPgX7h`, Triton3.8.0, Hopper API source SHA256
`c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc`.
The earlier probe established native acceptance of its shared layout and wait
API; this refactor still needs its own compilation. Public API precedent is
the [Triton v3.8 WGMMA tutorial](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py).

Kernel SHA256:
`6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`.
