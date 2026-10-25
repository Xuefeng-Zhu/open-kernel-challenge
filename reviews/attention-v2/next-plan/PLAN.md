# Next attention experiment: investigate full verify's matrix pipeline first

2026-10-25. Local analysis only. No GPU launch, official check, installation,
credential action or candidate edit in this turn. A2-11 remains selected.

The best next investigation is a **compile-gated full-verify producer**, not
another small full-decode B32 timing adjustment. First determine whether exact
widening of already-quantized Q/K enables Hopper warp-group matrix instructions
for the naturally full 64-row verify tile. Do not change runtime dispatch or
write a broad new attention algorithm before that evidence exists.

## A2-15: useful component evidence, insufficient reason for another score check

The paired Modal test measured **whole `kernel_fn` graph replay for one path**,
not a subkernel microbenchmark: raw medians 312.032→295.904 µs, median of six
reverse-order pair ratios 1.054731, corresponding to a 5.189108% latency cut.
It covered the full decode B32 scored family plus one adversarial case. It did
not measure all twelve paths or an official aggregate. The official A2-13
receipt reports 316.411810 µs for this path after the judge's calibration and
paired-ratio normalization. Do not subtract the Modal absolute times from that
official time, or combine the two calibrations.

The official score has six equally weighted loads. Within each load:

`C = 46 × sliding + 16 × full; score = geomean(S / C)`.

For A2-13 decode B32, `C = 46×20.270871 + 16×316.411810 = 5995.049004 µs`.
The full component's share is `5062.588958 / 5995.049004 = 0.8444616473`.
Transferring only the measured relative cut, while freezing every other time:

`score factor = (1 − 0.8444616473 × 0.0518910800)^(-1/6) = 1.0074961443`.

That is **modeled +0.7496% overall**, or a hypothetical 1.220069 score from
A2-13's historical 1.210991. Neither is achieved. Using the separately calibrated
paired ratio instead models +0.7574%; transferring each of the six individual
paired ratios gives +0.6884–0.7928%. This range is sensitivity, not a confidence
interval. Different runtime, inputs, controls and calibration remain limitations.

This one path would need a 10.12% cut to model +1.5% overall, or 19.24% to model
+3%. The actual 5.19% cut also misses its predeclared 10% target. **Defer a
performance-only official confirmation**: it has low information value relative
to the official minimum ±1.5% reported interval. An official check would still
have correctness value before eventual complete-candidate adoption; it is not
forbidden or statistically useless. Do not promote A2-15 without that coverage.

## Where score leverage remains

These are saved A2-13 official whole-call times and their same-run reference.
The full table, raw medians and six-load reconstruction are in `evidence.json`.

| Path | Candidate µs | SGLang µs | Share of its own load | Modeled gain if this path alone loses 20% latency |
|---|---:|---:|---:|---:|
| Full decode B16 | 200.209 | 199.936 | 83.01% | +3.07% |
| Full decode B32 | 316.412 | 384.832 | 84.45% | +3.13% |
| Full verify B16 | 204.454 | 204.208 | 75.90% | +2.78% |
| Full verify B32 | 389.167 | 388.992 | 82.90% | +3.07% |
| Sliding verify B1 | 17.141 | 22.656 | 62.32% | +2.24% |

The batched full FA3 paths are essentially at reference speed. Separate
instrumented activities attribute 189.8/373.9 µs to FA3's mainloop in full
verify B16/B32 (91.8%/95.5% of activity sums). This locates the expensive stage;
it does not establish memory bandwidth, occupancy or a hardware stall cause.
Profile sums cannot be substituted for calibrated whole-call timings.

Conditional whole-path models, with every other time fixed:

| Target family | 10% latency cut | 20% latency cut | 30% latency cut |
|---|---:|---:|---:|
| Both full verify B16/B32 | +2.80% | +5.94% | +9.50% |
| Those two plus full decode B16 | +4.29% | +9.19% | +14.85% |
| All three sliding verify paths | +1.78% | +3.69% | +5.74% |
| Four current custom fallback paths | +3.00% | +6.38% | +10.23% |

These are value scenarios, not forecasts or achievable ranges. QK's dynamic
fraction of runtime is unknown, so even doubling its arithmetic throughput
does not justify assuming a 50% whole-path reduction.

## FP8 encoding is not native FP8 matrix arithmetic

A2-13 keeps E4M3 Q/K operands in TTIR (`a213.producer.ttir:169`). The saved
TTGIR selects MMA version 2 and adds FP16 conversions (`:16`, `:200`,
`:284–286`); PTX contains `mma.sync...f32.f16.f16.f32` (`:772–810`). PV uses
BF16 instructions. A2-15 has the same precision lowering. These are actual
native artifacts for full decode B32, not evidence for every specialization.

Triton v3.8.0's Hopper eligibility test requires M divisible by 64 and rejects
FP8/F32 dots with `max_num_imprecise_acc < 32`. Our decode uses M16 and an
explicit bound of zero. Both independently prevent MMA3. The following
MMA2 lowering promotes FP8 to FP16 on sm90, matching the artifacts. The exact
wheel compiler Git revision was not saved; this is version-tag source plus
consistent emitted behavior, not a traced pass history of the installed binary.
[Eligibility](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Analysis/Utility.cpp#L1128-L1155),
[selection and promotion](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L900-L937).

Changing the precision flag is consequential. NVIDIA documents less-than-single
precision intermediate accumulation for FP8 WGMMA with an F32 destination.
BF16 WGMMA with F32 accumulation has a stronger precision guarantee, although
reduction order is unspecified. Therefore do not silently remove the zero
bound to obtain a faster instruction, and do not call this a harmless knob.
[PTX WGMMA precision](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).

The IR's `inputPrecision=tf32` field is not proof of TF32 multiplication of
these FP8 operands; that setting controls FP32 input multiplication.
[Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).

## Ranked experiments and rejection gates

### 1. Full-verify matrix pipeline feasibility — recommended first

**Parent and problem:** preserved A2-13 source, SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Its custom producer can be specialized for L8, but the scored full verify
entrypoint currently dispatches to FA3. This probe does not alter that routing.

**Hypothesis:** verify's `8 tokens × 8 GQA heads = 64` useful query rows can
use warp-group matrix operations without padded query work. Exact widening of
already-E4M3 Q/K to BF16 may remove the FP8 accumulation-policy blocker and
permit a different matrix pipeline while retaining represented input values.

**Exact isolated change:** for compile-only full B16/B32, L8, D64 specializations,
change only the QK operand selector from E4M3 to BF16 *after* the existing E4M3
quantization. Cache storage stays E4M3. Retain BF16 PV, FP32 softmax/partials,
`max_num_imprecise_acc=0`, partition, masks, new-token ownership, tickets/merge,
warps and stages. Compile the unchanged specialization as the control. Do not
generalize A2-15 lookahead or change merge simultaneously.

All 254 finite E4M3 encodings widen exactly to BF16 and FP16 on the local CPU
representation test, including signed zeros. This proves input representation
only, not GPU dot equivalence or final attention correctness. This proposed
route uses **BF16 WGMMA**, if emitted; it is not native FP8 peak-throughput mode.

**Compile gate:** inspect TTGIR/PTX/SASS for actual warp-group operations and
wait/commit placement, register/spill/shared-memory changes and any early wait
that serializes the products. Reject a redundant rewrite if generated work is
unchanged, or if a newly evidenced resource problem makes execution implausible.
Instruction presence alone is not evidence of overlap or speedup.

**Execution gate, only after that:** validate both implementations against the
official oracle with cache write/untouched-row checks, page1/page64, scale1/64,
changing actual lengths and same-address A→B→A graphs. Preserve the original
BF16-normalization boundaries. L8 ticket ownership is invocation-local but
overlapping independent calls still needs its own test. Then compare whole-call
timing against the current FA3 route, with cold-L2, predefined warmup and all
samples retained. Do not benchmark only the matrix multiply.

**Value and falsification:** a 10–20% reduction in *both complete FA3 paths*
would model +2.80–5.94% total. There is currently no measured likelihood of that
range. Old E2 full verify was 339.220/588.574 µs, versus present FA3
204.454/389.167 µs in different jobs. That illustrates a roughly 39.7%/33.9%
custom-producer deficit, not a paired comparison. Simply returning to E2 is
rejected. Require a distinct compiled mechanism, full correctness and repeated
≥10% whole-path reduction on both targets, with no unexplained >5% control
regression, before complete-file official verification. If the probe cannot
justify bridging this gap, stop this route rather than stacking unmeasured edits.

### 2. Sliding verify: eliminate the split-combine stage

**Parent:** selected A2-11's existing fused-preparation FA3 sliding verify path.
**Exact change:** only set supported FA3 `num_splits=1` for L8/window2048; keep
all normalization, precision, page/causal bounds and postprocessing unchanged.
First verify the installed wrapper supports the documented parameter. No saved
experiment currently overrides FA3 splits; this is stage elimination, not a
split-count sweep.

**Hypothesis and risk:** remove the 3.9–4.2 µs combine activity. However a packed
unsplit query exposes only approximately 2×B tiles, at most 64 for B32 on
132 SMs and just two for B1. Lost parallelism may cost more than the merge saves.
Current profiles include PDL overlap; subtracting combine activity directly
from latency would be invalid. Conditional 10–20% reductions across all three
paths model +1.78–3.69%, not predicted gains. Accept only correct whole-call
gains on the affected shapes, no >5% regressions, and a useful weighted effect;
otherwise retain the original FA3 scheduling. Lower priority than experiment 1.
[Documented wrapper](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_attn_interface.py#L609-L640).

### 3. Generalize the proven fused preparation to large full FA3 paths

**Parent:** A2-11, preserving the FA3 mainloop. **Exact change:** route only full
decode B16 and full verify B16/B32 through the already present fused Q/K norm,
FP8 query and cache-write preparation; retain all FA3/gate arguments. This
extends A2-05–08's successful mechanism but is not a new core architecture.
The main risks are BF16 rounding boundaries, non-unit scaling and live slot0.

Even assuming 5–7 µs saved in each of those three complete calls models only
+0.84–1.19% overall on A2-13's times. These are assumed savings, not profile
subtraction or measurements. Local implementation would be inexpensive, but a
dedicated score check has lower expected information value. Keep as a later
correctness-tested composition option, not the award-focused main experiment.

## Explicitly rejected shortcuts and limits

- `pack_gqa=True` alone likely repeats existing behavior: the pinned FA3 source
  forces packing for paged non-TMA or split paths. Its paged TMA path requires
  page size divisible by the KV tile; scored page1 cannot be assumed to use TMA.
  The official installed binary's exact identity is unverified, so this is an
  upstream explanation, not a captured FA3 instruction trace.
  [FA3 dispatch](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_api.cpp#L330-L428).
- Padding decode to M64 wastes eight times its useful query rows. A future
  transposed-QK probe (`K @ Q.T`, then transpose scores back) could avoid that,
  but requires layout/synchronization and accumulation scrutiny. It remains
  secondary to verify's naturally full tile, not another current B32 tweak.
- No repeat of failed deferred-denominator state, serial merge, query hoist,
  blind wider tiles or an unchanged `num_stages` flag. Their historical results
  remain preserved. Native code/dataflow must distinguish the next mechanism.
- No FA3 binary register/shared-memory/stall counters are available for these
  verify paths. The 96 registers / zero reported spills from A2-13/A2-15 belong
  to the full-decode B32 producer and cannot diagnose FA3.

## Local validation, provenance and resource status

`analyze.py` validates frozen source/receipt hashes and reconstructs both official
scores from all twelve timings. Six arithmetic tests passed. Three CPU tests
in `test_widening.py` passed; they establish finite E4M3 representation and
tile-row counts only. No candidate was modified or compiled in this turn.

```sh
.venv/bin/python -B reviews/attention-v2/next-plan/analyze.py --self-test
.venv/bin/python -B reviews/attention-v2/next-plan/analyze.py
.venv/bin/python -B reviews/attention-v2/next-plan/test_widening.py
```

Research consulted primary sources read-only; no third-party implementation was
copied or executed. Existing attribution/licenses remain. Two independent
read-only reviews covered compiler lowering and FA3 path prioritization.

The last verified Modal state remains stopped/tasks0 for
`ap-DjbFQfnZXQjpr1Zgt3rtFK`. No new app or check was dispatched. Conservative
accounting stays $1.919496218/$2, paid $0, remainder $0.080503782. That remainder
does not fit the previous $0.146368320 bounded-run reservation. No new GPU budget
or credential authority is assumed; this plan is ready for a separately scoped
feasibility decision. Platform quota was not refreshed because no dispatch was
planned. A2-11's 1.187678756× remains selected; A2-13's 1.210990803× remains
promising with overlapping intervals; projection is unchanged.

## 中文实验卡

**问题：** full verify B16/B32 仍接近 SGLang，主要时间在 attention 主循环。
**假设：** 8个词元×8个查询头正好形成64行；保留 FP8 数值、无损转 BF16
进行矩阵运算，可能让编译器采用 Hopper 的 warp-group 指令而不放宽累加精度。
**下一步：** 先比较生成指令与资源，再决定是否实现完整候选；不先花一次官方测速。
**价值：** 若这两条完整路径以后实测快10–20%，总分模型提升约2.8–5.9%，目前未实现。
**淘汰标准：** 指令/数据流没变化，正确性不通过，或整条路径仍比 FA3 慢，就停止该路线。
**当前：** A2-15 目标快5.19%只对应约0.75%总分模型收益；保留证据，不替换 A2-11。
