# P052 — execute the matched orientation mechanism

Status: local preflight in progress. No P052 app admitted, GPU result, timing,
official check or promotion at this checkpoint. P046 remains selected.

实验卡：P051 已证明调换矩阵计算方向能改变真实 GPU 指令和等待顺序。
这次检验它是否既正确又有实用速度。先验证边界形状与同地址 A→B→A
图重放，再同时对比原自定义内核和当前 P046。只战胜较慢的自定义对照
不算成功；本轮不改分块、split-K、精度、归约或 P046。

## Evidence and single hypothesis

Frozen P051 source: `projection_p051_orientation.py`, SHA256
`445ab7d44e36cdceafdf1ecdbe19d7fafa247d8033ab580ab38f74b98f3533db`.
P051 native compilation emitted BF16 mma.sync for the token-first control and
BF16 WGMMA for the weight-first variant. The latter keeps one committed group
potentially outstanding while issuing the next copies. Registers changed from
42 to 36 per thread; shared memory doubled from 10,240 to 20,480 bytes; raw
driver local-memory metadata was zero for both. These are static observations,
not proof of achieved overlap, occupancy or speed.

Hypothesis: preserving the logical tile, arithmetic, split partition and final
reducer while changing orientation can improve whole-call graph latency enough
to justify developing this dataflow further. Layout conversion and doubled
shared memory are explicit possible costs. P046 may already have a superior
vendor route, so an improvement over the matched control alone is insufficient.

## Frozen controls and scope

P051's fixed BM16/BN64/BK64, split-K 8, four warps, two stages, BF16 operands,
FP32 accumulation/partials/reduction and BF16 output remain unchanged. The
external partial tensor remains `[split, token, column]`. No candidate algorithm
is edited in this experiment.

The current working control is the original P046
`candidates/projection_pooler_column_batch2_hybrid.py`, SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Its real prepared state and dispatch are retained: M8 uses the output-column
batch-two BMM, while nearby M7 uses the transposed-weight matrix multiply.
Neither route is forced by modifying the candidate or library globals.

The first stage must pass the original oracle on all five P051 cases: canonical
pooler M8, nearby M7, M15/N65/K131 tails, M1/N63/K1, and M16/N67/K521 split tail.
Its 70 comparisons cover normal/outlier/sparse/zero inputs and captured graph
A→B→A content changes at the same address for both orientations. Timing cannot
start after a compile-only or incomplete receipt.

The execution screen times canonical pooler M8 and nearby M7 against both the
matched orientation control and P046. It uses fresh input values, initialized
CUDA events, 256 MiB L2 eviction before each replay, six predefined warmup
rounds and 24 measured rounds (12 reversed-order pairs) per comparison. The
comparison order also alternates across rounds. Oracle work occurs after both
members of each pair. All samples and tails are retained. Input and prepared
state mutation checks, additional pre/post A→B→A/sparse/zero checks, and exact
source hashes are recorded. No post-hoc sample deletion or fixed-cost subtraction.

## Predeclared gates and modeled value

Correctness is mandatory before any timing interpretation. Both M cases must
show at least a 10% median paired latency reduction against the matched control,
with at least 10 of 12 paired wins each. If the ratio is control/candidate, a
10% latency reduction means ratio >= 1/0.9, not merely 1.1. In addition, the
candidate may be no more than 5% slower than actual P046 on either case.

Passing these gates justifies a broader architecture study; it does not promote
this narrow prototype or establish an official score. P046's pooler M8 occupies
33.378% of one of six equally weighted loads. Even eliminating that path entirely
would only model about +7.0% overall; a 20% target reduction models about +1.16%.
These are conditional score sensitivities, not predictions. The research value
is whether a precision-preserving mechanism can later extend across higher-weight
pooler/QKV/unpooler/O paths. That extension is not implemented or validated here.

## Operational boundary

One ephemeral H200 container on the existing image, existing authentication and
free credits only, paid cap $0. Two bounded child stages, no automatic retry,
no persistent deployment, no official check. The launcher validates exact committed
allowlisted bytes, current billing/credit guards and the serialized GPU slot.
It must stop the app and verify zero tasks after success, failure or timeout.
Raw reports and PTX are source-bound and returned in bounded chunks. Credentials
and unrelated repository material are excluded.

Results, exact commands, checks, app ID, observed billing and conservative reserve
will be appended after execution. Missing GPU evidence remains explicitly missing.

## Local preflight

Independent source review found no blocker after receipt schema reconciliation.
The timing worker uses isolated function globals and retains the frozen P051
function bodies and constexpr annotations. It counts actual oracle records and
requires 320 screen checks, separately from the prior 70 correctness checks.

Mac Python 3.12.13, standard-library tests only:

```bash
python -B -m unittest \
  tests.test_projection_p052_orientation_screen tests.test_projection_p052_modal \
  tests.test_projection_p051_orientation tests.test_projection_p051_modal
```

Result: **59 passed in 0.317 seconds**. An earlier local test used the wrong
summary key `min` instead of the existing `minimum`; the fixture was corrected.
No kernel or frozen helper changed to address it. Four launcher self-test guards
also passed. These tests do not execute Triton or a GPU.

Frozen new worker SHA256:
`042c395d20ffa6b191629aa79d9b68b9766cb92ea94df56d7763e741d46cdd59`.
Launcher SHA256:
`40b8c8aca0e6e1e673da44bf278fff8326791c6009839918c10e0c3f4a5e0d70`.
The allowlist contains 14 files, 156,788 bytes; no credential/configuration files.
The source-bound launcher enforces 65-second internal/80-second hard limits per
stage, 180 seconds per function, 45 seconds startup and a 225-second app watchdog.
The 255-second resource reserve plus margin is **$0.38609925** at the observed
rate floor, raised if live rates are higher. Current browser evidence shows
$27.86 credits, paid $0 and spend cap $0. Read-only CLI verification at
23:28:17 UTC found existing authentication valid and every app stopped.
