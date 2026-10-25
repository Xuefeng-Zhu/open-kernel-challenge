# P050 — QKV-only public epilogue route expansion

Status: local preparation; no P050 GPU or official check. Selected P046 remains
unchanged at its official 1.1841399408925375×; P039 is the preserved rollback.

实验卡：P049 证明加零偏置确实改变了 QKV M128 的库内核选择，但该点只有
3.78% 的原始延迟收益，而且 unpooler 退步。这一轮完整恢复 unpooler，仅保留
QKV 的公开 addmm 路线。检验收益是否跨多个批量复现，而不是挑一个好看的点。
若整体价值或控制组检查不过关，不提交官方检查。

## Evidence and isolated change

P049's all-shape proposal was rejected by its predeclared gates; that decision
remains final. Its H200 screen passed 326 oracle comparisons. QKV M128 changed
from a split-K main plus reducer to a single bias kernel, yielding a paired gross
latency cut of 3.7804% in 10/12 pairs. Unpooler M16's bias variant regressed
3.5852%; it is removed completely, not given a relaxed threshold.

Candidate `candidates/projection_qkv_zero_bias.py`, SHA256
`b08b6e371fef83851268a8dbfa1159d1b6d5a8e141a410a596dfc3f159e0da8b`.
Performance parent is exact P046 SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Design predecessor is rejected P049; both originals remain immutable.

Only canonical contiguous BF16 QKV M>1 uses
`torch.addmm(zero_bias, X, W.T, beta=1, alpha=1, out=out)`. Extra prepared state is
4,608 bytes of input-independent BF16 zeros. All other original/packed weights,
precision settings, strides, output ownership and paths remain P046. Zero-bias
math is unchanged; different rounding/library workspace may still affect results
and must pass the original FP64 tolerance. This is not proof of bitwise identity.
No global backend setting, custom JIT, unsupported private binding or installation.

The public API route is supported independently of P047's unresolved exact-judge
FlashInfer inventory. Actual installed kernel selection must still be observed.
Upstream source and attribution are discussed in experiment049; no vendor source
was copied into the submission.

## Score leverage and falsification

`python -B projection_p050_model.py` derives the six-load model from P046's
immutable receipt. Uniform cuts across the five QKV M>1 targets of 5%,10%,20%
model relative score gains of +0.9683%,+1.9576%,+4.0020%. A uniform5.1617% cut
would be needed for +1% overall;7.7004% for +1.5%. P049's M128 cut hypothetically
applied only to that path models +0.1443%. These are sensitivities, not forecasts
or achieved scores; native gross ratios are not official calibrated latencies.

This explicitly limits the value of rescuing only one favorable P049 point.
Before any official request, native validation must show meaningful wider gains.

Predeclared native gates, fixed before any P050 run:

- All original-oracle, replay, input/state immutability and output checks pass.
- Five-target geometric gross latency cut at least4%; frozen-receipt sensitivity
  gain at least1%, computed from every target including regressions.
- At least four of five target medians improve, and each of those four wins at
  least8/12 reversed-order pairs. No target latency regression above2%.
- Nearby QKV latency regressions above 2%, or unchanged-control absolute median
  movements above 2%, require investigation and prevent clean readiness. Nearby
  improvements are retained without a failure flag because those paths also use
  the candidate route. This distinction was corrected during local design review
  before any P050 measurement. A flag is not automatic proof of a code defect;
  retain all samples and assess drift rather than discard them.
- No observed candidate copy or recognized unfused fallback; required parent and
  candidate activity available. Symbols alone do not prove tactic or instruction
  identity, and absence of a visible copy does not prove no hidden library work.

Passing these gates only warrants one official full-suite comparison; it does
not promote P050 or establish the modeled score. Thresholds will not be revised
in response to its results.

## Bounded native plan

Nine cases: scored QKV M8/16/32/128/256; nearby QKV M127/M129; unchanged QKV M1
and unpooler M16. Each uses one shared X address, distinct outputs and immutable
original/prepared weights. Before and after timing: normal A, outlier B, A again,
signed sparse power-of-two rows and zero. Six excluded warmup rounds and24 fresh
measured rounds make12 fixed reversed-order pairs. Each replay follows256MiB
L2 eviction; event initialization occurs before capture. No sample trimming,
fixed-cost subtraction, concurrent-call claim or numerical tolerance change.

Only after all timing, collect one activity replay per runner for QKV M8/M128/M256
and unpooler M16. Activity names/durations are separate diagnostic evidence, not
acceptance timing or hardware counters. Full completion totals728 oracle checks:
720 across nine paired cases plus8 activity replays. Required official QKV M512
and other official cases remain GPU-unverified for P050 until the official suite.

Use existing image/login and a fresh paid-cap/free-credit check. One ephemeral
H200; unchanged bounded launcher limits:120s internal,180s child,195s function,
45s startup,240s watchdog. Current resource-based reserve about$0.408 free credits,
actual paid$0. No launch until the coordinator grants a fresh serialized slot.
All code/harness files must match the reviewed commit and explicit allowlist.
No P050 app or official request exists at this preparation checkpoint.

## Why not a new larger-pooler shard choice

P044's actual P039 pooler M128 trace records128 CTAs,384 threads/CTA,
168 registers/thread and213,252 shared-memory bytes. This is launch/resource
metadata, not achieved occupancy, HBM traffic or stall evidence. The known H200
has132 SMs, so merely observing a nonmultiple grid is not a strong parallelism
argument. P045 removed its reducer yet regressed5.542%, with a changed main symbol.
Its actual BMM grid, registers/shared memory, tile-axis convention and split factor
were not saved. Opposite unverified tile-axis assumptions double different
operands in logical-read models; neither is an observed traffic fact.

The exact current P046 M128 pooler time23.157941µs is28.9961% of one of six loads.
Holding other paths fixed,10%/20% target cuts model only+0.4916%/+1.0007% overall;
+1.5% overall would require29.4722%. Reaching TASK's rounded14.6µs floor models
+1.9070%, an optimistic sensitivity only. See
`results/router-p049-pooler-followup-model.json`. Another shard-count sweep lacks
both a causal distinction and sufficient supported upside. A matched existing
P045/P046 activity trace could fill missing launch facts, but would still not
measure traffic or guarantee a better candidate.

## Local preflight

Thirteen candidate CPU tests passed; together with six bounded-launcher tests,
coordinator rerun passed19 tests in1.599s. Static submission checks also passed.
Exact AST comparison confirms every non-QKV P046 path restored. Full-K M2/M8
normal/outlier/sparse/zero cases and M8 same-address CPU ABA use the original
oracle; larger/nearby M arithmetic uses reduced K131. All30scored+9prerequisite
metadata routes are covered. These are not CUDA or all-shape numerical passes.
The optional NumPy initialization warning did not fail tests; no package installed.

```bash
python -B -m unittest tests.test_projection_qkv_zero_bias tests.test_projection_p050_modal -v
python -B static_check.py --submission candidates/projection_qkv_zero_bias.py
python -B projection_p050_model.py
```

Native worker review and freeze are separate; no external run has started.

Independent source review approved the candidate with CUDA limitations. A final
pre-run protocol review distinguished changed nearby paths from unchanged
controls: nearby gains are expected potential benefit, while nearby regressions
and either-direction unchanged-control drift need investigation. No P050 data
existed when this correction was made; other gates remain unchanged.

## Final local freeze

Coordinator combined rerun: **40/40 local tests passed in1.532s** (13candidate,
21worker/protocol,6launcher). Independent candidate and worker/launcher reviews
found no blocker, subject to a fresh coordinator slot and budget/auth preflight.
Worker `projection_p050_qkv_screen.py`, SHA256
`ab822cb0707f1b9b062b061bbc5f39ba5b97bf7d5fd8b75ccecc0560af838dfd`.
Launcher `projection_p050_modal.py`, SHA256
`baffdcd453bf590a5daea6ff98242a3711b5b5caffe133446e364bdbfc597f66`.

```bash
python -B -m unittest tests.test_projection_qkv_zero_bias tests.test_projection_p050_qkv_screen tests.test_projection_p050_modal -v
python -B projection_p050_modal.py self-test --control-root /path/to/control-workspace
# After fresh coordinator slot and account preflight, on the reviewed commit:
python -B projection_p050_modal.py launch --frozen-head <40-character-commit> --control-root /path/to/control-workspace
```
