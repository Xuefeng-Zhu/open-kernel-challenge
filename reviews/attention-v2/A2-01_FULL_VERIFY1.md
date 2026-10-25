# A2-01 — extend unchanged full FA3 adapter to verify1

Parent is selected E6 `mixed_fa3_sliding.py`, source5202cbe/SHA7a3bc586,
official4f626ab7437a42be, score0.9395909196×. Candidate is
`mixed_fa3_verify1.py`. The sole algorithm/dispatch delta is adding batch1 to
the full-attention q_len8/no-RoPE batch tuple. No helper, norm/precision, cache,
window, fullB16/B32 or sliding path changes. Every other eleven scored shapes
is an unchanged control. The parent is preserved.

Problem: full verify1 remains76.348µs vs same-receipt SGLang35.424µs after E6.
Hypothesis: existing official full FA3 adapter reduces this path's normalized
latency by at least20%. Its full/page/causal/scales/query offsets/scheduler
contract is already implemented; q_len8/B1 adds no new rounding policy.
Required BF16 normalized Q→FP8 E4M3, cache FP8, BF16 attention output before
RMSNorm/gate remain the official baseline behavior. No RoPE for this target.

Holding other E6 times fixed, the exact20%-cut and matching-SGLang arithmetic
scenarios are stored in A2-01-cpu-preflight.json; neither is measured performance.
Reject the target goal if native correctness fails or target reduction<20%.
Investigate other-path regressions>5% without automatically blaming source for
cross-run shifts; aggregate CIs determine whether a total gain is resolved.

Four new CPU stand-in/source tests passed:

```
.venv/bin/python reviews/attention-v2/test_mixed_fa3_verify1.py
```

AST proves every helper unchanged and exactly one batch literal added. Dispatch
covers112 nearby configurations. Full-oracle cases cover ctx0/1/64/2048, coherent
attention, page64/nonunit scales, live cache0, untouched rows and same-address
A→B→A with changing actual lengths/content. CPU PyTorch stand-ins do not execute
native FA3/Triton or CUDA graphs. Official fullB1/L8 ctx1048576 prerequisite will
exercise the new path if the single admitted check reaches it; no local claim
of native execution. Target explicit GPU ABA/page64/nonunit/untouched cache
coverage remains absent unless separately reported by the official receipt.

Prospective new-cycle command after exact source commit:

```
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-01 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-01 --cycle2
```

New budget0/10 used before dispatch; old E6 belongs only to the exhausted old
six-request cycle. Projection7checks unchanged. One GPU check at a time, off-board
$0 only, no Modal/push/new credentials/ranked submission. Never repeat an uncertain
POST; source/acceptance/receipt will be linked in the result commit.
