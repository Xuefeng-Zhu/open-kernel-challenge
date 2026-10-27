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


## Accepted; result pending at requested push checkpoint

Accepted once2026-10-25T02:22:12.437286Z, job6e761bdd1d9747ad.
Exact source commit030924103fe180b6645122ec3b02b2ed2af10b59,
SHA2c7bd6a0456c18555d0579b046e741086e527e2290dc3a11b642f8fb37eea357.
New budget1/10used,9left. No terminal outcome or GPU gain claimed yet.
Monitor the same job; do not dispatch a duplicate. E6 remains selected.


## Terminal outcome — adopt A2-01

Check6e761bdd1d9747ad passed16prerequisites+12scored shapes. Score1.0002799616×,
CI[.9852757622,1.0152841611], +6.4591% over E6 .9395909196× with nonoverlapping
parent/new intervals. This resolves improvement over E6, NOT an overall win over
SGLang: the new interval includes1.0. Full verify1 candidate76.348→36.173µs,
same-receipt SGLang36.176µs; normalized latency cut53.6046%, supporting >=20% goal.
All11unchanged controls below5%; minor cross-run shifts are not assigned to code.
Prior20%/matching-SGLang models .9576911/.9947651× were scenarios; actual1.000280×.
Full six-load arithmetic,12times/raw samples, model comparison and profile are
in A2-01-RESULT.md, -assessment.json, -per-shape.csv and -official-result.json.

Native new-path output/new-cache correctness includes the official B1/L8 full
ctx1048576 prerequisite (default page64, unit scale). Its scored path is page1,
unit scale with fresh-content graph timing. This does not prove explicit target
GPU same-address ABA with changing actual lengths, nonunit scales, untouched-old-
row assertions or concurrent calls. CPU stand-ins cover those separately;
unchanged E2 fallback still has shared-ticket concurrency limitations.

New attention budget1/10used,9left; actual platform17/20checksleft,0scoredused.
No extra check, Modal, credential, paid or ranked action. This result arrived
AFTER the one-time requested private push snapshot and remains local pending
separate future push authorization. GPU slot terminal before next experiment.
