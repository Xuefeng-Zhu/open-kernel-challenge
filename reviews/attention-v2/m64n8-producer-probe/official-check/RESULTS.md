# Direct official check: correctness PASS, performance rejected

Official nonranking check **3e0a7ef51d4849c6** reached `done`, `passed=true`,
returncode0 with no flags/rejections. All **16 prerequisites and 12 scored
benchmark calls pass** the original oracle. The whole score is
**1.0995474217354162**, 95% interval **[1.083054210409385,1.1160406330614474]**,
versus working selected **1.2336330588627955**. The official point-score
change is −10.8692%; this complete producer is rejected for selection.

The candidate is byte-identical to the reviewed source:
`96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970`,
source commit `04447c1cc8f64fdb489b74e1e942a64e0fae3cbb`, kernel SHA256
`eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480`.
The working selected3da source and A2-11 rollback are unchanged.

## Why this ran after the earlier static STOP

The earlier absolute zero-stack/local gate remains recorded as failed. A
later direct request authorized one official check. The common32-byte stack
and inlined RoPE range-reduction local sites also exist in selected controls;
they were reassessed as performance uncertainty rather than invalidity.
Two new independent source/native-ordering reviews found no known correctness
or contract blocker. The old result was not relabeled PASS, and no source,
precision, input, tolerance, flag, tile, split or buffer was changed for this run.
See `../RUNNER_DECISION.md` and the two runner rechecks.

## Changed paths and controls

Adjusted complete-call timings below come from the official paired
candidate-versus-SGLang measurements. Historical selected comparisons use
`(C_new/S_new)/(C_selected/S_selected)`, not same-session selected/new pairs.
Positive normalized changes mean slower.

| Changed path | Selected C µs | New C µs | Reference-normalized latency change |
|---|---:|---:|---:|
| Decode B1 sliding |11.4771|13.7898|+21.0917%|
| Decode B16 sliding |13.9211|17.0724|+20.7343%|
| Decode B32 sliding |20.1036|25.9469|+29.5344%|
| Decode B32 full |304.0319|546.0326|+79.5972%|

All four targets regress. Eight unchanged controls have maximum absolute
normalized drift **1.9018%**, below the declared3% screen. Applying only the
normalized target changes to the old six weighted loads gives modeled score
**1.1006272179140026**, a **−10.7816%** change. The experiment fails the
predeclared +3% target-only score and no-target-regression screens.

The evaluator code, image and complete file-hash dictionary match selected
check5090b83f1d63407b. Both report H200, driver580.159.03, SM1980MHz,
memory3201MHz and power limit700W. Nodes, fresh inputs/seeds, temperatures
and instantaneous draw differ. This is a descriptive historical comparison
with reference normalization and stable controls, not statistical proof from
a paired candidate-versus-selected run. No promotion is inferred from a point.

## Profile localization, not root-cause proof

| Path | Selected producer → new, µs | Selected merge → new, µs |
|---|---:|---:|
| B1 sliding |7.8 →9.2|1.7 →1.7|
| B16 sliding |9.0 →11.7|1.8 →1.8|
| B32 sliding |14.8 →20.3|2.2 →1.9|
| B32 full |297.2 →528.6|2.7 →2.8|

These rounded **instrumented activity** durations associate the regression
with the producer; they are separate from calibrated whole-call timing. They
do not establish CTA residency, memory throughput, tensor utilization, stalls
or a particular causal instruction. Offline lower register counts, extra
BF16 shared staging/transposition, narrower4-byte async copies and changed
wait/layout work remain possible tradeoffs, not measured bottlenecks.

All four selected target paths now have official native oracle/timing evidence,
including S9/S5 that were not compiled by the earlier offline stop. Their exact
judge opcodes/resources/binary identity are not exposed here. Frozen code emits
no metadata marker; FA3 resolved PackGQA/splits/internal dtype remain UNKNOWN.
This rejects this four-warp BF16/page1 integration, not every transposed/XQA/
WGMMA architecture. Targeted metadata mutation, serving concurrency and unseen
native edge cases are still not certified by this one official result.

## Admission, closure and evidence

Exactly one `check` POST was accepted at02:18:19UTC on2026-10-27. It queued
before being observed running at02:30:19 and terminal at02:31:01. Queue and
job wall time are not kernel latency. Quota went20→19; ranked usage remains0.
The final account query at02:37:35 reports this job done and no active jobs.
The read-only wait client exited. No Modal app, credential action, formal entry
or paid resource was used; monetary cost is$0.

Raw `official-result.json`, all three profile artifacts, quota/closure receipts,
`analysis.json`, the 14-test analyzer and independent reviews are retained.
The profile and raw twelve-shape records corroborate each other. The official
service content hash is retained separately from the exact local source SHA.
Keep working selected1.233633 and the original rollback. No further check of
this unchanged candidate or post-result knob sweep is scheduled.

Before another candidate, use the published failed-experiment index and require
an actual-best producer/dataflow comparison explaining a missing capability.
Resource counts or a win against an inferior prototype are insufficient.
