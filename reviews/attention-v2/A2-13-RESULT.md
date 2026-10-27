# A2-13 actual official result

Job `c2fb7b23433f4888`, source `21fefb946524a25d8a06562dd28b631d49cd5175`, SHA256 `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Parent `cfa8a8670e394f80`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.2109908035×, CI [1.1928259414317173, 1.2291556655362366], vs parent 1.1876787561×, relative +1.9628%.
Parent/new intervals overlap: True; declared target goal met: False.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 11.821 | 10.912 | 20.352 | +5.584% |
| decode 1/full | 27.297 | 27.183 | 31.264 | +0.214% |
| decode 16/sliding | 15.157 | 14.254 | 26.672 | +3.533% |
| decode 16/full | 200.575 | 200.209 | 199.936 | -0.145% |
| decode 32/sliding | 21.036 | 20.271 | 30.992 | +2.243% |
| decode 32/full | 362.877 | 316.412 | 384.832 | +12.671% |
| verify 1/sliding | 16.466 | 17.141 | 22.656 | -4.396% |
| verify 1/full | 29.886 | 29.802 | 35.952 | -0.697% |
| verify 16/sliding | 22.562 | 22.577 | 29.008 | -0.947% |
| verify 16/full | 205.992 | 204.454 | 204.208 | -0.015% |
| verify 32/sliding | 27.454 | 27.913 | 34.336 | -1.482% |
| verify 32/full | 389.601 | 389.167 | 388.992 | -0.053% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 936.883 | 1436.416 | 1.533187 |
| decode 16 | 3859.007 | 4425.888 | 1.146898 |
| decode 32 | 5995.049 | 7582.944 | 1.264868 |
| verify 1 | 1265.322 | 1617.408 | 1.278258 |
| verify 16 | 4309.796 | 4601.696 | 1.067729 |
| verify 32 | 7510.684 | 7803.328 | 1.038964 |

Target-only arithmetic using observed normalized change: 1.2190351939×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
Historical user count ceiling: removed; daily platform limits still apply. Current payload/egress scope is governed by CURRENT_AUTHORIZATION.md; no ranked entry or Modal.

## Decision: retain promising FP8 candidate, keep A2-11 selected

All four intended paths improved in this receipt. Full decode B32 is the
strongest result: adjusted whole-call time362.877→316.412µs, a12.671%
normalized reduction against each run's own SGLang reference. The other target
cuts are5.584%,3.533%,2.243%. Target geometric-mean latency cut is6.096%,
below the predeclared10% goal. No path regressed more than5%; the largest
unchanged-control regression is verify1/sliding at4.396%. That source path is
identical, so this comparison cannot attribute its change to the FP8 edit.

The new measured point score is1.2109908035x versus the selected parent's
1.1876787561x (+1.9628%). Their reported95% intervals overlap. Treat the
aggregate gain as promising, not statistically resolved or repeated. Retain
this correctness-passing candidate and full evidence, while A2-11 remains the
selected version pending a separately authorized repeat/selection decision.
The frozen10%-per-target model predicted1.2239352772x, not achieved.

Six weighted loads, not twelve unweighted ratios, reproduce the receipt score.
Raw times are CUDAgraph medians; adjusted times subtract measured graph cost
and use the paired ratios per the official harness. Candidate versus parent
is a comparison across jobs with different fresh data, not a paired A/B run.

## Precision and coverage

Prerequisite output relative-L2 errors range3.935–9.903%; maximum reported
worst-token-row error is12.33%. All case bounds passed (11.1–15%). Several
parent fallback cases had about0.1% error; this legal E4M3 change uses more of
the allowed error margin. New-cache checks passed, but successful cache error
magnitudes are not emitted. Both k/v scale1/64 and page64 occur in prerequisites.

The public checker compares BF16 output against FP32 arithmetic with BF16
queries and BF16 final rounding, and derives adaptive bounds from its E4M3
query reference. Output row means a token's concatenated heads. It collects
output/new cache slots, not all cache rows. Explicit same-address A/B/A,
untouched rows, input immutability and overlapping independent graphs have
not been run for these new bytes; the parent's A13 passes cannot be inherited.

The evaluator image and code hash match the parent. The receipt identifies
NVIDIA H200 and driver580.159.03; it does not emit exact Torch/Triton package
versions or the candidate's PTX/SASS/register/spill/shared-memory data. Public
Triton3.8 source informed the static review, but is not proof of this worker's
compiler version. Native compilation succeeded; native FP8 instruction choice
and why the latency changed remain unverified.

The fullD32 profile attributes311.1µs to_fused and2.6µs to_merge. These
instrumented activities locate the cost but are neither calibrated whole-call
timings nor hardware-counter evidence. Do not claim a proven compute/memory
bottleneck or speculate that every path uses native FP8 MMA.

## Commands, quota and scope

Source commit21fefb946524a25d8a06562dd28b631d49cd5175 preceded dispatch.
Fifteen CPU/source tests passed; exact commands/environment are in
A2-13-cpu-preflight.json. Native command (already consumed, do not repeat):
`bin/okc check reviews/attention-v2/mixed_fa3_fp8_qk.py --no-wait --json`.
Read existing result: `bin/okc status c2fb7b23433f4888 --json`.
Offline metrics: `.venv/bin/python -B reviews/attention-v2/assess_cycle2.py A2-13`
(recomputes metrics; this decision/coverage commentary is manually reviewed).

Accepted06:51:04UTC, observed terminal06:56:24UTC on2026-10-25. One approved
check used, zero uncertain requests. Attention daily quota15/20used,5left;
projection unchanged3/20used,17left; ranked0/10both. Reset2026-10-26
00:00UTC. The current exact-payload check permission is consumed; no new
check follows from remaining platform quota. No Modal, payment, credential
change, ranked entry or extra GPU job. Prior apps remain stopped by saved
verified receipts; no new Modal status/launch was attempted here.

Current code/source/tests/results push is explicitly approved for the existing
private repository only. Review outgoing history and verify remote SHA after
push; prior denials are resolved for this named payload, not future payloads.

Receipt retains60timing sample entries per shape except fullverifyB32with56,
including candidate/reference times and ratios; no post-hoc sample deletion.
These are recorded sample counts, not an inferred count of underlying replays.
Final read-only snapshot06:59:09UTC confirms no active jobs among36returned.
