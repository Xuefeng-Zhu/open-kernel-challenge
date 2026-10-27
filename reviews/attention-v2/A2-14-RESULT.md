# A2-14 actual official result

Job `6d6019056cbc4f81`, source `2092957b1c8312b1fc90b43f64c3ed5eed8f4658`, SHA256 `bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d`.
Parent `c2fb7b23433f4888`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1687086418×, CI [1.1511780121827333, 1.1862392714370296], vs parent 1.2109908035×, relative -3.4915%.
Parent/new intervals overlap: False; declared target goal met: False.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 10.912 | 10.999 | 20.560 | +0.223% |
| decode 1/full | 27.183 | 26.669 | 31.040 | +1.183% |
| decode 16/sliding | 14.254 | 14.417 | 27.024 | +0.169% |
| decode 16/full | 200.209 | 199.943 | 199.888 | +0.109% |
| decode 32/sliding | 20.271 | 19.983 | 31.072 | +1.675% |
| decode 32/full | 316.412 | 422.966 | 384.112 | -33.926% |
| verify 1/sliding | 17.141 | 16.627 | 22.624 | +2.861% |
| verify 1/full | 29.802 | 29.270 | 35.648 | +0.948% |
| verify 16/sliding | 22.577 | 22.740 | 29.184 | -0.118% |
| verify 16/full | 204.454 | 204.985 | 205.056 | +0.155% |
| verify 32/sliding | 27.913 | 27.681 | 34.304 | +0.740% |
| verify 32/full | 389.167 | 388.897 | 389.200 | +0.123% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 932.658 | 1442.400 | 1.546547 |
| decode 16 | 3862.276 | 4441.312 | 1.149921 |
| decode 32 | 7686.663 | 7575.104 | 0.985487 |
| verify 1 | 1233.168 | 1611.072 | 1.306449 |
| verify 16 | 4325.814 | 4623.360 | 1.068784 |
| verify 32 | 7495.662 | 7805.184 | 1.041293 |

Target-only arithmetic using observed normalized change: 1.1611977255×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
Historical user count ceiling: removed; daily platform limits still apply. Current payload/egress scope is governed by CURRENT_AUTHORIZATION.md; no ranked entry or Modal.


## Decision and what this experiment establishes

**Reject A2-14.** The only changed scored path, full decode B32, is 33.9263%
slower after normalizing against each job's same-run SGLang reference. The
predeclared goal was a 20% latency reduction. The result disproves the value
of this particular full-column deferred-denominator implementation at this
shape. It does not prove all deferred or lane-local denominator designs lose.
No additional tuning check is authorized by this exact payload approval.

The aggregate declined 3.4915% against A2-13, and their reported intervals do
not overlap. No unchanged control regressed over 5%; the largest control
regression is verify16/sliding at 0.1179%. Cross-job comparisons are not a
paired parent/candidate trial, but the isolated target's large loss, stable
controls and producer profile support rejecting this implementation. Do not
select a candidate solely on the aggregate point score.

A2-11 remains selected at 1.1876787561×; promising A2-13 remains separately
preserved at 1.2109908035×. Projection P039 remains unchanged at 1.1732389990×.
The preflight 20%-target model was 1.2489104×, not an achieved result. The actual
full-D32 pair ratio is 0.9081395192 versus the parent's 1.2162378047; the goal
would have required at least 1.5202972559. No benchmark sample was discarded.

## Native correctness, timing and diagnostic coverage

The H200 worker compiled and executed this exact submitted source. All 16
prerequisites and 12 timed cases passed. The changed full-D32 timed case
passed, but its detailed numerical error is not emitted in the benchmark
record. Prerequisite output relative-L2 errors span 3.679–9.414%, maximum
reported worst row 12.23%, within adaptive bounds. Those prerequisite paths
bypass the new B32 dispatch, so these numbers do not measure the numerical
change caused by deferred reduction. Successful new-cache checks do not emit
separate error magnitudes. The output oracle uses the candidate's new entries.

The prerequisite suite includes page64 and non-unit scales, but not on the
new B32 branch. Target-specific page64/scales/nearby lengths, same-address
A→B→A, concurrent independent calls/graphs, untouched cache rows and input
immutability were not independently stress-tested for these bytes. Preserve
that distinction; parent A13 diagnostic evidence cannot be inherited.

The official timing protocol uses fresh-input graph replays and cold L2,
with paired/order-swapped reference timings and replay-overhead adjustment.
The full receipt preserves candidate/reference raw samples, ratios and
calibration. There are 60 recorded timing samples for each shape except full
verify B32, which has 58; all benchmark flags are empty. These counts are not
an inference about the number of low-level graph replays.

The evaluator identity matches A2-13. The worker identifies NVIDIA H200 and
NVIDIA driver 580.159.03. Exact Torch/Triton versions, PTX/SASS, registers,
spills and shared-memory allocation were not emitted. The producer profile
moves from 311.1 to 422.0 µs, while merge moves from 2.6 to 2.7 µs. This locates
the regression in producer activity, not its hardware cause. Longer-lived
FP32 state and extra multiplications remain hypotheses; spilling is unproven.
Profile activities are distinct from calibrated whole-call timing.

## Commands, quota, provenance and scope

Source commit `2092957b1c8312b1fc90b43f64c3ed5eed8f4658` preceded dispatch.
Seven local source/CPU tests passed again immediately before dispatch; eleven
scoring-analysis tests passed in the preserved preflight. CPU tests remain
separate from native execution. Independent review reproduced the final score
and target/control calculations and agreed with rejection.

Executed exactly once (do not repeat):
`bin/okc check reviews/attention-v2/mixed_fa3_deferred_denom.py --no-wait --json`.
Read the existing result: `bin/okc status 6d6019056cbc4f81 --json`.
Recompute arithmetic: `.venv/bin/python -B reviews/attention-v2/assess_cycle2.py A2-14`.
That command regenerates the numeric report; this reviewed decision/coverage
appendix is separate editorial evidence and must be retained.

Accepted 2026-10-25 07:44:40 UTC, worker started 07:44:52, service finished
07:50:06, terminal receipt observed 07:50:25. One check consumed, zero uncertain
requests. Read-only snapshot 07:50:53 UTC: attention 16/20 used, 4 remaining;
projection 3/20 used, 17 remaining; scored 0/10 each. Reset is October26
00:00 UTC. No active jobs among 37 returned. Remaining platform capacity is
not permission for another candidate upload under this completed exact grant.

No Modal launch, payment, credential change, ranked entry or extra GPU request
occurred. The first read-only helper incorrectly expected JSON from whoami;
that local parse error was corrected with supported text handling before any
upload. Authentication itself succeeded. No external action was denied or had
an unresolved outcome in this turn.

The original receipt is retained only in ignored local notes. The publishable
receipt removes account/worker identifiers and records the original receipt
hash; all timing/correctness records are unchanged. Reviewed source, tests,
related analysis and sanitized result evidence are approved for the existing
private repository on `attention-fa3-sanitized`. The actual verified push SHA
is reported in the task and ignored push receipt, avoiding a recursive Git
status claim. Never push the original research branch history.
