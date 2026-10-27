# A2-06 actual official result

Job `05b52e7834064b14`, source `58cf5cf6750900645986e29018606ad18e221f2e`, SHA256 `c40127a320442953630276206b0dfd0ebadd550cc58bdbfe124d800b31b13b18`.
Parent `50465a4c440f4b75`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1381117300×, CI [1.121040054051507, 1.1598261225179742], vs parent 1.1157987383×, relative +1.9997%.
Parent/new intervals overlap: True; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.075 | 14.144 | 20.448 | -0.963% |
| decode 1/full | 31.092 | 30.986 | 30.912 | -0.433% |
| decode 16/sliding | 16.139 | 15.427 | 26.960 | +3.218% |
| decode 16/full | 200.598 | 199.853 | 200.176 | +0.101% |
| decode 32/sliding | 19.959 | 20.019 | 31.328 | +0.724% |
| decode 32/full | 369.571 | 366.377 | 385.808 | +0.696% |
| verify 1/sliding | 16.823 | 16.826 | 22.992 | +0.470% |
| verify 1/full | 36.020 | 35.401 | 35.712 | -0.306% |
| verify 16/sliding | 29.246 | 22.484 | 29.264 | +23.456% |
| verify 16/full | 204.768 | 204.647 | 204.944 | -0.323% |
| verify 32/sliding | 34.575 | 27.861 | 34.560 | +19.494% |
| verify 32/full | 389.404 | 389.547 | 390.208 | -0.016% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1146.401 | 1435.200 | 1.251918 |
| decode 16 | 3907.296 | 4442.976 | 1.137097 |
| decode 32 | 6782.881 | 7614.016 | 1.122534 |
| verify 1 | 1340.388 | 1629.024 | 1.215338 |
| verify 16 | 4308.619 | 4625.248 | 1.073487 |
| verify 32 | 7514.341 | 7833.088 | 1.042419 |

Target-only arithmetic using observed normalized change: 1.1366654952×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 4; no ranked entry or Modal; reviewed private auto-push is authorized.

## Retention decision and exact new native coverage

Retain the batch generalization: both target latency goals are exceeded, and
ten timed controls have no>5% normalized regression. The+2.00% aggregate increment
has overlapping intervals and remains inconclusive as a precise total gain.
Target-only frozen-parent arithmetic1.1366655x closely tracks achieved1.1381117x.

Native new-path prerequisites passed: B5mixed lengths L2=.08157/bound.15; B3
scale1/64 L2=.08095/bound.15; B4soft L2=.06752/bound.15. Previously these nearby
batches used E2; now they use allowed FA3 FP8 queries, so their error is larger
than the old BF16-query E2 path. This is explicit official-allowed FP8 conversion,
not bit-exact equivalence or a claim of improved numerical accuracy. All cache
checks and row bounds pass. These three cases are page1. Native sliding page64,
explicit A-B-A same-graph stress and untouched-row/concurrency stress remain
unverified. Batched scored cases do use fresh agentic lengths as provided by
the evaluator; exact bespoke replay sequence is not evidenced.

Separate profiles show fused prologue1.8/2.3us for B16/B32, five calls per path.
No register/spill/shared-memory/counter data are exposed; sums are not walltime.
