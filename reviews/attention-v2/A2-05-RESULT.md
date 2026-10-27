# A2-05 actual official result

Job `50465a4c440f4b75`, source `535d602b46db74aae38dcd45ddd676b620837180`, SHA256 `a7930cda0389cf55f60759ea3f9247f0562bf3412fad00c81dcbfc48520f93de`.
Parent `66b44d7586184708`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1157987383×, CI [1.0990617572488919, 1.1386331073714417], vs parent 1.0770658900×, relative +3.5961%.
Parent/new intervals overlap: False; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 13.772 | 14.075 | 20.544 | -3.078% |
| decode 1/full | 31.670 | 31.092 | 31.152 | +1.473% |
| decode 16/sliding | 15.751 | 16.139 | 27.296 | +0.061% |
| decode 16/full | 201.077 | 200.598 | 200.720 | +0.310% |
| decode 32/sliding | 19.835 | 19.959 | 31.008 | +0.257% |
| decode 32/full | 367.676 | 369.571 | 386.464 | +0.329% |
| verify 1/sliding | 23.390 | 16.823 | 22.880 | +28.178% |
| verify 1/full | 36.447 | 36.020 | 36.448 | +1.909% |
| verify 16/sliding | 28.783 | 29.246 | 29.136 | -0.101% |
| verify 16/full | 205.297 | 204.768 | 205.728 | +0.684% |
| verify 32/sliding | 34.599 | 34.575 | 34.528 | -0.162% |
| verify 32/full | 389.997 | 389.404 | 390.128 | +0.119% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1144.913 | 1443.456 | 1.260756 |
| decode 16 | 3951.954 | 4467.136 | 1.130361 |
| decode 32 | 6831.239 | 7609.792 | 1.113969 |
| verify 1 | 1350.166 | 1635.648 | 1.211442 |
| verify 16 | 4621.586 | 4631.904 | 1.002233 |
| verify 32 | 7820.919 | 7830.336 | 1.001204 |

Target-only arithmetic using observed normalized change: 1.1139060682×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 5; no ranked entry or Modal; reviewed private auto-push is authorized.

## Decision and coverage limits

Adopt A2-05. One target changes; native H200 check supports the prologue-fusion
hypothesis. The separate instrumented profile has five calls rather than the
parent's nine, including `_fa3_prepare_fused` at1.3us. Its16.1us summed kernels
and14.6us GPUbusy are not the16.823us adjusted median; PDL overlap matters.
No compiled register/spill/shared-memory report or hardware counters are exposed.

The16 prerequisite cases run preserved paths. New fusion executes only the
B1/L8/sliding scored path (page1, scale1, fixed80000context). Its graph/fresh-data
checks passed as provided by the official evaluator. Specific native ABA with
changed actual lengths, page64, nonunit scales, untouched-old-row and concurrency
stress are not separately evidenced for this new kernel. CPU models are distinct.

The15% target-cut scenario predicted1.0955944x; measured cut28.178% exceeded it.
Frozen-parent arithmetic with the observed normalized target cut yields1.1139061x,
versus achieved1.1157987x. Difference includes unchanged control measurements;
do not assign all cross-run variation to source. Source/hash preserved above.
