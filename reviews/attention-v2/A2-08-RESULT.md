# A2-08 actual official result

Job `cf979895a3f24685`, source `f01799cea791006f109f09bd064f953eff6b584b`, SHA256 `34b0c6bc67357aac9b3306e32010f5254bd4df2ae14b07b827b8686902fa60e9`.
Parent `ba9e02c1d6584cc4`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1557671624×, CI [1.138430654999424, 1.1731036698725028], vs parent 1.1505332420×, relative +0.4549%.
Parent/new intervals overlap: True; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.511 | 14.010 | 20.096 | -3.619% |
| decode 1/full | 31.874 | 27.429 | 31.216 | +12.536% |
| decode 16/sliding | 15.802 | 15.707 | 26.976 | +0.484% |
| decode 16/full | 201.746 | 201.119 | 200.976 | +0.025% |
| decode 32/sliding | 20.336 | 20.358 | 31.312 | -2.306% |
| decode 32/full | 366.185 | 368.464 | 384.160 | -1.306% |
| verify 1/sliding | 17.370 | 16.766 | 22.816 | +1.378% |
| verify 1/full | 30.485 | 30.283 | 36.272 | +0.004% |
| verify 16/sliding | 23.265 | 22.909 | 29.504 | +0.516% |
| verify 16/full | 206.031 | 206.088 | 206.000 | -0.074% |
| verify 32/sliding | 28.326 | 27.810 | 34.400 | +0.723% |
| verify 32/full | 391.681 | 390.584 | 390.592 | +0.039% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1083.326 | 1423.872 | 1.314352 |
| decode 16 | 3940.441 | 4456.512 | 1.130968 |
| decode 32 | 6831.895 | 7586.912 | 1.110513 |
| verify 1 | 1255.778 | 1629.888 | 1.297911 |
| verify 16 | 4351.198 | 4653.184 | 1.069403 |
| verify 32 | 7528.624 | 7831.872 | 1.040279 |

Target-only arithmetic using observed normalized change: 1.1612876278×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 2; no ranked entry or Modal; reviewed private auto-push is authorized.

## Retention and bounded stopping decision

Retain the complete candidate for the supported12.536% normalized target cut,
not for the inconclusive+0.4549% aggregate increment. Other11timedpaths have no
>5% normalized regression, but decode1/sliding shifts-3.62% and decode32/sliding
-2.31%; their code is unchanged. These shifts explain why target-only arithmetic
1.1612876x exceeds actual1.1557672x. No source-causal claim for those controls.

Native target uses real supplied FA3 scheduler, page1/scale1. Four instrumented
calls remain (fused producer1.1us, FA3forward/combine, finalgate), versus the
parent's multiple producer calls. Profile24.3us sum/22.8us GPUbusy are not the
27.429us adjusted median. No register/spill/shared-memory evidence is exposed.

All16prerequisites and12timedcases pass for this ACTUAL complete combined file,
including preserved fullL8 million-context/page64. Native L1page64/nonunit/DP,
explicit same-address ABA, untouched-cache and concurrency stress remain absent;
CPU models are separate. E2 fallback simultaneous-call ticket limits remain.

Stop new algorithm dispatch after this bounded series; retain2attentionchecks
including one final reserve. See BOUNDED_NEXT_STEPS.md for exact-path timing and
conditional value calculations. No duplicate check is needed to relabel an
already-tested complete file. Projection5checks remain untouched this resume.
