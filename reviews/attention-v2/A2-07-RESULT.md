# A2-07 actual official result

Job `ba9e02c1d6584cc4`, source `f1b3e8c0d89bea39e0bb92f5f05e2b9a5847520b`, SHA256 `f20126cb295419276c00eb1cb1927c8920a6a96f0c7d3e62f362b10d2176f95d`.
Parent `05b52e7834064b14`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1505332420×, CI [1.1332752434029907, 1.1738497008309647], vs parent 1.1381117300×, relative +1.0914%.
Parent/new intervals overlap: True; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.144 | 14.511 | 21.568 | +2.732% |
| decode 1/full | 30.986 | 31.874 | 31.728 | -0.222% |
| decode 16/sliding | 15.427 | 15.802 | 27.008 | -2.249% |
| decode 16/full | 199.853 | 201.746 | 201.552 | -0.258% |
| decode 32/sliding | 20.019 | 20.336 | 32.000 | +0.546% |
| decode 32/full | 366.377 | 366.185 | 386.768 | +0.301% |
| verify 1/sliding | 16.826 | 17.370 | 23.312 | -1.820% |
| verify 1/full | 35.401 | 30.485 | 36.512 | +15.773% |
| verify 16/sliding | 22.484 | 23.265 | 29.808 | -1.583% |
| verify 16/full | 204.647 | 206.031 | 206.096 | -0.114% |
| verify 32/sliding | 27.861 | 28.326 | 34.784 | -1.014% |
| verify 32/full | 389.547 | 391.681 | 391.536 | -0.207% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1177.503 | 1499.776 | 1.273692 |
| decode 16 | 3954.846 | 4467.200 | 1.129551 |
| decode 32 | 6794.426 | 7660.288 | 1.127437 |
| verify 1 | 1286.785 | 1656.544 | 1.287351 |
| verify 16 | 4366.670 | 4668.704 | 1.069168 |
| verify 32 | 7569.884 | 7864.640 | 1.038938 |

Target-only arithmetic using observed normalized change: 1.1512715640×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 3; no ranked entry or Modal; reviewed private auto-push is authorized.

## Decision and new native coverage

Retain this isolated full-verify prologue: target15.77%normalized improvement
exceeds10%goal, and other11timedpaths show no>5%regression. Total+1.0914% has
overlapping CIs and is not a resolved aggregate increment. Target-only arithmetic
1.1512716x versus achieved1.1505332x shows the scale of unchanged-control drift.

The million-token full B1/L8 prerequisite passed on this new path: L2=.09493,
worstrow=.1150, bound=.15. Its omitted page field uses official default64; the
scored fullverify1 uses page1. No new precision conversion versus parent fullFA3.
Native nonunit-scale/DP/explicitA-B-A/concurrency/untouched-row stress for this
new specialization remains unverified; CPU coverage is reported separately.

Separate profile: fused prologue1.3us, five calls;27.0us sum and25.5us GPUbusy
are not the30.485us adjusted median. No register/spill/shared-memory counters.
