# A2-01 actual official result

Job `6e761bdd1d9747ad`, source `030924103fe180b6645122ec3b02b2ed2af10b59`, SHA256 `2c7bd6a0456c18555d0579b046e741086e527e2290dc3a11b642f8fb37eea357`.
Parent `4f626ab7437a42be`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.0002799616×, CI [0.9852757622170714, 1.0152841610663221], vs parent 0.9395909196×, relative +6.4591%.
Parent/new intervals overlap: False; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.125 | 14.093 | 21.072 | +3.939% |
| decode 1/full | 41.293 | 41.381 | 31.696 | +2.214% |
| decode 16/sliding | 15.807 | 15.685 | 26.800 | -0.475% |
| decode 16/full | 257.617 | 256.310 | 201.296 | +1.005% |
| decode 32/sliding | 19.730 | 20.001 | 31.280 | -0.235% |
| decode 32/full | 366.271 | 368.851 | 385.312 | -0.085% |
| verify 1/sliding | 22.542 | 23.465 | 23.088 | -0.416% |
| verify 1/full | 76.348 | 36.173 | 36.176 | +53.605% |
| verify 16/sliding | 36.864 | 37.247 | 29.328 | +0.117% |
| verify 16/full | 203.985 | 204.039 | 203.968 | -0.183% |
| verify 32/sliding | 45.130 | 44.883 | 34.416 | -0.517% |
| verify 32/full | 387.983 | 390.498 | 390.400 | -0.042% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1310.378 | 1476.448 | 1.126734 |
| decode 16 | 4822.475 | 4453.536 | 0.923496 |
| decode 32 | 6821.660 | 7603.872 | 1.114666 |
| verify 1 | 1658.185 | 1640.864 | 0.989554 |
| verify 16 | 4977.983 | 4612.576 | 0.926595 |
| verify 32 | 8312.614 | 7829.536 | 0.941886 |

Target-only arithmetic using observed normalized change: 0.9947694129×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 9; no push, ranked entry or Modal in this experiment.
