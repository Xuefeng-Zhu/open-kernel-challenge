# A2-03 actual official result

Job `bdd11e743365439a`, source `c94ec7109d7b5525d2c849d297a3c6c28a886d55`, SHA256 `8e1db32fe3d2c803efecbe62bbff5dd752e133cc093743ef28143352a3d28905`.
Parent `3e85261cb8ed4899`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.0585245184×, CI [1.0426466505820096, 1.0744023861327305], vs parent 1.0124269677×, relative +4.5532%.
Parent/new intervals overlap: False; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.239 | 14.203 | 20.752 | +2.020% |
| decode 1/full | 41.264 | 41.447 | 31.776 | +2.592% |
| decode 16/sliding | 15.711 | 15.665 | 27.008 | +0.644% |
| decode 16/full | 257.392 | 201.998 | 201.840 | +22.088% |
| decode 32/sliding | 20.437 | 20.206 | 31.600 | +2.084% |
| decode 32/full | 371.912 | 368.288 | 387.632 | +0.680% |
| verify 1/sliding | 23.561 | 23.367 | 23.264 | +3.687% |
| verify 1/full | 35.980 | 36.525 | 36.544 | +0.841% |
| verify 16/sliding | 29.372 | 29.633 | 29.600 | -0.068% |
| verify 16/full | 206.352 | 206.745 | 206.496 | -0.066% |
| verify 32/sliding | 34.389 | 34.655 | 34.704 | +0.018% |
| verify 32/full | 390.637 | 392.475 | 392.768 | +0.086% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1316.502 | 1463.008 | 1.111284 |
| decode 16 | 3952.571 | 4471.808 | 1.131367 |
| decode 32 | 6822.085 | 7655.712 | 1.122195 |
| verify 1 | 1659.290 | 1654.848 | 0.997323 |
| verify 16 | 4671.035 | 4665.536 | 0.998823 |
| verify 32 | 7873.704 | 7880.672 | 1.000885 |

Target-only arithmetic using observed normalized change: 1.0481632080×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 7; no push, ranked entry or Modal in this experiment.
