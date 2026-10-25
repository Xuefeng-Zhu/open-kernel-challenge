# A2-04 actual official result

Job `66b44d7586184708`, source `28991e79b6062602705bc25463139eec8c154903`, SHA256 `5ef5c9e93b56ca382aea363d0022d558a8163f5df0ef8ae5673ba1a854bbf97b`.
Parent `bdd11e743365439a`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.0770658900×, CI [1.060909901676615, 1.0932218783774255], vs parent 1.0585245184×, relative +1.7516%.
Parent/new intervals overlap: True; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.203 | 13.772 | 20.720 | +2.890% |
| decode 1/full | 41.447 | 31.670 | 31.264 | +22.338% |
| decode 16/sliding | 15.665 | 15.751 | 26.624 | -1.997% |
| decode 16/full | 201.998 | 201.077 | 200.576 | -0.171% |
| decode 32/sliding | 20.206 | 19.835 | 30.736 | -0.922% |
| decode 32/full | 368.288 | 367.676 | 383.216 | -0.984% |
| verify 1/sliding | 23.367 | 23.390 | 22.848 | -1.920% |
| verify 1/full | 36.525 | 36.447 | 36.176 | -0.802% |
| verify 16/sliding | 29.633 | 28.783 | 28.704 | -0.163% |
| verify 16/full | 206.745 | 205.297 | 204.848 | -0.098% |
| verify 32/sliding | 34.655 | 34.599 | 34.608 | -0.117% |
| verify 32/full | 392.475 | 389.997 | 390.256 | -0.008% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1140.215 | 1453.344 | 1.274622 |
| decode 16 | 3941.780 | 4433.920 | 1.124852 |
| decode 32 | 6795.203 | 7545.312 | 1.110388 |
| verify 1 | 1659.094 | 1629.824 | 0.982358 |
| verify 16 | 4608.762 | 4597.952 | 0.997654 |
| verify 32 | 7831.520 | 7836.064 | 1.000580 |

Target-only arithmetic using observed normalized change: 1.0797943432×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 6; no push, ranked entry or Modal in this experiment.
