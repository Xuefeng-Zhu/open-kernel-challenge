# A2-02 actual official result

Job `3e85261cb8ed4899`, source `790d4256fe311086eb9f899ae88b0915ce0eb5ab`, SHA256 `25b35cc748262774a85c3a8ed584ed02b37dbb429b967d437c9372e7afe9037d`.
Parent `6e761bdd1d9747ad`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.0124269677×, CI [0.997240563175782, 1.0276133722065164], vs parent 1.0002799616×, relative +1.2144%.
Parent/new intervals overlap: True; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.093 | 14.239 | 20.384 | -4.447% |
| decode 1/full | 41.381 | 41.264 | 30.816 | -2.564% |
| decode 16/sliding | 15.685 | 15.711 | 26.912 | +0.251% |
| decode 16/full | 256.310 | 257.392 | 200.384 | -0.879% |
| decode 32/sliding | 20.001 | 20.437 | 31.296 | -2.129% |
| decode 32/full | 368.851 | 371.912 | 388.784 | +0.070% |
| verify 1/sliding | 23.465 | 23.561 | 22.592 | -2.611% |
| verify 1/full | 36.173 | 35.980 | 35.696 | -0.803% |
| verify 16/sliding | 37.247 | 29.372 | 29.360 | +21.227% |
| verify 16/full | 204.039 | 206.352 | 206.240 | -0.020% |
| verify 32/sliding | 44.883 | 34.389 | 34.432 | +23.417% |
| verify 32/full | 390.498 | 390.637 | 390.592 | +0.014% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1315.222 | 1430.720 | 1.087816 |
| decode 16 | 4840.974 | 4444.096 | 0.918017 |
| decode 32 | 6890.709 | 7660.160 | 1.111665 |
| verify 1 | 1659.478 | 1610.368 | 0.970406 |
| verify 16 | 4652.764 | 4650.400 | 0.999492 |
| verify 32 | 7832.096 | 7833.344 | 1.000159 |

Target-only arithmetic using observed normalized change: 1.0231760921×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention checks remaining: 8; no push, ranked entry or Modal in this experiment.
