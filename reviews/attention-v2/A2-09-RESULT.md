# A2-09 actual official result

Job `3fb683ba982d448c`, source `ef5f09a8ad4fa01aac57845a9021e0a3846c578c`, SHA256 `3b5d5d711a16e309cf512fbde2ff1a06c85b3e6fd52a73712cf66a5ffa10ea12`.
Parent `cf979895a3f24685`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1466598821×, CI [1.1294599838248205, 1.1690678066480464], vs parent 1.1557671624×, relative -0.7880%.
Parent/new intervals overlap: True; declared target goal met: False.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 14.010 | 15.311 | 20.448 | -7.401% |
| decode 1/full | 27.429 | 27.107 | 31.408 | +1.779% |
| decode 16/sliding | 15.707 | 17.063 | 27.216 | -7.675% |
| decode 16/full | 201.119 | 200.284 | 200.400 | +0.129% |
| decode 32/sliding | 20.358 | 21.231 | 31.104 | -4.986% |
| decode 32/full | 368.464 | 370.171 | 384.080 | -0.484% |
| verify 1/sliding | 16.766 | 16.474 | 22.704 | +1.257% |
| verify 1/full | 30.283 | 29.584 | 35.792 | +0.997% |
| verify 16/sliding | 22.909 | 22.662 | 29.184 | -0.009% |
| verify 16/full | 206.088 | 204.913 | 205.120 | +0.143% |
| verify 32/sliding | 27.810 | 27.901 | 34.368 | -0.418% |
| verify 32/full | 390.584 | 388.487 | 388.240 | -0.066% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1137.991 | 1443.136 | 1.268143 |
| decode 16 | 3989.443 | 4458.336 | 1.117533 |
| decode 32 | 6899.369 | 7576.064 | 1.098081 |
| verify 1 | 1231.167 | 1617.056 | 1.313434 |
| verify 16 | 4321.077 | 4624.384 | 1.070192 |
| verify 32 | 7499.237 | 7792.768 | 1.039142 |

Target-only arithmetic using observed normalized change: 1.1427399571×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User count ceiling: removed; platform attention checks remaining: 9; no ranked entry or Modal; reviewed private auto-push is authorized.

## Interpretation and next controlled test

This repairs ownership in source, not a claimed native concurrent-stream pass.
Retain it as the checked invocation-local counter baseline. The faster A2-08
remains the preserved sequential-only performance reference. Its cross-call
counter limitation is explicit. Whole-score loss0.788% is unresolved by overlapping
intervals; however sliding B1/B16 normalized latency rises7.40/7.68%, exceeding
the predefined5%cost flag. B32sliding rises4.99%, fullB32only0.48%.

Next A2-10 can isolate separate final merge only on L1 fallback paths, eliminating
zero initialization and ticket election while preserving E2 partition/precision/
math. E1 did not test sliding paths and used different fullD32partitioning; its
prior full-path losses remain a warning. No assumed register/occupancy gain.
Nearby L8 stays on A2-09's local counter. Full native concurrency/ABA remains a
separate test requirement; the new local harness is not yet run on H200.
