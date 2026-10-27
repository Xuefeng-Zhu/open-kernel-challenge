# A2-12 actual official result

Job `b24bbd05419c499a`, source `e2dc35578f1d4aa8aa151fea27a89ed5bd82bbd8`, SHA256 `45003f03d471d484f5451fa35b534ced9819a886a7be8dc6c3bed339d5b45597`.
Parent `cfa8a8670e394f80`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1734299856×, CI [1.1558285358203726, 1.1910314353885056], vs parent 1.1876787561×, relative -1.1997%.
Parent/new intervals overlap: True; declared target goal met: False.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 11.821 | 12.916 | 20.928 | -8.676% |
| decode 1/full | 27.297 | 27.206 | 31.408 | +0.588% |
| decode 16/sliding | 15.157 | 14.903 | 26.656 | -0.925% |
| decode 16/full | 200.575 | 201.371 | 201.872 | +0.240% |
| decode 32/sliding | 21.036 | 21.065 | 31.088 | -1.273% |
| decode 32/full | 362.877 | 363.821 | 384.496 | -0.502% |
| verify 1/sliding | 16.466 | 16.977 | 22.944 | -2.098% |
| verify 1/full | 29.886 | 30.261 | 36.352 | -1.121% |
| verify 16/sliding | 22.562 | 22.445 | 29.120 | +0.027% |
| verify 16/full | 205.992 | 205.321 | 205.488 | +0.186% |
| verify 32/sliding | 27.454 | 27.913 | 34.288 | -1.621% |
| verify 32/full | 389.601 | 390.376 | 390.208 | -0.051% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1029.414 | 1465.216 | 1.423350 |
| decode 16 | 3907.494 | 4456.128 | 1.140406 |
| decode 32 | 6790.125 | 7581.984 | 1.116619 |
| verify 1 | 1265.107 | 1637.056 | 1.294006 |
| verify 16 | 4317.610 | 4627.328 | 1.071734 |
| verify 32 | 7529.999 | 7820.576 | 1.038589 |

Target-only arithmetic using observed normalized change: 1.1784126706×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention count ceiling: removed; daily platform limits still apply; no ranked entry or Modal; reviewed private auto-push is authorized.

## Decision: reject the query hoist, retain A2-11

The target needed at least 15% lower normalized whole-call latency. Instead it
increased 8.676%; no unchanged control regressed more than 5%. The aggregate
point score decreased 1.200%, with overlapping intervals. This is a rejected
performance hypothesis, not a correctness failure.

Instrumented activities: producer 8.1→6.8 µs, new preparation 2.5 µs, merge
1.7→1.6 µs. These separate profile activities support the added-cost explanation;
they are not calibrated whole-call times or hardware-counter measurements.
The frozen-other-shapes 15% target model predicted 1.20499534×; the actual
complete run was 1.17342999×. No tuning sweep or repeat is justified by this result.

Selected source remains `mixed_fa3_decode_parallel_merge.py`, job
`cfa8a8670e394f80`, 1.1876787561×. All four ownership/merge/query experiments
A2-09 through A2-12 are terminal. No new Modal resources or ranked entries.
