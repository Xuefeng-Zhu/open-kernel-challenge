# A2-10 actual official result

Job `08235841c4044048`, source `a0e49b48ed456536a5e72db7d5ee5bb767aa1823`, SHA256 `94cfb257da13daddec1e8eeb5e7c0b9abe768f54029ecfe2bca8733f8088b93f`.
Parent `3fb683ba982d448c`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1405486256×, CI [1.1234403962600552, 1.1576568550293969], vs parent 1.1466598821×, relative -0.5330%.
Parent/new intervals overlap: True; declared target goal met: False.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 15.311 | 16.056 | 20.960 | -2.308% |
| decode 1/full | 27.107 | 27.496 | 31.584 | -0.872% |
| decode 16/sliding | 17.063 | 18.178 | 27.376 | -5.909% |
| decode 16/full | 200.284 | 200.827 | 201.232 | +0.143% |
| decode 32/sliding | 21.231 | 22.322 | 31.456 | -3.959% |
| decode 32/full | 370.171 | 368.606 | 383.984 | +0.398% |
| verify 1/sliding | 16.474 | 16.552 | 22.496 | -1.398% |
| verify 1/full | 29.584 | 29.910 | 36.256 | +0.193% |
| verify 16/sliding | 22.662 | 22.357 | 29.152 | +1.237% |
| verify 16/full | 204.913 | 205.128 | 205.152 | -0.089% |
| verify 32/sliding | 27.901 | 27.745 | 34.704 | +1.520% |
| verify 32/full | 388.487 | 389.872 | 389.712 | +0.023% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1178.521 | 1469.504 | 1.246905 |
| decode 16 | 4049.411 | 4479.008 | 1.106089 |
| decode 32 | 6924.487 | 7590.720 | 1.096214 |
| verify 1 | 1239.929 | 1614.912 | 1.302423 |
| verify 16 | 4310.492 | 4623.424 | 1.072598 |
| verify 32 | 7514.241 | 7831.776 | 1.042258 |

Target-only arithmetic using observed normalized change: 1.1413350918×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User count ceiling: removed; platform attention checks remaining: 8; no ranked entry or Modal; reviewed private auto-push is authorized.

## Decision and supported follow-up

Do not promote: all16+12pass but the three sliding targets regress2.31/5.91/3.96%
against A2-09. FullD32improvesonly0.40%, inconclusive. Aggregate-0.53% overlapsCI.
Preserve the isolated failure and A2-09 counter-local baseline.

The independent merge is material in separate instrumented profiles:
slidingB1 partial7.6us/merge6.6us, B16partial9.7/merge4.8,
B32partial15.8/merge3.2, fullB32partial359.3/merge8.0.
These activities are not calibrated whole-call medians or hardware counters.
They justify testing parallel split reduction rather than a generic tile sweep.

A2-11 will keep the exact partial computation and all other paths, change only
L1 separate merge to per-query-head FP32 parallel split reduction. Target>=15%
normalized cut on sliding B1/B16, inspect other affected paths, correctness first.
Float32sumorder changes explicitly; no claim of bitwise equivalence or proven
register reduction. This does not reuse third-party competitor code.
