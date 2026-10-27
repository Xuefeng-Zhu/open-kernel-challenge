# A2-11 actual official result

Job `cfa8a8670e394f80`, source `7e302d2366b194c32c433f6d6f9cc7b0b59b3928`, SHA256 `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`.
Parent `08235841c4044048`; same evaluator: True.
Passed 16 prerequisites and 12 timed shapes.
Score 1.1876787561×, CI [1.1698635747455417, 1.2109342502243532], vs parent 1.1405486256×, relative +4.1322%.
Parent/new intervals overlap: False; declared target goal met: True.
Unchanged controls with >5% normalized latency regression: 0.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized latency cut |
|---|---:|---:|---:|---:|
| decode 1/sliding | 16.056 | 11.821 | 20.816 | +25.868% |
| decode 1/full | 27.496 | 27.297 | 31.328 | -0.087% |
| decode 16/sliding | 18.178 | 15.157 | 27.360 | +16.571% |
| decode 16/full | 200.827 | 200.575 | 200.592 | -0.193% |
| decode 32/sliding | 22.322 | 21.036 | 31.440 | +5.712% |
| decode 32/full | 368.606 | 362.877 | 385.424 | +1.922% |
| verify 1/sliding | 16.552 | 16.466 | 22.720 | +1.499% |
| verify 1/full | 29.910 | 29.886 | 36.304 | +0.213% |
| verify 16/sliding | 22.357 | 22.562 | 29.264 | -0.530% |
| verify 16/full | 205.128 | 205.992 | 205.776 | -0.117% |
| verify 32/sliding | 27.745 | 27.454 | 34.272 | -0.199% |
| verify 32/full | 389.872 | 389.601 | 389.632 | +0.049% |

Six loads: C=46×sliding+16×full, same for S; score=geometric mean(S/C).

| Load | Weighted C µs | Weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 980.516 | 1458.784 | 1.487771 |
| decode 16 | 3906.412 | 4468.032 | 1.143769 |
| decode 32 | 6773.685 | 7613.024 | 1.123912 |
| verify 1 | 1235.594 | 1625.984 | 1.315953 |
| verify 16 | 4333.741 | 4638.560 | 1.070336 |
| verify 32 | 7496.511 | 7810.624 | 1.041901 |

Target-only arithmetic using observed normalized change: 1.1864712805×; NOT another achieved run.
Raw medians, adjusted times and paired ratios are separate in the CSV/receipt.
Separate jobs are not directly paired parent/candidate trials. No hardware-counter claim.
Native output/new-cache and fresh-input graph evidence is limited to actual receipt cases.
CPU ABA/page/scales/untouched-row tests remain stand-ins unless explicitly confirmed natively.
User attention count ceiling: removed; daily platform limits still apply; no ranked entry or Modal; reviewed private auto-push is authorized.

## Retain the complete candidate

All predeclared per-path goals met, no>5%unchanged-control regressions.
Compared with A2-10, separate-run aggregate improves4.132% and CIs do not overlap.
Compared with counter-local A2-09 it improves3.577%; intervals narrowly do not
overlap. Compared with faster historical A2-08 it improves2.761%, but CIs
slightly overlap: do not call the total gain over A2-08 statistically resolved.
Full per-shape comparisons are saved in A2-11-previous-best-comparisons.json.

Independent profile activities: sliding B1 partial8.1us/merge1.7us; B16
9.6/1.8; B32 15.5/2.0; full B32 359.7/2.7. These are not the calibrated
whole-call medians and do not expose registers/spills/occupancy/counters.
The measured merge activity drop supports parallel-reduction hypothesis.

Current complete candidate has no device-global mutable completion counter.
L1fallback is counter-free; L8fallback retains per-invocation zeroed counters.
Native ordinary graph correctness passed. Dedicated independent-graph/stream
ABA stress remains unrun; source ownership reasoning is not that runtime test.
Same graph/output/cache concurrency and explicitly shared capture pools remain
unsupported. Historical E0/E2 ABA cannot be transferred to this new file.
