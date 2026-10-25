# Attention experiments

Selected complete file: [mixed_fa3_decode_parallel_merge.py](mixed_fa3_decode_parallel_merge.py).
Official off-board job `cfa8a8670e394f80`, source `7e302d2`, **1.1876787561x**,
CI [1.1698635747,1.2109342502]. All16prerequisites+12timedshapes passed.
A2-09 fixes shared counter ownership with initialization cost; A2-10 separate
serial merge regresses; A2-11 parallel merge recovers cost and improves the
previous A2-08 score by2.761%, with slight interval overlap. Complete comparisons
and qualifications are in [A2-11 result](A2-11-RESULT.md).

| Experiment | Isolated change | Official score |
|---|---|---:|
| A2-01 | Full verify B1 to official FA3 |1.00027996x|
| A2-02 | Sliding verify B16/32 to FA3 |1.01242697x|
| A2-03 | Full decode B16 to FA3 |1.05852452x|
| A2-04 | Full decode B1 to FA3 |1.07706589x|
| [A2-05](A2-05-RESULT.md) | Fuse sliding verify B1 prologue |1.11579874x|
| [A2-06](A2-06-RESULT.md) | Same fusion, other sliding verify batches |1.13811173x|
| [A2-07](A2-07-RESULT.md) | Full verify B1 prologue, no RoPE |1.15053324x|
| [A2-08](A2-08-RESULT.md) | Same full fusion, single-token decode |1.15576716x|
| [A2-09](A2-09-RESULT.md) | Per-invocation fallback counter ownership |1.14665988x|
| [A2-10](A2-10-RESULT.md) | Decode fallback independent serial merge |1.14054863x|
| [A2-11](A2-11-RESULT.md) | Per-head parallel split merge |1.18767876x|
| [A2-12](A2-12-RESULT.md) | Sliding decode B1 query hoist, rejected |1.17342999x|
| [A2-13](A2-13-RESULT.md) | FP8 fallback QK, retained provisionally |1.21099080x|

A2-05..08 target normalized cuts:28.18%,23.46%/19.49%,15.77%,12.54%.
A2-06..08 aggregate increments have overlapping intervals; do not call them
individually significant. All raw samples, calibrated times, paired ratios,
12shape CSVs and six46*sliding+16*full loads remain in each result/assessment.
Each run has11shapes with60timing sample entries and fullverify32 with56,
not a claimed exact number of underlying graph replays. Profile sums are not walltime.

A2-09 through A2-12 are all terminal. The [completed cycle report](OWNERSHIP_MERGE_CYCLE.md) records target results, commands, source identities and the exact next validation prerequisite. A2-12 passed correctness but slowed its target by 8.676%; A2-11 remains selected. A2-11 removes completion counters on L1fallback;
L8fallback retains invocation-local zeroed counters. No device-global completion
counter remains. A13 subsequently passed 42 bounded-context native ABA/cache replays. Actual cross-graph overlap was observed for full decode B32 only; six other profiled paths did not overlap. See [A13 result](A13-RESULT.md) and [harness notes](NATIVE_OWNERSHIP_PROBE.md).

A2-13 passed16+12: fullD32cut12.67%, targetgeomean6.10%below10%goal,
aggregate+1.96%CIoverlap. No>5%regression. FP8 consumes more outputtolerance;
parentA13ABA/concurrency evidence does not transfer. A2-11 remains selected.

User count ceilings were removed explicitly; actual daily quotas remain enforced
in the [ledger](attention-cycle2-budget.json). Historical ten-check and six-check
cycles remain auditable records. No ranked entries, paid resources, new credentials
or Modal launches. The renewed exact FP8 payload approval covers this source,
tests and sanitized results push; no unrelated future upload follows from it.

```
.venv/bin/python reviews/attention-v2/summarize_fusion_series.py
```

Source-specific CPU commands live in each card/preflight report. They use CPU
models/stubs and do not execute Triton. Native new-path coverage includes sliding
scale1/64 and mixed batches plus fullL8 million-context/page64. L1 full new path
has nativepage1/scale1 with real supplied scheduler. Native bespoke same-address
ABA, untouched-row stress and concurrent calls are not separately verified for
these fused sources; the old E2 shared-ticket limitation is repaired in A2-11, but bespoke native
concurrency remains unverified.

Historical E0..E4 [report](CYCLE_REPORT.md), E5/E6 receipts, original draft and
compilation failurebf4b49bd8bb34dae are preserved. `final_attention.py` is historical
E2, not current selection. Old Modal replay coverage does not transfer to new
sources. After the separately approved A13 run, conservative Modal accounting is $1.787819802/$2; all apps stopped; old A09 report
retrieval401 unresolved, no restart/renewal. Original local research commit IDs
map to private sanitized-export commits via analysis/PUSH_EXPORT_MAP.json.
