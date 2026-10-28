# Muse Attention

The current optimization control is **pristine vjeux v298**, SHA256 `0cc4a0668acf7553c0d2c6d5c3a2e7d5c85374a70ad774463e9d2fd9e5d0f883`. The original remains immutable; experimental variants have separate directories.

## Baselines and validation

| Version | Role | Verified evidence |
|---|---|---|
| [v298 source](vjeux-v298-probe/candidate.py) | Current same-H200 profiling/optimization control | [2.618804× non-ranked check](vjeux-v298-probe/official-check/RESULTS.md), 16+12 passed · [same-H200 comparison](vjeux-v298-probe/RESULTS.md) · [provenance/license](vjeux-v298-probe/ATTRIBUTION.md) |
| [Fusion + lookahead source](fusion-lookahead-probe/candidate.py), SHA `3da22a16…` | Preserved previous in-house selection | [1.233633× check and selection](fusion-lookahead-probe/RESULTS.md) · [1.242925× ranked receipt](fa3-next-evidence/selected-ranked-submission/RESULTS.md) |
| [A2-11 source](mixed_fa3_decode_parallel_merge.py) | Earlier rollback baseline | [1.187679× check](A2-11-RESULT.md) |

v298's [serial Graph stress](vjeux-v298-probe/graph-stress/RESULTS.md) passes the original oracle across retained independent captures, live input changes and scratch growth. It does **not** establish simultaneous-stream safety or universal scratch allocation lifetime. The old ranked result is from different source; it is not a ranked v298 score.

## Current profiling and decisions

| Evidence | Scope | Result / limitation |
|---|---|---|
| Profile 01 (local evidence; `vjeux-v298-probe/profile-01/RESULTS.md`) | B16/L8 full | Valid baseline NCU report; the original B32 capture was not completed |
| Profile 02 (local evidence; `vjeux-v298-probe/profile-02-l1/RESULTS.md`) | B16/L1 full | Native traffic and aggregate counters; all-state PC samples have no stall-reason attribution |
| Profile 03 (local evidence; `vjeux-v298-probe/profile-03-b32-l8/RESULTS.md`) | B32/L8 full | Reason-linked native page-index dependency evidence; eager capture is distinct from Graph timing |
| Profile 04 (local evidence; `vjeux-v298-probe/profile-04-page-bank-paired/RESULTS.md`) | Matched B32/L8 Graph-node baseline/page-bank | Waits redistributed; no net benefit. Original retained. Per-PC samples are not latency shares |

CTA diagnostics remain in isolated local trial directories. Unpublished evidence is labelled as local rather than linked to absent files; it does not establish a new performance claim or candidate promotion.

[Failed/rejected experiment index](../../FAILED_EXPERIMENTS.md#attention) · [both-track experiment map](../../EXPERIMENTS.md) · [official contract](official/TASK.md).

<details>
<summary>Historical A2-01–A2-13 ledger — original dated selections and limits</summary>

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

</details>
