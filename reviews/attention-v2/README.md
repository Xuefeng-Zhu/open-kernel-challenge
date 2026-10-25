# Attention experiments

Selected complete file: [mixed_fa3_fused_full_small.py](mixed_fa3_fused_full_small.py).
Official off-board job`cf979895a3f24685`, source`f01799c`, **1.1557671624x**,
CI[1.1384306550,1.1731036699]. All16prerequisites+12timedshapes passed.
See [complete fusion-series evidence](FUSION_SERIES_RESULTS.md) and
[reasoned next-step review](BOUNDED_NEXT_STEPS.md).

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

A2-05..08 target normalized cuts:28.18%,23.46%/19.49%,15.77%,12.54%.
A2-06..08 aggregate increments have overlapping intervals; do not call them
individually significant. All raw samples, calibrated times, paired ratios,
12shape CSVs and six46*sliding+16*full loads remain in each result/assessment.
Each run has11shapes with60timing sample entries and fullverify32 with56,
not a claimed exact number of underlying graph replays. Profile sums are not walltime.

A2-09 (`mixed_fa3_local_tickets.py`) is an isolated per-invocation counter repair;
its official job `3fb683ba982d448c` passed at 1.14665988x. Sliding decode
B1/B16 normalized latency regressed 7.40/7.68%, supporting A2-10
(`mixed_fa3_decode_merge.py`), job `08235841c4044048`, now running.
The measured A2-08 remains the sequential-only performance reference. Five CPU ownership/protocol/AST tests pass,
including 96 modeled interleavings; this is not a native concurrent-stream test.

User count ceilings were removed explicitly; actual daily quotas remain enforced
in the [ledger](attention-cycle2-budget.json). Historical ten-check and six-check
cycles remain auditable records. No ranked entries, paid resources, new credentials
or Modal launches. Reviewed automatic private pushes remain authorized.

```
.venv/bin/python reviews/attention-v2/summarize_fusion_series.py
```

Source-specific CPU commands live in each card/preflight report. They use CPU
models/stubs and do not execute Triton. Native new-path coverage includes sliding
scale1/64 and mixed batches plus fullL8 million-context/page64. L1 full new path
has nativepage1/scale1 with real supplied scheduler. Native bespoke same-address
ABA, untouched-row stress and concurrent calls are not separately verified for
these fused sources; E2 fallback shared-ticket concurrency limitations remain.

Historical E0..E4 [report](CYCLE_REPORT.md), E5/E6 receipts, original draft and
compilation failurebf4b49bd8bb34dae are preserved. `final_attention.py` is historical
E2, not current selection. Old Modal replay coverage does not transfer to new
sources. Modal cost guard remains$1.708917914/$2, prior apps stopped; old A09 report
retrieval401 unresolved, no restart/renewal. Original local research commit IDs
map to private sanitized-export commits via analysis/PUSH_EXPORT_MAP.json.
