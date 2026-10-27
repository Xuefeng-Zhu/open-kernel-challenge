# Real baseline report recovered and verified

One H200 app, `ap-iWkGTqu50RkaYoIsbXg8aq`, executed frozen source
`c1208f41841a560a08726a4c95b94e7a44dcf4f3`. It captured the intended B32/L8
full-attention FA3 mainloop and retained a **35,756-byte** real NCU report:

`4104eaf52a0f44bbfc03703af2355902d16cb1f19bebde4816c3b87a58a3b6b6`

The complete 870-character NCU signature, grid `[132,1,1]`, block `[256,1,1]`
and all 15 source/wrapper/binary/fixture hashes match their predeclared values.
All **14 main and 3 child checks passed**, including five original-oracle calls.
Actual installed NCU is **2025.3.1.0**. Its same-path report import exited 0 and
left the report bytes unchanged.

## Frozen failure and offline recovery

The original `result.json` and receipt remain **failed**. Live CSV printed two
decimal places, while the initial import printed six, so the unchanged exact
numeric comparator rejected the different representations. No old gate was
relaxed or rewritten.

A separate **CPU-only** app, `ap-1XKhhpozlSiSFkrtthxHp7`, imported that identical
report twice with supported `--print-units base --print-fp`. Both real outputs
are byte-identical (14,985 bytes, SHA256
`80177532081993b7cbaff5fcc872ae8ce2419aa02a71b889f47999c0c877fdca`). Their
signature, launch, four raw numeric strings and units match the original
capture **exactly**, with no rounding or tolerance applied. This verifies the
retained artifact separately; it does not rewrite the original failed receipt.
`--print-fp` alone is not claimed to guarantee any decimal precision.

| Requested metric | Verified report value |
|---|---:|
| Tensor-pipe activity / sustained peak, elapsed cycles | 11.42% |
| DRAM throughput / sustained peak, elapsed cycles | 36.55% |
| Long-scoreboard stall / active warp | 43.37% |
| Active warps / sustained peak, unit-active cycles | 12.49% |

These denominators differ. They are not additive portions of latency and do
not establish HBM saturation, register causality or a whole-call bottleneck.
Uninstrumented whole-call medians were 382.272, 383.264 and 382.976 us before
activity profiling, after activity profiling and after NCU respectively (three
samples each, cold L2, no empty-graph subtraction). This is baseline evidence,
not an optimization or official-score comparison. Graph ABI and internal
packing/split/scheduler-state equivalence remain unresolved.

## Future executor and resource closure

Only the future import formatter was corrected: worker imports now delegate
to the hash-bound command builder with the two proven options. The executed
source remains recoverable at the immutable `c1208f4` Git commit; all original
raw reports, logs, receipts, manifests and earlier dryrun evidence are retained.
**40 CPU tests** and a new replay of the actual saved native report/canonical
output through the collector, transport and materializer pass. That replay
adds no GPU measurement. See `CANONICAL_ANALYSIS.json`,
`CANONICAL_REPLAY_SUMMARY.json` and the independent reviews.

Independent reads verified H200 stopped/tasks 0 at **06:09:40 UTC**, CPU app
stopped/tasks 0 at **06:21:34 UTC**, and all listed apps stopped/tasks 0. Observed
account gross increments were **$0.08** and **$0.00** respectively; these are
provisional rounded account readings, not final per-app invoices. Actual paid
spending remained **$0**. No further GPU run, candidate/math change, official
check, ranked entry, credential or security change occurred. Selected source
remains `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
