# Identity recapture failed; no new counter result

The one authorized app `ap-Wg106ohTxcewSqroHjDItS` ran frozen commit
`8e6ad398d8f89859e570303e66bbcee4086a9d9c`. NCU exited 0 but explicitly
reported **`No kernels were profiled`**. The exact Torch-demangled filter did
not select a kernel. No `baseline.ncu-rep` was produced, no offline import was
attempted, and completion correctly remained false. This was neither a
permission denial nor a timeout. There are **zero new counter rows**.

The failed assumption was that Torch and NCU use identical demangled spelling.
The graph name has 1380 characters; NCU's available-kernel list has one FA3
family name of 870 characters, using `(int)64`, `(bool)1` and `T1::Params`
where Torch uses `64`, `true` and the expanded parameter type. Full unmodified
strings and hashes are in `analysis.json` and raw `6-stdout.txt`. The available
name is not a profiled launch row. No normalization or identity-gate relaxation
has been applied.

All **14 main checks and 3 child checks passed**, including five original-oracle
calls. Uninstrumented whole-call graph medians were 379.232 us before the Torch
trace, 380.768 us afterward and 380.448 us after NCU. These are unchanged
baseline observations, not paired optimization gains or official timings.

The app was independently verified **stopped/tasks 0 at 05:38:54 UTC**; all
listed apps were stopped/tasks 0. Account gross usage moved $2.66 to $2.75
($0.09 observed delta); actual billed spending remained **$0**. Metering is
provisional and not consistently monotonic across reads; see `COSTS.json`.

Selected source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
No candidate, official check, ranked entry, credential or security change was
made. No further GPU launch is planned in this task. The offline repair plan
and explicit unmet acceptance criteria are in `NEXT_CAPTURE_PLAN.md`.
