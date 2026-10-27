# DROP physical placement as the next major L8 route

The single bounded diagnostic completed successfully; the predeclared
**15% per-fixture** prioritization gate failed on both fresh fixtures.

| Seed | Random median | Contiguous median | Balanced paired reduction | Favorable pairs |
|---|---:|---:|---:|---:|
|135831|391.632 us|372.544 us|4.8148%|6/6|
|135832|394.336 us|375.808 us|4.7931%|6/6|

These are uninstrumented whole `kernel_fn` graph timings on the actual selected
B32/L8/full/80K/TP8/D64/page1 route. One captured graph and fixed addresses were
used across layouts and seeds, with cold L2 and reverse-order pairs. Logical
FP8 bytes, causal ranges, scales, QKV and FA3 arguments were unchanged.

All **4 original oracles and 70 checked graph replays passed**. The full
post-call caches are byte-exact after inverse permutation; output bytes match
across layouts on both seeds. Actual graph-correlated full symbols and native
node/launch metadata are observed equal. No eager-to-graph counter transfer is
needed for this A/B conclusion.

Physical placement has a small consistent effect here. It is not the large
recoverable cost needed to prioritize this route under the declared gate.
This does not rule out page-table/address-instruction dependencies, distinguish
TLB/cache latency from transaction amplification, or establish another hardware
bottleneck. The earlier L1 custom-producer failures are not evidence against
this L8 FA3 result.

**Scoring:** this diagnostic earns zero contest gain. If a later legal change
recovered all the measured placement difference at zero extra cost, keeping
every other path fixed, the historical six-load model gives only about
**+0.68% overall**. This is an optimistic scenario, not a measured score or
universal algorithm bound. Even setting this entire full path's time to zero
would yield at most a 1.341058 factor in that fixed-other-path model (1.65437
from the saved 1.233633 point). No contiguous-input result is submitted.

Terminal decision: **DROP as the next major optimization branch; no further
candidate implementation or additional experiment follows in this task.**
Selected source remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

Executed source commit: `19313efb186ba91a48fe6b9cd6565e81382f27b0`.
App `ap-HPcxoqismBdiogLGQ1HejU` is independently confirmed stopped/tasks 0 at
07:05:15 UTC; all listed apps are stopped/tasks 0. Raw results and the complete
activity trace are retained. No official check, ranked entry, kernel/math,
credential or security change occurred. Existing accounting is in `COSTS.json`;
closure is recorded in `independent-cleanup.json`.
