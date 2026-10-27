# Final bounded B32/L8 locality diagnostic

One question: does scattered physical page1 placement cause a substantial
whole-call cost in the actual selected full B32/L8 FA3 route? The reference
uses random physical pages; the retained eager counters justify this question
but do not identify a graph bottleneck or HBM saturation. Prior repack and
slower-custom-control results did not isolate this contrast.

Only B32/L8/full/context80000/TP8/D64/page1, agentic lengths, coherent data,
unit scales. Seeds **135831 and 135832** are fixed before execution. Selected
source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

A is the original random mapping. B is a full-capacity opaque FP8 K/V-row
bijection giving contiguous per-request physical rows, with corresponding
page-table/cache_loc permutation. Preserve all logical history, causal query
ranges, new-KV ownership, unused rows, shapes, strides, capacities, scales,
QKV and FA3 arguments. Logical bytes are fixed; physical transactions and
translation/cache behavior may change. This is diagnostic input preparation,
not an eligible candidate or judge-input modification.

Use one captured complete `kernel_fn` graph and fixed input/output addresses
for both layouts and both seeds. Restores, permutation, validation and 256MiB
L2 eviction stay outside event intervals. Each fixture has four warmup rounds
and **six complete reverse-order pairs** (24 measured graph replays). Original
oracles, unchanged-input/write ownership, full post-call inverse-cache bytes
and exact live-address checks are mandatory. One separate activity session
checks both layouts on the second seed using graph-launch correlation plus
matching GPU annotation/external ID, not CPU/GPU timestamp containment.

**KEEP for research prioritization only:** each fixture has at least 15%
balanced paired whole-call reduction and all six pair ratios favor B.
Otherwise **DROP** this as the next dominant route. No candidate follows in
this task. Failed logical/oracle/ownership checks or unknown/different native
graph identity give an inconclusive technical result and stop attribution.
No automatic rerun, parameter sweep or additional ranked/official entry.

Diagnostic score gain is zero. If a later legal implementation net-recovers
15–30% on this path with all other shapes unchanged, its full fraction
0.8280845119160929 gives a conditional overall score factor
`(1 - 0.8280845119160929 * reduction)^(-1/6)`, or **+2.235% to +4.875%**.
Copy/preparation costs must be included in any later real candidate; a fast
contiguous diagnostic arm does not demonstrate recoverable net savings.
Even the impossible zero-time limit for this full path alone gives only a
1.341058 score factor (1.65437 from the saved 1.233633 point). This is a
fixed-other-path coverage ceiling, not a hardware bound or a forecast.

One existing-image H200 app, free credits only, paid limit $0, fresh guardrails,
bounded worker/child/function/watchdog, no retries, explicit stop and independent
stopped/tasks0 verification. Eager/graph state equivalence is not assumed: this
experiment compares actual graph A/B directly. No counter collection is needed.
