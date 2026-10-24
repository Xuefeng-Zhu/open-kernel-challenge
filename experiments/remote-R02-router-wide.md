# R02 — one wide-N hi/lo GEMM on H200

Hypothesis: fusing the two router products into one [M,K] x [K,2N] GEMM followed by
an FP32 half-sum reduces launch overhead and changes useful tiling at m>8.
Only that router path changes from R01; m<=8 and all BF16 projections stay baseline.
Source: `candidates/router_fused_wide.py`, commit
`2d7f55d150ea4ca266bbc95fa70c3555c1d75cd9`, SHA256
`016dbc6828bbf85dd6479a119b0b6f4b06cdb13fdc486fa1df16a09139537ccd`.
Configuration: threshold 8, BF16 hi+lo columns [K,2N], one FP32-output product, FP32 add.

```sh
python3 remote_check.py send R02-router-wide candidates/router_fused_wide.py
python3 remote_check.py poll R02-router-wide
python3 summarize_check.py R02-router-wide
```

The sole POST was accepted as **89fe00c5034a4198**, mode check, terminal **done**.
**9/9 prerequisites and 30/30 scored-shape checks pass** with no checker flags.
Official speedup: **1.0476946231× SGLang**, 95% interval
**[1.0319792037,1.0634100424]**.
The interval is above 1: a measured improvement in this off-board trial, not a final
leaderboard placement. Reference-normalized ratio over the original starter is
**1.1543491221**; this cross-run comparison is not paired
same-input timing against the old starter. Judge-normalized displayed time:
**4610.885552 µs**, not a directly measured single-load latency.

The actual per-shape adjusted times, 60 timing samples/shape, profiles, all six load
aggregates, device/driver fields, evaluator/image hashes and interval are preserved in
`results/remote-R02-router-wide.json` and companion CSV. No performance is inferred
from CPU proxies. The m<=8 router remains a separate bottleneck/hypothesis.

Budget admission: **new request 2 of at most 10**. The prior baseline is excluded.
Quota snapshots include other admitted work; each request uses one check, not the
whole observed before/after account delta. Scored usage remains 0. No paid resources,
new grant/agreement, final entry or duplicate POST. The result commit is the commit
containing this record; exact links will be in the final history index.
