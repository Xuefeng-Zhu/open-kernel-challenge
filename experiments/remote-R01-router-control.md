# R01 — H200 validation of the stock-style hi+lo control

Hypothesis: replacing only the original starter's m>8 FP32 router with SGLang's
permitted two-BF16-product/FP32-accumulation path removes its measured router deficit.
Source: `candidates/router_split_after8.py`, commit
`4fb8e81fba03a2b9eedbdbdb9ce671d1be4ea4f5`, SHA256
`6b1b66e2875ad4a755e98e41235243c5f0b5131c32f5f6b08e49d2f4f7d5c352`.
Configuration is unchanged from local experiment 002: threshold 8, contiguous [K,N]
hi/lo, two FP32-output products. The original starter remains separately preserved.

```sh
python3 remote_check.py send R01-router-control candidates/router_split_after8.py
python3 remote_check.py poll R01-router-control
python3 summarize_check.py R01-router-control
```

The send was issued **once**, off-board check mode; job **59abe5825690427c** is **done**.
**9/9 prerequisites and 30/30 scored-shape checks pass**, no checker flags/rejections.
Official speedup: **1.0012943013×**, 95% interval **[0.9862748867,1.0163137158]**.
This is **parity with SGLang**, not evidence of a speedup over it. Versus the original
starter's 0.9076063758 score, the reference-normalized ratio is **1.1032252835**;
that is a cross-run comparison, not a paired same-input trial against the old starter.

Router m=16/32/128/256 now takes **15.01/15.06/16.55/17.46 µs**, versus this run's
SGLang **15.07/15.17/16.38/17.50 µs**. m=8 remains **26.25 µs**. This validates the
control and motivates a separately admitted dispatch/low-batch-kernel experiment.
All 30 raw timing records, profile records, exact evaluator/image/hardware details,
weighted loads and official interval are in `results/remote-R01-router-control.json`;
per-shape adjusted times are in its companion CSV. NVIDIA H200 target; actual driver,
clocks and hashes are preserved in the result, not inferred from CPU tests.

Admission consumes **1 of the 10-new-experiment allowance**. Account quota before
this request was 1/20 checks used. The completion snapshot is 3/20 because R02 was
also admitted while R01 waited; do not attribute that two-count account delta to R01.
Scored usage remains 0/10. No paid compute, new grant/agreement or scored entry.
One transient read-only status-format error on the other job was reconciled by
reading its accepted ID; no POST was repeated.

Result commit: the commit containing this completed record and sanitized evidence.
