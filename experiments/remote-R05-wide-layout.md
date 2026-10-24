# R05 — wide-GEMM transpose-view layout on H200

Change only R02's packed weight layout: contiguous [2N,K] hi/lo rows passed as a
[K,2N] transpose view, instead of contiguous [K,2N]. Same threshold 8, GEMM/FP32 half-sum,
original weight precision and all other projections. Source:
`candidates/router_fused_view.py`, commit `b5f6eddf6f7e6ec59692aaefb2f3600d8f4738a4`,
SHA256 `a7ba4be49f33c4eae5eba30a4b16950fcb867091cb662a9ef1ad57448a47c53c`.

```sh
python3 remote_check.py send R05-wide-layout candidates/router_fused_view.py
python3 remote_check.py poll R05-wide-layout
python3 summarize_check.py R05-wide-layout
```

The one accepted job **1631810d19af4af4**, check mode, is **done**. **9/9 prerequisites
and 30/30 scored-shape checks pass** with no checker flags/rejections. Actual CUDA
strided out_dtype behavior and graph capture passed these target cases; CPU was not
used as a GPU proxy.

Official speedup **1.0509508061× SGLang**, 95% interval
**[1.0351865440,1.0667150682]**.
This is the highest observed passing overall score so far, with normalized ratio
**1.1579367820×** over the original starter. That latter
comparison is across separately normalized runs. R05/R02/R04 intervals overlap: do
not claim the small score differences are statistically resolved or that this is a
final leaderboard placement. R02 remains a credible contiguous-layout alternative.

All actual adjusted/raw shape samples, profiles, device/driver and evaluator/image
hashes, load aggregation and confidence interval are in `results/remote-R05-wide-layout.json`
and companion CSV. Admission 5/10; zero scored entries or paid resources. All five
admitted new jobs are now terminal. The distinct planned R06 pooler POST was rejected
before process creation by automatic approval review, not executed or charged to this
budget; no workaround/retry was attempted. See the budget ledger's admission pause.

## Regression qualification

The layout change is **not established as a better router implementation than R02**.
R05's m=128/256 router is **11.66/12.63 µs**, versus R02's **10.06/11.30 µs**;
the corresponding normalized router ratios also favor R02 at those larger batches.
The aggregate point score is only about 0.3% higher and its interval overlaps R02's.
Untouched projection ratios vary across runs; do not attribute all aggregate variation
to the layout change. Keep R02 as the more convincing large-M component for a future
controlled composition with R04's measured low-M path. That composition is not yet GPU-validated.
