# R04 — fused low-batch Triton router, split-K 16

Hypothesis: replace only router m<=8 with paired BF16 hi/lo dots sharing a tiled
split-K pass and FP32 reduction; keep R01's m>8 control unchanged. Source:
`candidates/router_triton_split16.py`, commit `129969fc2ba10836f25664df8f13f0734d8001dc`,
SHA256 `64aeef36775c050908dd5ddcbd9125a10083ea0d6a2bdbaad704b348debb7725`.
Configuration: S=16, N tile 32, K tile 64, M tile 16, 4 warps, 2 stages; FP32 partials.

```sh
python3 remote_check.py send R04-router-triton16 candidates/router_triton_split16.py
python3 remote_check.py poll R04-router-triton16
python3 summarize_check.py R04-router-triton16
```

Sole job **e3bc1481851f4a23**, check mode, terminal **done**. **9/9 prerequisites and
30/30 scored-shape checks pass**, no flags or rejections. This establishes actual
Triton compilation/execution, CUDA graph capture and numerical validity for this run;
the preceding CPU model alone did not establish any of these.

Official speedup **1.0459769292× SGLang**, 95% interval
**[1.0302872753,1.0616665831]**.
Normalized ratio to original starter: **1.1524565683**, a
cross-run comparison. Overall R02/R04 intervals overlap, so their small aggregate
score difference is inconclusive. They improve distinct shape regions.

Router m=1 is **9.20 vs 10.24 µs**, m=8 **9.27 vs 26.00 µs**. Larger-M paths remain
near stock. Diagnostic profiles show partial about 4.0–4.1 µs (79%) and reduction
about 1.1 µs (21%); those warm diagnostic times are not substituted for cold-L2 adjusted
judge times. The known 128 partial CTAs suggest a bounded S=32 parallelism experiment,
not a claim of measured occupancy or a reason for an unbounded parameter sweep.

Full actual timings, profiles, weights/load aggregation, official interval and
hardware/evaluator/image provenance: `results/remote-R04-router-triton16.json` and CSV.
Admission 4 of at most 10 new checks; no duplicate POST, paid resource, scored/final
entry, credential change or new material agreement. The result commit contains this
completed record; pending R05 remains immutable.
