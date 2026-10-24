# 011 — compose independently measured native/wide router paths

Hypothesis: R04 wins the low-M router and R02 wins the larger-M router. Compose
those exact paths with a shape-only m<=8 dispatch; do not change tile/split settings,
packing, arithmetic, other projections or output precision. R05's aggregate point
score is slightly higher but its m128/256 router regresses R02; that is why the
contiguous R02 layout is used. This trial must validate composition and the combined
six-load aggregate, rather than assume independent gains automatically add.

Source: `candidates/router_hybrid_native.py`; SHA256 `d8e29be3b47aa9ad248f7e068edfe24c3432a2c138b6b464cdfe7cddbf58cace`.
Controls: R04 `e3bc1481851f4a23`, R02 `89fe00c5034a4198`.
Low-M: S16, BN32, BK64, BM16, warps4, stages2, BF16 hi+lo with FP32 partials.
Large-M: contiguous [K,2N] BF16 pair, one GEMM producing FP32 then FP32 half-sum.

```sh
.venv/bin/python check.py --submission candidates/router_hybrid_native.py --device cpu --suite all --output results/router-hybrid-native-cpu.json
ROUTER_CANDIDATE=candidates/router_hybrid_native.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_hybrid_native.py
.venv/bin/python -m py_compile candidates/router_hybrid_native.py
```

Actual result: **39/39 CPU oracle cases, 4 focused tests and static/syntax checks pass**.
Environment: macOS arm64 M2, Python3.12.13, PyTorch2.14.1, noCUDA, full suite four threads.
No CPU speed inference. Native compilation/capture/correctness/performance for this
combined file remain unverified until its own H200 result. Prepare adds weight-only
representations; no activation caching or work hoisted from timed calls.
