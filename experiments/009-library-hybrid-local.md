# 009 — compose measured winning library router regions (local gate)

Hypothesis: combine R03's verified improvement at m=8 (14.54 vs 26.30 µs) with R02's
verified m>8 wide product (roughly 1.55–1.60× router speedups). This is a transparent
composition of independently validated regions, not a claim that gains simply add.
M=1 retains original FP32, 2<=M<=8 uses two BF16 hi+lo products, and m>8 uses one
wide-N product with FP32 half-sum. Other projections remain unchanged.

Source: `candidates/router_hybrid_library.py`; SHA256
`a53b2957a311d90d26e8d69c02b79ef0973b2a83cc99c26e8daf4ef3ef208f19`.
Controls: R02 job `89fe00c5034a4198`, R03 job `718e0730e1bd4c30`; thresholds are static
shape dispatch, not input-value/seed/test-case detection. Weight work alone is prepared.

```sh
.venv/bin/python check.py --submission candidates/router_hybrid_library.py --device cpu --suite all --output results/router-hybrid-library-cpu.json
ROUTER_CANDIDATE=candidates/router_hybrid_library.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_hybrid_library.py
```

Actual local gates: **39/39 full-size CPU oracle cases, 4 focused tests and static
checks pass**. CPU environment: M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1; full
suite four threads. CPU uses FP32 products of BF16 operands, not CUDA execution.
The two components' separate GPU passes do **not** validate this combined file's
CUDA capture, whole-workload correctness/performance or extra prepared-weight footprint.
No GPU run belongs to this local gate. Preserve it as a simple fallback; choose its
admission only after inspecting R04/R05 rather than spending a check automatically.
