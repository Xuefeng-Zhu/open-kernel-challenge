# 008 — change only the verified wide-GEMM weight layout (local gate)

R02 verified the contiguous [K,2N] wide-GEMM router at 1.0477× SGLang. This hypothesis
keeps its math, threshold 8 and FP32 half-sum unchanged, and changes only packing:
concatenate hi/lo rows into contiguous [2N,K], then pass a [K,2N] transpose view.
The resulting strides may select a different useful GEMM access/tactic pattern.
No performance claim follows from the CPU model or the fact that a view avoids a copy.

Source: `candidates/router_fused_view.py`; SHA256
`a7ba4be49f33c4eae5eba30a4b16950fcb867091cb662a9ef1ad57448a47c53c`.
Control source: `candidates/router_fused_wide.py`, verified job `89fe00c5034a4198`.

```sh
.venv/bin/python check.py --submission candidates/router_fused_view.py --device cpu --suite all --output results/router-fused-view-cpu.json
ROUTER_CANDIDATE=candidates/router_fused_view.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_fused_view.py
```

Actual local result: **39/39 full-size CPU cases, 4 focused tests and static checks pass**.
Environment: M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1, four threads for the full
suite. CPU explicitly multiplies FP32 representations of BF16 operands; it does not
validate CUDA strided out_dtype dispatch, graph capture or H200 performance. No remote
job belongs to this local gate. Risk: PyTorch may select a slower tactic or make an
internal copy. The pending R03/R04 sources are not changed or replaced by this file.
