# 005 — isolate router weight layout

Hypothesis: retaining BF16 hi/lo in row-major [N,K] storage and passing transpose
views to GEMM may select a different useful access/tactic pattern than experiment
002's contiguous [K,N] copies. This changes only the physical weight layout: same
threshold 8, two products, FP32 accumulation and other-projection math. The official
profile motivates router work but does not establish that its current layout is bad.

Candidate: `candidates/router_transpose_view.py`; SHA256 `7215c300f268694f3f96df10801e07cb55e786b003e4939972a78f107df0dff0`.
Control: `candidates/router_split_after8.py` (experiment 002). Prepare drops only the
two `.contiguous()` copies after transpose. Shapes/values and all original weights
remain unchanged; PyTorch receives the explicit tensor strides.

```sh
.venv/bin/python check.py --submission candidates/router_transpose_view.py --device cpu --suite all --output results/router-transpose-view-cpu.json
ROUTER_CANDIDATE=candidates/router_transpose_view.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_transpose_view.py
```

Actual results: **39/39 CPU oracle cases, 3/3 focused tests and static checks pass**.
Maximum router relative L2: **2.729e-06**, below 1e-4. Coverage includes dispatch
boundaries, fresh activations, outliers, weight-column probes, zero rows, BF16-only
negative control and unchanged original weights/non-router slices. Evidence:
`results/router-transpose-view-cpu.json`.

Environment: Apple M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1; full suite uses four
CPU threads. The CPU model multiplies FP32 representations of the BF16 operands.
**CUDA compilation/execution, strided CUDA out_dtype behavior, graph capture and H200
performance remain unverified.** No GPU run ID, code upload, extra check or paid compute.
This is lower priority than establishing the stock hi+lo control; non-contiguous
operands might be slower or require internal copies. No CPU performance ranking is
claimed. Exact local source/result commit is in the final experiment index.
