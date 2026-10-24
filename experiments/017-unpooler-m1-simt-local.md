# 017 — unpooler single-token SIMT, local preflight

Control: R10 `router_pooler_hybrid.py`, job `02e099c3724e4a05`.
Reuse its unchanged pooler SIMT body for unpooler m=1, preserving original BF16
weights, the requested output slice and every other path. This is independent of
experiment016; it does not compose the unverified QKV path. At scored m=1 the
unpooler uses K=4096,N=1024. The smaller grid and launch-bound regime may make
SIMT less effective than pooler; test against same-run R10 before accepting it.

Source: `candidates/projection_unpooler_m1_simt.py`, SHA256
`65ae42bfc6214dc84239f9e6c0155980464f53c77c1f8c42d19f76ed82290ec0`.

```sh
.venv/bin/python check.py --submission candidates/projection_unpooler_m1_simt.py --device cpu --suite all --output results/router-cycle2-unpooler-m1-cpu.json
PROJECTION_CANDIDATE=projection_unpooler_m1_simt.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_unpooler_m1_simt.py
.venv/bin/python -m py_compile candidates/projection_unpooler_m1_simt.py
```

Actual: 39/39 CPU oracle cases, 3/3 focused fresh/probe/slice/tail/control tests,
static and syntax checks pass. macOS26.6.2/M2 arm64, Python3.12.13, PyTorch2.14.1,
no CUDA/Triton. CPU path models FP32 products/reduction, not native Triton.
**GPU compilation, correctness, graph capture and performance unverified.**
No remote request or run ID. No speedup claimed. Official dispatch paused for
Modal-first testing; Modal account login/actual credit/cap/auth remain prerequisites.
Local-only commit, no push.
