# 016 — QKV single-token SIMT, local preflight

Control: frozen R10 `router_pooler_hybrid.py`, job `02e099c3724e4a05`.
Hypothesis: reuse its validated pooler single-row SIMT reduction for QKV at m=1.
Only dispatch changes; no new tile or arithmetic body. QKV has N=2304,K=8192,
versus pooler N=4096,K=8192. Fewer CTAs can reduce bandwidth utilization, so this
is a measurement hypothesis, not a predicted speedup. All other paths retain R10.

Source: `candidates/projection_qkv_m1_simt.py`, SHA256
`9cea9ceed1dab548368746e83540f25edff94f472e8701c05526ea9995759d55`.

```sh
.venv/bin/python check.py --submission candidates/projection_qkv_m1_simt.py --device cpu --suite all --output results/router-cycle2-qkv-m1-cpu.json
PROJECTION_CANDIDATE=projection_qkv_m1_simt.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_qkv_m1_simt.py
.venv/bin/python -m py_compile candidates/projection_qkv_m1_simt.py
```

Actual result: 39/39 CPU oracle cases, 3/3 focused fresh-input/probe/tail/control
tests, static and syntax checks passed. macOS26.6.2/M2 arm64, Python3.12.13,
PyTorch2.14.1, no CUDA/Triton; CPU path models FP32 products and reduction.
**GPU compilation, numerical behavior, graph capture and performance unverified.**
No remote request, run ID, Modal upload or measured speedup. Official dispatch is
paused for the user's Modal-first workflow. New work remains local, unpushed.
