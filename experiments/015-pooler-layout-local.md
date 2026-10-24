# 015 / cycle2 P2-01 — pooler weight layout, existing BF16 library GEMM

Control R10 `02e099c3724e4a05`; preserve its m1 SIMT pooler and all router/native paths.
Pooler is32.31% of average per-load time. R10 m8/16/32/128/256 takes about
20.84/20.78/21.69/24.78/28.79µs. Actual profiles show cuBLAS kernels, sometimes with
split-K reduction. Change only pooler m>1 to x @ contiguous(W.T), keeping BF16.
A legal weight-layout change may select a better existing library tactic; it does
not reduce compulsory bytes or promise fewer launches. Prepare retains original
weights and adds a contiguous[K,N]BF16 copy (~64MiB). No activation precomputation.

Source `candidates/projection_pooler_layout.py`, SHA256 `b3346d5359602c098735e80dda5b38ab22f069fb19abd777fda5dc988f6a7571`.

```sh
.venv/bin/python check.py --submission candidates/projection_pooler_layout.py --device cpu --suite all --output results/router-cycle2-pooler-layout-cpu.json
PROJECTION_CANDIDATE=projection_pooler_layout.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_pooler_layout.py
.venv/bin/python -m py_compile candidates/projection_pooler_layout.py tests/test_projection_cycle2.py
```

Actual **39/39CPU oracle cases +3focused tests**, syntax/static/JIT-global checks pass.
Focused tests cover fresh inputs, zero/probe rows, N/K tails, m1/2/7/8/9/16/32/128/256,
real8192-wide probes, original-weight preservation, and untouched router/o controls.
M2/macOS arm64,Python3.12.13,PyTorch2.14.1,noCUDA,4threads full suite.
GPU tactic/capture/correctness/performance for this layout remain unverified.
Compare all six actual weighted loads and unchanged paths; CPU timing is not evidence.
Risks: worse library tactic, slice strides, and additional weight storage.
