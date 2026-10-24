# 013 — single-row BF16 pooler SIMT GEMV

R06's tensor-core split8 pooler passed correctness but regressed m1/8. Its partial
kernel accounts for94–95% of diagnostic time (20.7–21.8µs); changing the final
reduction alone cannot address most of that measured cost. This distinct hypothesis
uses one CTA per output row and FP32 multiplication/reduction directly on original
BF16 operands ONLY for pooler m==1. It avoids the16-row tensor-core padding and the
separate partial-output reduction kernel. All other paths remain R01, including
pooler m>=2. This does not assert a measured occupancy or memory bottleneck.

Source `candidates/pooler_simt_row.py`; SHA256 `f78af55dfd533ec18d776752f93e9754005ec0361c4bd43f437aa072f9a04e30`.
One CTA/output row, power-of-two K block (8192official),4warps. W uses `.cg` hardware
cache policy and current x `.ca` for within-call L1 reuse. No persistent input-derived
state, precomputation of activations or output cache exists. Original BF16 weights
and activations are loaded as-is, FP32 products/sum, final BF16 output. No reduced
weight precision, TF32 arithmetic or approximate multiply is introduced.

```sh
.venv/bin/python check.py --submission candidates/pooler_simt_row.py --device cpu --suite all --output results/router-pooler-simt-cpu.json
.venv/bin/python -m unittest discover -s tests -p test_pooler_candidate.py -v
.venv/bin/python static_check.py --submission candidates/pooler_simt_row.py
.venv/bin/python -m py_compile candidates/pooler_simt_row.py tests/test_pooler_candidate.py
```

Actual local result **39/39CPU cases +3focused SIMT-candidate tests**, static/syntax
pass. Shared projection gates rerun poolerSplit8/QKV as well (9/9combined), covering
m1/2/7/8/9, N37/K131 tails, unchanged router, original weights, full-width probes at
K boundaries including eight fresh one-row calls with exact BF16 outputs. Environment
M2/macOS arm64,Python3.12.13,PyTorch2.14.1,noCUDA,fourthreads full suite.
CPU uses FP32 matmul and final BF16 rounding, not Triton's reduction tree or native
loads/cache behavior; **GPU compilation/capture/correctness/performance unverified**.

Compare actual pooler m1 against R01/R06 within-run SGLang ratios; other pooler shapes
must remain baseline. Main risks are full-width register pressure, reduction/shuffle
cost, CTA scheduling and failure to saturate HBM. A theoretical14µs floor is not a
prediction. Only a free slot and remaining approved budget can admit its own check.
