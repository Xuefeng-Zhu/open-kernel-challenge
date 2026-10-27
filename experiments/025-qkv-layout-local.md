# 025 — next isolated hypothesis, QKV m>1 BF16 library layout

Control: officially passing full composition b9b0fd9, job48b1357016134e52.
Only QKV m>1 changes to x @ contiguous(W.T)[:, :n]. Original BF16 weights remain;
prepare adds36MiB (2304*8192*2bytes) weight-only storage. QKV m1, unpooler, pooler
and router preserve the verified composition. No input-derived cache, precision
change or manual algorithm/tile tuning.

Source candidates/projection_qkv_layout.py, SHA256
50a76d4eb203732b7d0dede9dc4797e9f4f3ffa49aae947348a32e956c993e00.

Measured reason: the official full-composition profile shows QKV m8/16/32/128
cuBLAS split-K main kernels plus separate reductions about1.7/1.6/1.9/3.0us,
roughly15–19% of that profile's kernel total. m256 already has one19.0us GEMM.
Layout changed pooler tactics profitably in the same-app M23 controls at m8/16/32/
128, supporting a tractable QKV hypothesis without introducing a new framework.
QKV dimensions differ; this is not evidence of QKV benefit. A selected tactic
could keep split-K, make its main kernel slower or regress m256/nearby shapes.
Profile totals are not the official paired timing and cannot be subtracted to
predict a new score. No unsupported lower precision or cross-call fusion.

```sh
.venv/bin/python check.py --submission candidates/projection_qkv_layout.py --device cpu --suite all --output results/router-cycle2-qkv-layout-cpu.json
PROJECTION_CANDIDATE=projection_qkv_layout.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_qkv_layout.py
.venv/bin/python -m py_compile candidates/projection_qkv_layout.py
```

Actual39/39CPU oracle cases +3focused fresh/probe/tail/control tests pass, static
and syntax pass. macOS26.6.2/M2, Python3.12.13, Torch2.14.1, no CUDA. Unchanged
full composition remains best verified; this candidate is CPU-only. No GPU compile,
capture, performance, Modal or official request; no speedup claimed.

Proposed next measurement after this hypothesis is reported: full CUDA gates plus
the predeclared6graph warmups and balanced steady comparison against full control,
cover every scored and nearby shape, retain tails, inspect actual kernel selection.
Hold all new GPU requests pending the next stage direction. Same$2credit/$0paid
constraints; existing official request is complete, all Modal apps stopped.
