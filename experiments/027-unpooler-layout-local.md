# 027 — independent unpooler large-M BF16 layout

Only unpooler m>8 switches to existing torch GEMM with a contiguous[K,N] original-
BF16 weight copy. Adds64MiB weight-only storage; preserve original W and <=8 slice/
m1 native paths. Base remains official best c34c497b; rejected QKV layout absent.
Source candidates/projection_unpooler_layout.py, SHA256
9122d171a15f622dec494cb864cbaebae501791706edecf2a8b4236e131d9a68.

Official profile shows large unpooler still one library GEMM around17.9–25us;
not an eliminated-reduction hypothesis. The K4096/N8192 orientation differs from
pooler's K8192/N4096. A layout may change library tactic/traffic, but pooler's
benefit does not establish a win here; QKV layout's m256 regression is preserved.
Measure every shape including m8/9 boundary and nearby output/tail cases.

```sh
.venv/bin/python check.py --submission candidates/projection_unpooler_layout.py --device cpu --suite all --output results/router-cycle2-unpooler-layout-cpu.json
PROJECTION_CANDIDATE=projection_unpooler_layout.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_unpooler_layout.py
.venv/bin/python -m py_compile candidates/projection_unpooler_layout.py
```

Actual39CPU+3focused tests, static and syntax pass on M2/macOS/Python3.12.13/
Torch2.14.1. Original dtype/weights preserved. No GPU compile/correctness/timing
yet; independent Modal screen follows only within the same$2credit/$0paid ceiling.
