# 031 — one-variable pooler output tile width

Source candidates/projection_pooler_narrow.py SHA256 8269021351b0fc4ae659c11480a622c6a5fd93655978865bd33bd3e336788310.
Parent source029 (573f2f9e), rejected after M30. Change only BN128 to BN32 in
launch/grid. BM32/BK64/GM8/warps4/stages3/full BF16 operands/FP32acc unchanged.
M128 grid grows128 to512CTAs on132SM; M256256to1024. More parallel CTAs may hide
latency at cost of extra CTA/X traffic. The M30 timings do not identify the cause;
this falsifiable occupancy hypothesis is the last planned basic-tile screen if
still substantially slower. No unmeasured claimed speedup; preserve official best.

```sh
.venv/bin/python check.py --submission candidates/projection_pooler_narrow.py --device cpu --suite all --output results/router-cycle2-pooler-narrow-cpu.json
PROJECTION_CANDIDATE=projection_pooler_narrow.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_pooler_narrow.py
.venv/bin/python -m py_compile candidates/projection_pooler_narrow.py
```

CPU numerical model/full39/focused3 and static/syntax checks are gates, not CUDA
execution. Actual39/39CPU and3/3focused tests pass, static/syntax pass on MacM2/
Python3.12.13/Torch2.14.1. GPU compilation/correctness/performance pending.
