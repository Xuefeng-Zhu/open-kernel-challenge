# 029 — large-M pooler grouped tensor-core GEMM

Independent of rejected QKV layout and pending unpooler layout. Base is official
best full composition c34c497b. Only pooler m>=128 changes: BM32/BN128/BK64,
group eight M tiles for within-call L2 weight reuse, full original BF16 operands,
FP32 accumulation, one CTA/output tile, no split-K reduction. Masks cover M/N/K.
Source SHA573f2f9e0e196c0c098ca9f9ca5a48e610df5482bc981933b1119d29fda75f3c.

Official m256 pooler profile is one GEMM ~29.2us versus documented floor17.37us.
This motivates testing tile/reuse, not a predicted speedup. Prior R06 small-M
split-K attempt is a distinct failed hypothesis. All other paths preserved.

```sh
.venv/bin/python check.py --submission candidates/projection_pooler_grouped.py --device cpu --suite all --output results/router-cycle2-pooler-grouped-cpu.json
PROJECTION_CANDIDATE=projection_pooler_grouped.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_pooler_grouped.py
.venv/bin/python -m py_compile candidates/projection_pooler_grouped.py
```

39CPU cases and three focused tests pass, static/syntax pass on Mac M2/Python
3.12.13/Torch2.14.1. CPU branch is a mathematical FP32 model; no Triton code was
executed. Native compiler, CUDA correctness, graph replay and performance pending.
Planned nearby graph correctness m127/128/129/255/257 and noncanonical N4093.
No official check unless Modal screening provides meaningful favorable evidence.
