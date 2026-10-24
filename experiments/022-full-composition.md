# 022 — add isolated BF16 pooler layout to native composition

Source candidates/projection_full_composition.py, SHA256
c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7.
Only changes from021: weight-only BF16 contiguous[K,N] pooler copy in prepare,
then pooler m>1 uses that layout with existing torch GEMM. Keep original weights.
QKV/unpooler m1 and router remain021/R10. All independent sources preserved.

Justification is provisional: P2-01 official cross-run pooler m8/16/32/128/256
normalized ratios vs R10 are1.02754/1.02765/1.04072/1.07911/1.03558. They are actual
per-shape evidence, not proof against run-to-run noise; full-score CIs overlap.
No R06 split8 pooler or other measured regression is included. Same-app023 will
directly compare021 vs022 and R10 vsP2-01, so an unhelpful layout can be rejected.

```sh
.venv/bin/python check.py --submission candidates/projection_full_composition.py --device cpu --suite all --output results/router-cycle2-full-compose-cpu.json
PROJECTION_CANDIDATE=projection_full_composition.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_full_composition.py
.venv/bin/python -m py_compile candidates/projection_full_composition.py
```

Actual39/39CPU oracle +9focused tests pass, syntax/static pass. Focused tests cover
fresh inputs, probes/zero rows, N/K tails, real widths and unchanged router/o paths.
M2/macOS26.6.2, Python3.12.13, Torch2.14.1; no CPU performance claim. Combined GPU
correctness, graph behavior and performance pending experiment023. No OKC/push.
