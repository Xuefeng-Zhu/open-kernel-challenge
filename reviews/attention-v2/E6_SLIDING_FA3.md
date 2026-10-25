# E6 — one-request verify sliding FA3 dispatch

Parent: adopted `mixed_fa3.py`, source commit
`51d7fca04ccc38a972039ef1b086011d8e456264`, SHA256
`708d3f06f46bfb24a7a43963022eb51426f3a68bb2155c77dea8e9b6ae255423`.
Official parent check `e0b3cd9474084069`: 0.8919466283×.
Standalone candidate: `mixed_fa3_sliding.py`. Preserve parent and projection files.

Hypothesis: B=1, q_len=8, window=2048, rope_theta=500000 can reduce its
reference-normalized latency by at least 20% using the official FA3 backend.
Only this dispatch changes; the other eleven scored paths retain identical
functions. No third-party competitor source is copied or executed.

The sliding adapter follows the official baseline, including aspects absent
from the parent's full-attention adapter:

- FusedQKNorm receives supplied positions and an interleaved cos/sin table.
  The pinned implementation rounds normalized Q/K to BF16 **before** RoPE,
  then stores BF16 RoPE outputs. Q is then rounded to FP8 E4M3 for FA3; query
  factor is folded into softmax scale. These permitted rounding differences
  from E2 are explicit parts of the official backend change.
- The table is config-only float32 [2^21, 64], approximately 512 MiB at d=64,
  cached by theta/dimension/device. No input-derived tensor/output is cached.
  It follows the official baseline's table bound and initialization model.
- FA3 is causal, with window=(2047, 0); supplied page table, cache lengths,
  query offsets, scheduler metadata and positions are used. Page64 and scales
  are respected. New K/V are scaled and stored as FP8 bytes, including live
  row zero (`reserved_skip_index=-1`).
- FA3 output is BF16 before per-head RMSNorm and sigmoid gate. Other precision,
  full-verify FA3 paths and E2 fallback remain unchanged.

Predeclared model, freezing other paths and parent's reference: target parent
43.976461µs vs SGLang 22.896000µs. A 20% target latency cut predicts score
0.9119521× (+2.24%); matching SGLang predicts 0.9462766× (+6.09%). These are
estimates, not results. Reject the declared goal if correctness/cache/replay
fails, target normalized latency cut is below20%, or another path regresses
over5%. Separate-run shifts require comparison to reference-normalized controls;
overlapping aggregate CIs do not establish a global gain.

## Local verification before the one official check

```
.venv/bin/python reviews/attention-v2/test_mixed_fa3_sliding.py
.venv/bin/python reviews/attention-v2/test_mixed_fa3.py
```

Five new and four parent CPU tests passed on Darwin arm64, Python3.12.13,
PyTorch2.14.1, CUDA unavailable. Tests use independent CPU stand-ins for native
operations; they do not execute native SGLang, Triton, FA3 or CUDA graphs.
`E6-cpu-preflight.json` records commands, output, hashes and exact coverage.
Current official TASK/AGENTS/reference/checks/baseline/template bytes match
the saved contract. Pinned official FusedQKNorm source was inspected read-only
for BF16 rounding before RoPE; existing Apache-2.0 attribution/text is retained.

CPU tests cover 700 nearby dispatch configurations, parent AST preservation,
table layout/cache, window edges at ctx0/1/2047/2048/2049, page1/64, unit/nonunit
scales, tp8/dp heads, live row0, new cache oracle, untouched rows and same-address
A→B→A with changed actual lengths/content. Only the CPU table extent is reduced
to8192 to avoid allocating512MiB; the candidate's 2^21 bound is asserted.
Native GPU table initialization, target page64/nonunit/explicit ABA coverage
remain unverified unless the official receipt supplies that evidence. The
unchanged E2 fallback retains its simultaneous-call ticket limitation.

Existing OKC authentication verified Xuefeng-Zhu; daily attention quota19/20
checks available,0 scored used. User authorization has exactly one check left.
No Modal, paid compute, credential changes, push or scored entry.

Prospective command, once exact source is committed:

```
.venv/bin/python reviews/attention-v2/check_cycle.py send E6
.venv/bin/python reviews/attention-v2/check_cycle.py poll E6
```

The coordinator reserves the request before dispatch; any uncertainty remains
counted and must be reconciled without repeating POST. After acceptance the
user attention budget is zero; projection's seven checks remain untouched.
Actual acceptance/results will be recorded separately after this source commit.
