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

## Actual official result — adopt E6

Source commit5202cbea994771eebffa1f479e7e88915106ca14, SHA256
7a3bc5862d117f448879982158554e3a4fd9a80d92c8e6de4c8b91f93acd7755.
Command: `.venv/bin/python reviews/attention-v2/check_cycle.py send E6`.
Accepted once at2026-10-25T02:09:12.833662Z, job4f626ab7437a42be;
monitored that exact ID to done. No duplicate upload or uncertain result.
All16prerequisites and12timed cases passed, no static flags. H200, same
pinned evaluator/image as E5. Full sanitized receipt/raw samples are retained.

Overall0.9395909196×,95%CI[0.9254970558,0.9584871441], +5.3416% versus
E5 .8919466283×,CI[.8785674289,.9053258277]. Intervals do not overlap;
still below SGLang overall. Target43.976→22.542µs, same-receipt SGLang22.272µs;
reference-normalized speedup versus parent1.89766×, latency reduced47.3036%.
The >=20% target hypothesis is supported. Other eleven paths unchanged in source;
normalized control latency regressions max3.90%, none>5%. This does not establish
that small cross-run control shifts were caused by the source change.

Prior20% cut predicted0.9119521×; matching prior SGLang predicted0.9462766×.
Actual0.9395909×. Freezing parent controls but applying the measured target's
normalized change models0.9453938×. That last number is still an arithmetic
scenario, not an additional run: reference/unchanged-control variation also
contributes to the observed total. No raw/adjusted/profile times are mixed.

| Shape | Parent C µs | New C µs | Same-run S µs | Normalized new/parent speedup |
|---|---:|---:|---:|---:|
| decode 1 / sliding | 13.841 | 14.125 | 20.288 | 0.962461 |
| decode 1 / full | 41.167 | 41.293 | 30.928 | 0.977227 |
| decode 16 / sliding | 15.430 | 15.807 | 27.136 | 0.996728 |
| decode 16 / full | 256.005 | 257.617 | 200.288 | 0.993742 |
| decode 32 / sliding | 19.760 | 19.730 | 30.928 | 1.003609 |
| decode 32 / full | 370.063 | 366.271 | 382.944 | 1.003810 |
| verify 1 / sliding | 43.976 | 22.542 | 22.272 | 1.897663 |
| verify 1 / full | 76.491 | 76.348 | 35.424 | 0.984530 |
| verify 16 / sliding | 36.993 | 36.864 | 28.992 | 0.998008 |
| verify 16 / full | 204.304 | 203.985 | 204.288 | 1.001644 |
| verify 32 / sliding | 45.256 | 45.130 | 34.784 | 1.010218 |
| verify 32 / full | 389.661 | 387.983 | 388.048 | 0.999667 |

Six loads use46×sliding+16×full, then geometric mean of S/C (not12ratios).

| Load | New weighted C µs | New weighted S µs | S/C |
|---|---:|---:|---:|
| decode 1 | 1310.439 | 1428.096 | 1.089785 |
| decode 16 | 4848.975 | 4452.864 | 0.918310 |
| decode 32 | 6767.907 | 7549.792 | 1.115528 |
| verify 1 | 2258.516 | 1591.296 | 0.704576 |
| verify 16 | 4959.483 | 4602.240 | 0.927968 |
| verify 32 | 8283.706 | 7808.832 | 0.942674 |

The target's separate instrumented profile confirms native norm+RoPE, cache store,
FA3+combine, casts and gate ran (GPU busy19.5µs in that one sample); this is not
its22.542µs official adjusted median. No compiled register/shared/spill counts
were captured. Native output/new-cache correctness and fresh-input CUDA graph
replays pass on scored page1/scale1. The16prerequisites do not enter this exact
B1/L8/sliding route; explicit target GPU A→B→A with actual length changes,
page64/nonunit scales and untouched-old-row assertions remain unverified.
Their CPU stand-in evidence is recorded separately. No universal concurrency
claim; unchanged E2 ticket limitation persists.

Old attention budget0/6 left (E4 duplicate counted); new independent budget10/10
left. Actual platform attention18/20 checks left,0scored used. Projection7left.
Shared GPU slot terminal, no Modal launched; all earlier Modal stopped receipts
and$1.708917914/$2 conservative guard unchanged. No paid/push/ranked/credential
or terms action. Parent E5 and projection best remain preserved.

Reproduce receipt analysis: `.venv/bin/python reviews/attention-v2/assess_E6.py`.
See E6-per-shape.csv for raw plus adjusted medians, paired ratios and all controls;
E6-assessment.json retains full precision, models, intervals and evaluator identity.
