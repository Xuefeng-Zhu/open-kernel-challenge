# 019 — independent unpooler m1 SIMT diagnostic

Continuation of the user's Modal-first projection work, not an OKC check.
Existing candidate experiment017; SHA256
65ae42bfc6214dc84239f9e6c0155980464f53c77c1f8c42d19f76ed82290ec0.
Only unpooler m1 reuses the unchanged R10 one-row SIMT body. Preserve output
slice N=1024 at scored m1, K=4096. Hypothesis: reduce launch/tactic overhead at
this small GEMV; risk: too few CTAs to utilize bandwidth. No new algorithm edits.

Plan: candidate/R10/SGLang 39 native correctness cases each with seed offset400000,
then all30 scored shapes,12 fresh graph repetitions/runner/shape with all six
orders repeated twice. Retain every sample. Opposite-order pairs use indices
(0,5),(1,3),(2,4) per six-order block; summarize ratios and full distributions,
no official score/CI or fixed-overhead subtraction.

Reuse existing image im-cPTbGpNiX3RarKResPgX7h, from the initial smoke's official
public SGLang base with entrypoint cleared. Only the reviewed17-file bundle and
manifest are mounted; no auth/config/source auto-upload. One H200, max1/min0,
no detached/deployed app, no schedules/volumes/secrets; retries0, internal540s,
wrapper555s,function600s,startup120s,scaledown2. CPU request/limit2, RAM8GiB hard
limit. Stage ceiling$2 credits including initial smoke; reserve$1 before launch
using conservative highest observed billing value. Stop on nonzero net charge,
insufficient reserve or an unresolved prior app. No new credential/cap change.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile modal_gpu_worker.py modal_diagnostics.py modal_analysis.py
.venv/bin/python modal_preflight.py --candidate candidates/projection_unpooler_m1_simt.py --build local-notes/modal-m19-bundle
.venv/bin/python modal_diagnostics.py --experiment M19-unpooler --bundle local-notes/modal-m19-bundle
```

Results pending at source commit. Full CPU contract39+focused3 tests already pass;
GPU and performance remain unverified until this exact accepted app completes.
After it stops, reconcile cost and record results before independent QKV repeat.
Updated harness: all72local unit tests pass, including opposite-order pairing,
outlier retention and duplicate-seed rejection. Syntax checks pass; no candidate
algorithm changes. The previous initial-smoke one-run guard is left unchanged;
this is a separately scoped continuation ledger, not a retry of that launch.
