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

Result: app `ap-dMJsl67NMlcBiZs5KJWUFv` completed and verified stopped/tasks0.
Candidate/R10/SGLang each39/39native CUDA checks pass;30graph shapes x12fresh
repetitions per runner pass. Same H200/Torch2.14.1+cu130/Triton3.8.0/public SGLang
base as initial smoke. Setup2.281s, worker47.775s, app lifetime56s. No timeout.

Unpooler m1 gross median8.368us vs R10 11.552us vs SGLang11.744us. Candidate
range8.128–9.728us, p908.573us, MAD0.176us (12samples). R10 range11.424–12.352us.
Six opposite-order paired ratios vs R10:1.39442/1.37214/1.40825/1.37965/1.32040/
1.42301; median1.38704. All retained, no outlier removal, overhead subtraction,
CI or official score. m8 remains the original path (paired R10 ratio1.00057).

Latest cumulative billing$0.13metered/credits-$0.13/billed$0, detailed cost
$0.12837141. Conservative stage bound$0.218105626 including historical initial
smoke high-water and whole-app-time cost margin. No parallel app, new credentials,
paid activation, cap change or official check. Raw evidence/distributions/each
seed and pairing: results/remote-modal-M19-unpooler.json/.csv. Unchanged paths and
nearby official prerequisite shapes preserved; not proof of a whole-workload win.
Updated harness: all72local unit tests pass, including opposite-order pairing,
outlier retention and duplicate-seed rejection. Syntax checks pass; no candidate
algorithm changes. The previous initial-smoke one-run guard is left unchanged;
this is a separately scoped continuation ledger, not a retry of that launch.
