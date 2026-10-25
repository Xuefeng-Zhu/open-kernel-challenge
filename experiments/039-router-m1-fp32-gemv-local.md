# P039: router m1 original-FP32 GEMV reuse, local preflight only

New standalone candidate candidates/projection_router_m1_fp32_gemv.py, reviewed
by the coordinator for a separate local source commit. No GPU/OKC admission or
result is implied. The CPU preflight JSON is a pre-commit snapshot.
Original best and every earlier candidate/experiment remain unchanged.

Parent source candidates/projection_full_composition.py, source commit
b9b0fd9fb0cf326b5836167ffc29d1a0d3386fa7, SHA256
c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7.
Official best48b1357016134e52:1.1486499523461586x,
CI[1.1314202030609661,1.165879701631351]. This candidate is based on that best,
not on rejected P038. Current worktree base75d1dc1; no history rewritten.

## Isolated hypothesis and code change

Router m1 may benefit from a single FP32 SIMT GEMV using original FP32 weights,
replacing its padded16-row hi/lo split32 tensor-core path plus separate reducer.
Reuse the existing _gemv body EXACTLY, next_power_of_2(K),4warps; no new tile/kernel
or parameter sweep. Untimed prepare adds only router_fp32_rows=router.contiguous(),
which preserves every FP32 value and may alias already contiguous original weights.
Other prepared layouts remain unchanged. GPU output is FP32; loaded BF16 x widens
to FP32, original FP32 W stays FP32, multiply/sum/output stay FP32. No TF32,
weight-precision reduction, activation/output caching or extra untimed input work.

Only router m==1 gains a new branch. Native gate requires positive K/N, BF16 x,
FP32 W, same device, sufficient N rows/matching K and dense row-major strides.
K tails use the unchanged GEMV mask, and N prefixes use exactly N output CTAs.
General-strided/incompatible-metadata m1 falls back to timed x.float() @ original
FP32 W[:n].T, because the existing split-K kernel assumes contiguous X strides.
All other dispatch and all existing JIT/helper bodies are AST-identical to parent.
The prepare AST is identical after removal of its single new state assignment.
Complete existing Apache2 LICENSE is embedded as comments for standalone upload.
No new imports or static-check changes.

## Structural evidence and limitations

Actual best adjusted router m1 latency7.46882887us; best profile separately reports
_partial3.2us plus _reduce1.3us. Instrumented activities are not steady timing,
and subtracting the reducer does not predict the new latency.
Evidence: results/remote-P2-02-full-composition.csv and remote-P2-02-profile.json.

Canonical N256/K8192: old grid=ceil(256/32)*32=256CTAs. Its per-CTA weight
request=32outputs*256K*2BF16components*2bytes=32KiB. New grid=N=256CTAs;
per-CTA weight request=8192K*4FP32bytes=32KiB. Both request8MiB weights overall.
New logical X requests256*8192*2=4MiB; old valid masked X requests256*256*2=128KiB.
Hardware cache reuse may satisfy repeated X reads. These are source-level logical
requests, NOT measured HBM traffic, bandwidth, occupancy, spills or proof of speed.

Related positive evidence: unchanged GEMV body won BF16 single-token pooler,
QKV and unpooler (R09, M19, M20, then official best). Those cases do NOT validate
this original-FP32 weight specialization. Prior router experiments stop native
hi/lo dispatch at m8; no saved native FP32 SIMT router m1 result exists.

P038 pooler TMA54.191us versus best30.263us is rejected; ordinary-copy control
remains unmeasured. Other full-K pooler/QKV and layout failures remain preserved.
This new single-token/direct-FP32 mechanism is independent of those failed tiles.

## Bounded value, not a performance claim

Official decode1 total49.06514949us, of which router m1 is7.46882887us. With every
other timing frozen, target6/5/4.12336597us yields aggregate+0.5078/+0.8642/+1.1840%.
A1.5%aggregate gain requires target3.27582877us. Ordinary official uncertainty is
at least+/-1.5%, so a plausible narrow gain may not resolve aggregate superiority.
These arithmetic scenarios are not native results, predictions, or transferred
unpooler performance. A robust shape-local gain can still be a real improvement;
it must not be presented as a statistically resolved whole-workload score gain.

## Reproducible local validation and actual outcome

Use existing primary-workspace .venv/bin/python, from this projection worktree:

```sh
PY=<primary-workspace>/.venv/bin/python
$PY check.py --submission candidates/projection_router_m1_fp32_gemv.py --device cpu --suite all --output results/router-p039-m1-fp32-gemv-cpu.json
$PY -m unittest discover -s tests -p test_router_m1_fp32_gemv.py -v
$PY static_check.py --submission candidates/projection_router_m1_fp32_gemv.py
$PY -m py_compile candidates/projection_router_m1_fp32_gemv.py tests/test_router_m1_fp32_gemv.py
$PY -m unittest discover -s tests -v
```

Actual on MacM2/Python3.12.13/Torch2.14.1/noCUDA/noTriton:39/39full-size CPU oracle
cases pass with original weights/inputs preserved;6focused test methods and125/125
full regression tests pass;static/syntax and git diff --check pass. CPU oracle
source SHA matches final candidate. FP32 CPU model is elementwise product/sum,
not Triton execution or a timing proxy. Known missing-NumPy warning retained;
no dependency installation was attempted.

Focused coverage includes fresh/outlier/zero/mixed-sign inputs;K1/3/63/64/65/127/
131/8191/8192 and N1/34/37 prefixes;8full-width K8192 sparse-column probes at
0/63/64/511/512/4095/4096/8191 with N253, exact original-FP32 outputs; canonical N256/K8192 dense/outlier/zero inputs; BF16-only
negative control; noncontiguous original weights repacked losslessly; strided X
fallback; all other layer/M dispatch equality and AST invariants. Router oracle
uses1e-4L2 and0.1%column probe bounds, not the BF16 projection bound.

Evidence: results/router-p039-m1-fp32-gemv-cpu.json and
results/router-p039-preflight-cpu.json. Ignored raw local logs:
results/p039-cpu-oracle.local.txt and results/p039-preflight-tests.local.txt.

## Native risks, falsification and ownership

Still UNVERIFIED: native FP32 specialization compilation, reduction order/rounding,
register pressure/spills, cache behavior, CUDA capture/fresh replay correctness,
and H200 performance. Existing BF16 specialization success does not certify them.
Full-K reduction may lose despite removing a launch. Increased logical X requests
and FP32 vector behavior may offset any saving. No main-kernel speed prediction.

If coordinator later admits a check, use this exact committed source after review,
preserve full raw outcome, all9+30correctness/timing cases, and actual source/profile
provenance. Compare target reference-normalized latency to immutable best, report
all29unchanged shapes and the whole score/CI. Fresh m1router outliers/probes/tails/
strided inputs need explicit native coverage as available. Stop on correctness
failure or no meaningful target benefit; no automatic tile/split-count sweep.

No ledger edits, commit, GPU/OKC/Modal/networkauth/install/push/credential action.
Attention E6 still owns the GPU slot; primary coordinator exclusively schedules
and decides admissions. Seven remaining projection checks are not a launch by
this local experiment. No operation was denied, pending or uncertain.

Final candidate SHA256: 0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc


## Coordinator admission decision

Source commitcdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da,
SHA0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc.
Coordinator independently reran6focused CPU tests and static checks: pass.
One native check is justified by removal of the separate reducer using an
already-tested kernel body, not a tile sweep or an assumption that aggregate
0.5-1.2% will be statistically resolved. Source traffic is not hardware evidence.
Attention A2-01 is terminal before this admission; shared GPU slot must be free.

A first LOCAL preparation command used a missing worktree-relative .venv path;
it did not run. The unchanged old admission-pause guard then rejected the wrapper
BEFORE any CLI/network/quota/POST call. No reservation, upload, job or cost.
Known-not-executed; fix is primary-workspace absolute Python plus the already
authorized autonomous scope, not a new grant or bypass of a policy denial.

Prospective exact wrapper command (existing CLI/shared slot; no credentials):

```sh
OKC_CLI=<primary-workspace>/bin/okc OKC_GPU_SLOT=<primary-workspace>/local-notes/gpu-slot.local.json <primary-workspace>/.venv/bin/python cycle2_check.py send P2-04-router-m1-fp32-gemv candidates/projection_router_m1_fp32_gemv.py
```

Projection7remaining before this request. Failures count; do not repeat uncertain
POST. No additional push authorization; new admission/result commits stay local.
