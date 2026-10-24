# Modal preparation and completed first H200 smoke

Latest composition: MODAL_COMPOSITION_RESULTS.md. Full source
candidates/projection_full_composition.py (c34c497b...) is best observed Modal
point estimate: weighted gross1.022014x over R10,1.010889x over P2-01; no official
score/CI. App ap-XxzsrPfRejsEpPVQCxIqzJ passed all five39-case CUDA suites and all
warmup/steady outputs, now stopped/tasks0. QKV steady max17.952us after declared
six warmups;40us warmup sample retained. Pooler m8/16/32/128 gains supported,
m256 unresolved. All4apps stopped, cumulative credits$0.29/net$0, remaining$29.71,
conservative stage bound$0.44625<$2. Source/protocol/result commits separate;
no push/OKC call. Report before expanding; preserve earlier evidence below.

Current continuation: see MODAL_STAGE2_RESULTS.md. Unpooler019 and independent
QKV020 both completed/passed and stopped; all native correctness checks pass.
Latest cumulative Modal billing$0.21credits/net$0, conservative bound$0.3286below
$2. No new official check or push. Report before expanding experiments.

Read **MODAL_RESULTS.md** for the completed native result and current budget.
One initial smoke completed and the app is verified stopped; the requested continuation has now also completed. New official OKC dispatch remains paused. All work stays local/unpushed.

## Actual access and budget status

Existing GitHub login completed in the user's Mac Chrome without new OAuth or
terms. Workspace `xuefeng-zhu`, connected GitHub `Xuefeng-Zhu`, Starter. Before
running, actual credits $30.00/$0 used and an existing payment method were verified.
Set net Spend limit from default $12.50 to $0 under the user's express constraint;
it persisted and was rechecked after the run. Gross usage limit remains $42.50.

Official Modal CLI 1.6.2 installed. One original token command was rejected before
execution due to logging risk. After new explicit approval, its sole safer retry
without output redirection succeeded. Details/history: DENIED_MODAL_TOKEN.md.
Token is in ignored task-local `local-notes/modal-auth.toml`, mode0600; supported
read-only metadata verifies user/workspace xuefeng-zhu and expiry
2026-10-25 06:26:47UTC. No secret is printed, logged, committed or uploaded.
Do not create another credential or extend its lifetime automatically.

The approved 17-file source/license bundle and generated manifest were uploaded
only for app `ap-ukvOyTeF1ztdOLggl5xJee`. Actual H200 execution passed all checks;
app stopped at18:40:10UTC with tasks0. The portal's unexplained “Your service is
paused” did not prevent this credit-covered run. No resume, payment change,
upgrade or new agreement. Latest supported billing snapshot: gross$0.12,
credits-$0.12, billed$0; detailed ephemeral-app cost$0.12054281. UI balances lag.

A spend limit caps net charges after credits, whereas a usage budget caps gross
usage before credits. Keep $0 net cap; recheck applicable credits and expiry
before any newly authorized run. Do not promise absolute billing enforcement or
use per-token Shared Endpoints, which the included compute credits do not cover.
Local screenshots under local-notes are ignored and never part of a transfer.

## Local deliverables and limits

`modal_preflight.py` creates a **local** whitelist bundle and SHA256 manifest; it
has no Modal dependency or network calls. Only committed exact bytes can enter.
It excludes git history, accounts/config, caches, CLI binaries, raw logs, attention
drafts and all unrelated projects. The manifest includes the candidate, frozen
R10, downloaded official oracle/fixtures/baseline, harness and licenses. Review
this manifest before separately authorized source transfer. It is not permission
to upload. Default manifest gates are false historical planning metadata, not authorization.
The separate modal_smoke.py launcher has a durable one-attempt guard; its one
approved attempt is already consumed, so do not rerun it.

```sh
.venv/bin/python modal_preflight.py --candidate candidates/projection_qkv_m1_simt.py
.venv/bin/python modal_preflight.py --candidate candidates/projection_qkv_m1_simt.py --build local-notes/modal-qkv-bundle
.venv/bin/python -m unittest discover -s tests -p test_modal_preflight.py -v
.venv/bin/python -m py_compile modal_preflight.py modal_gpu_worker.py
```

Future authorized remote runner must use only this bundle, disable implicit
project source upload, and use an ephemeral app (`modal run`, no detach/deploy,
schedules or volumes/secrets). Request **H200**, 2 CPU/8192MiB, max_containers=1,
min_containers=0, buffer_containers=0, scaledown_window=2, timeout=600,
startup_timeout=120, retries=0. Clear the SGLang image entrypoint. Verify these
options against the installed SDK before any launch, and verify app stopped after
the client exits. These options were verified against installed Modal1.6.2 and used for the completed smoke.

The advertised public base is
`lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`,
SGLang commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`
([official environment](https://openkernelchallenge.dev/api/v1/serving)). This is
**not** the private judge overlay `localhost:30500/kc/judge@sha256:9a7f…` reported
by OKC. Do not substitute that overlay digest as a public image or seek registry
credentials. Public-base pull and cleared entrypoint succeeded. Runtime Python was
/opt/sglang/bin/python; Torch2.14.1+cu130/Triton3.8.0 and the baseline all executed. Probe actual Torch/Triton/SGLang versions and CUDA driver before
tests; do not silently install a different GPU stack or use a fallback GPU.

Once provisioned with separate authorization, invoke inside the frozen bundle:

```sh
python modal_gpu_worker.py --candidate candidates/projection_qkv_m1_simt.py --output-dir /tmp/qkv-result --repeats 6
```

The worker validates manifest hashes, refuses non-H200 hardware, and supervises
three full 39-case correctness runs (candidate/R10/SGLang), then a three-runner
graph diagnostic with six balanced execution orders on the official 30 shapes.
Inputs are fresh for every repetition; the unmodified FP64/probe oracle checks
every output. A 256MiB cold-L2 read precedes each event interval. Reported times
include graph replay overhead: no guessed 4µs subtraction, official score or
confidence interval. Compilation/warmup/capture durations are recorded separately.
A 540-second absolute supervisor deadline kills the process group on timeout,
without retry. Report container/image startup and app wall time separately in
the eventual launcher. Correctness or timeout failures stop the pipeline and
retain logs, rather than benchmarking an invalid candidate.

H200 is publicly $0.001261/s (~$4.5396/h). With requested 2 CPU and 8GiB,
600 seconds is approximately **$0.782964 gross**, excluding build/startup/idle and
possible resource bursts. This is an estimate, not a spending guarantee. Actual
credits, payment method and verified cap are prerequisite to launch.

Sources: [GPU requirements](https://modal.com/docs/guide/gpu),
[prices](https://modal.com/pricing), [budget semantics](https://modal.com/docs/guide/budgets),
[ephemeral apps](https://modal.com/docs/guide/apps),
[temporary tokens](https://modal.com/docs/cli/latest/token),
[local credential configuration](https://modal.com/docs/sdk/py/latest/config),
[image API](https://modal.com/docs/sdk/py/latest/Image).

## Official cycle2 checkpoint

New dispatch paused for Modal-first workflow. P2-01 `eeec05d897ef4946` is terminal,
passed all 39 cases, score 1.1290207051× (95% interval [1.1120853945,1.1459560157]).
R10 was 1.1207320490×; intervals overlap, so the whole-workload gain is unresolved.
Cycle2: **1/10 used, 9 remaining, 0 active, 0 uncertain**, separate from exhausted
prior budgets. Last platform quota snapshot: 12/20 projection checks, 0/10 scored;
historical only, not a current allowance. No new source push authorized.
