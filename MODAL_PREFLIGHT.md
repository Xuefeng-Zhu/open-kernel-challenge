# Modal preparation — local only, launch gated

Local verification 2026-10-24 18:09 UTC: all **69 unittest tests passed**,
including 3 bundle-boundary tests; syntax checks passed. Worker benchmark entry
on this Mac correctly records `Explicit H200 required; no fallback GPU`, zero
cases. No native CUDA, container image or Modal SDK execution has occurred.

## Actual access and budget status

2026-10-24 (rechecked after local tests): Mac Chrome (profile Xuefeng) shows
`https://modal.com/login?next=%2Fapps`, not a signed-in workspace. Stopped before
“Continue with GitHub”; the tab was handed to the user. No actual credit balance,
payment method, workspace identity or spend cap observed. No local Modal CLI,
package, `~/.modal.toml`, or Modal token environment variables found. No token
values inspected; no credential, account grant, upload, image build or GPU launch.

User permits free-credit-only experiments with additional out-of-pocket $0.
Before launching: user completes web login; inspect actual workspace credits,
payment status and spend limit. A **spend limit** caps net usage after credits,
whereas a **usage budget** caps gross usage before credits. Public documentation
does not establish that this account has credits, that the UI accepts zero, or
that enforcement guarantees zero overshoot. If the requested $0 constraint cannot
be established, stop and report. Do not add a payment method or upgrade.

No CLI authorization exists. After account/workspace is known, a separate request
must identify that workspace and ask for the official temporary CLI authorization
(documented `modal token new --expires-in 12h`) on this Mac. The user completes
login/grants; never paste credentials into chat. Do not run that command now.

## Local deliverables and limits

`modal_preflight.py` creates a **local** whitelist bundle and SHA256 manifest; it
has no Modal dependency or network calls. Only committed exact bytes can enter.
It excludes git history, accounts/config, caches, CLI binaries, raw logs, attention
drafts and all unrelated projects. The manifest includes the candidate, frozen
R10, downloaded official oracle/fixtures/baseline, harness and licenses. Review
this manifest before separately authorized source transfer. It is not permission
to upload. Default gates are all false; there is deliberately no launch command.

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
the client exits. No Modal SDK has been installed or tested here.

The advertised public base is
`lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`,
SGLang commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`
([official environment](https://openkernelchallenge.dev/api/v1/serving)). This is
**not** the private judge overlay `localhost:30500/kc/judge@sha256:9a7f…` reported
by OKC. Do not substitute that overlay digest as a public image or seek registry
credentials. Public-base pull, Python path, entrypoint and dependency compatibility
remain untested. Probe actual Torch/Triton/SGLang versions and CUDA driver before
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
[image API](https://modal.com/docs/sdk/py/latest/Image).

## Official cycle2 checkpoint

New dispatch paused for Modal-first workflow. P2-01 `eeec05d897ef4946` is terminal,
passed all 39 cases, score 1.1290207051× (95% interval [1.1120853945,1.1459560157]).
R10 was 1.1207320490×; intervals overlap, so the whole-workload gain is unresolved.
Cycle2: **1/10 used, 9 remaining, 0 active, 0 uncertain**, separate from exhausted
prior budgets. Last platform quota snapshot: 12/20 projection checks, 0/10 scored;
historical only, not a current allowance. No new source push authorized.
