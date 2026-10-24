# 018 — first authorized Modal H200 diagnostic

Status: completed/passed; app `ap-ukvOyTeF1ztdOLggl5xJee` verified stopped/tasks0.
One initial smoke allowance consumed; durable ledger
`modal-smoke-budget.json` reserves before app startup, refuses all automatic repeats.
Official checks remain paused; this is not another OKC request.

Hypothesis: experiment016's R10-derived single-token QKV SIMT kernel compiles and
passes the official oracle on H200, and may improve that shape's gross graph time.
Candidate source SHA256 `9cea9ceed1dab548368746e83540f25edff94f472e8701c05526ea9995759d55`;
source commit544f4a8. Compare unchanged frozen R10 and official SGLang in the same
ephemeral H200 container. Do not combine unverified unpooler/layout changes.

User approved the 17-file whitelist bundle (plus generated hash manifest), one H200,
free credits only, additional paid$0. Verified accountxuefeng-zhu has$30credits,
net spend limit$0, existing payment method, current Starter plan. Supported billing
summary reports zero metered/billed usage; app list empty before launch. Stripe's
pause wording remains unexplained; no resume, plan/payment or cap change permitted.
A service/billing rejection stops this operation; no retry/newgrant/workaround.

Official client1.6.2, 12-hour token safely verified and stored in ignored mode0600
task-local config. No auth/config files enter the bundle. App(include_source=False)
and Function(include_source=False,serialized=True) send only the standalone wrapper
defined inside the allowlisted worker, plus explicit bundle mount. Wrapper has no
host closure/globals or credentials. No volumes, secrets, schedules or deployments;
network blocked and Modal workspace access restricted inside the GPU function.

Public SGLang base digest and exact resource limits are in modal_preflight.py.
Compile/correctness/capture setup and gross graph-replay timing are separate.
Native compatibility and diagnostic graph timing are now measured; see results below.

```sh
.venv/bin/python -m py_compile modal_smoke.py modal_gpu_worker.py
.venv/bin/python -m unittest discover -s tests -p test_modal_preflight.py -v
.venv/bin/python modal_preflight.py --candidate candidates/projection_qkv_m1_simt.py --build local-notes/modal-qkv-smoke-v1
MODAL_CONFIG_PATH="$PWD/local-notes/modal-auth.toml" .venv/bin/python modal_smoke.py --bundle local-notes/modal-qkv-smoke-v1 --approved-initial-smoke
```

Local syntax and3boundary tests passed; all69local tests still pass after client
installation, `pip check` clean. Actual H200: candidate/R10/SGLang39/39correctness
each,30graph shapes x6fresh inputs x3runners pass. App setup486.849s, worker48.875s,
client542.120s. QKV m1 gross median16.960us vs R10 19.152us vs SGLang19.728us;
first candidate38.176us sample retained. Smoke evidence only, no official score/CI.
Billing snapshot$0.12metered, credits-$0.12,billed$0; breakdown$0.12054281,
asynchronous UI counters differ. No paid activation or new official request.
Complete raw evidence/qualifications: MODAL_RESULTS.md and results/remote-modal-smoke.json.
Do not rerun this command without a new budget authorization.
