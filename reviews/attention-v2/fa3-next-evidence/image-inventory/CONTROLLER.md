# CPU-only image eligibility inventory controller

This controller prepares one read-only inventory of existing vendor module and
library metadata in immutable image `im-cPTbGpNiX3RarKResPgX7h`. It supports
major-redesign eligibility research, not a performance candidate. Module
presence, absent source or a library fingerprint alone does not establish a
usable implementation, GPU capability, bottleneck or speedup.

Preparation is local only. No service, authentication read, network, CPU app or
GPU action is performed by the default preflight or eight focused unit tests.
The parent must independently review/freeze the final worker and controller,
run actual preflight, and explicitly admit one CPU inventory within standing
project authorization. No GPU measurement slot or GPU reserve is consumed.

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python -B \
  reviews/attention-v2/fa3-next-evidence/image-inventory/controller.py \
  --frozen-head <reviewed-40-character-commit>
```

Default preflight requires exact current `HEAD`, identical committed source
bytes and fixed license hashes, then parses Python without importing the
worker, vendor packages, CUDA or Modal. The mocked Git unit test uses a tiny
worker fixture and cannot substitute for this actual completed-worker check.

## CPU resources, budget and supported access

Explicit `--launch --billing-receipt <existing-local-receipt>` performs the
supported existing-authentication checks and fresh CLI guardrails. There is
no GPU argument in the Modal function declaration: resources are exactly
CPU `(1,1)`, memory `(2048,2048)` MiB, min0/max1 containers, no idle buffer and
zero retries. Bounds are30s worker,35s hard child timeout,45s function,
10s startup and60s independent app-stop watchdog. The child process is only a
supervisor for the worker; the inventory worker itself starts no subprocess.

The fixed free-credit reserve is **$0.02**, actual paid authority **$0**.
The conservative resource bound is
`2 ×60s ×(one CPU +2 GiB live hourly rate)/3600 +$0.005`.
It must fit inside the reserve before admission. The rate calculation never
reads a GPU-price field. At previously observed CPU/memory rates it is about
$0.00711, a reserve model rather than an invoice. No artificial cumulative
stage ceiling, billing-setting change, install or paid resource is introduced.

The existing regular billing receipt must name `workspace: xuefeng-zhu`,
`verified_net_spend_cap_usd: 0`, `total_spend_usd: 0`, `changes_made: false`,
`observed_utc`, `verified_credit_balance_usd` and `total_usage_usd`. It must be
under15 minutes old. Supported CLI reads independently require current actual
paid spend zero, all listed Modal apps stopped/tasks0 and a conservatively
adjusted free-credit floor exceeding the full$0.02 reserve. Official service
GPU scheduling is not changed by this CPU app; no GPU slot receipt is used.

Authentication remains in the existing mode600 original
`local-notes/modal-auth.toml`, selected through supported `MODAL_CONFIG_PATH`
and profile `xuefeng-zhu`. Credential environment expansion is rejected. No
credential is printed, copied or renewed. Fresh CLI failures stop admission
rather than invoking another access route.

## Exact payload and bounded result

Only three allowlisted files are uploaded: the reviewed new worker, root
`LICENSE`, and root `THIRD_PARTY_NOTICES.md`, mapped flat beneath
`/opt/fa3-image-inventory/`. The controller is frozen as well but not uploaded.
Receipts, configs, credentials, candidate kernels and other workspace files
are outside the payload. The worker is invoked with explicit `--deadline30`.
The function blocks network and restricts Modal access.

The worker reads bounded vendor source/metadata using standard-library module
discovery and file APIs. It requests no GPU/context, imports no Torch/ops code
for execution, starts no subprocess, reads no user/config/credential paths,
and makes no network or driver query. It caps64 source files,8 MiB aggregate
source bytes,30 seconds and262,144-byte JSON output. Source snippets and
per-file limits are independently reviewed in the worker rather than inferred
from controller completion. Missing packages/source and reached bounds remain
explicit incomplete eligibility evidence, not permission or hardware findings.

Transport requires the exact returned SHA256, a JSON object no larger than
262,144 bytes, exact frozen worker hash, `gpu_requested: false`,
`cuda_context_requested: false` and integer `candidate_kernel_launches: 0`.
The unchanged raw JSON is exclusively stored in `result.json`; its library
presence metadata never authorizes a kernel launch or an official check.

## Ledger and closure

The ignored `inventory-run.local.json` ledger is exclusively created before
app startup. An existing ledger/result blocks duplicate requests. A60s
watchdog requests stop; `finally` also requests stop only if no previous stop
was recorded. App state and billing are independently read and preserved.
Initial `stopping` or nonzero tasks remain explicitly unclosed and require a
separate stopped/tasks0 reconciliation before another job. No receipt is
rewritten to imply closure, and no uncertain request is automatically retried.

The controller returns success only for a retrieved exit0 result and verified
app shutdown. Failed/partial metadata remains diagnostic evidence. The parent
records final cost and closure separately before archiving reviewed artifacts
to the existing private repository. No ranked entry or visibility change.
