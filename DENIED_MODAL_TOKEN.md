# Modal CLI authentication — creation command denied before execution

2026-10-24: user explicitly approved official client installation and a 12-hour
token for workspace `xuefeng-zhu`, stored privately only on the Mac, with no
credential values in chat/logs/repo. Official PyPI client1.6.2 installed in `.venv`.

Exact attempted operation (not executed):

```sh
umask 077
MODAL_CONFIG_PATH=/Users/frank/Documents/Codex/2026-10-23/task/local-notes/modal-auth.toml .venv/bin/python -m modal token new --expires-in 12h > local-notes/modal-auth-flow.local.txt 2>&1
```

Automatic approval review rejected process creation. Stated reason: credential
creation is authorized, but redirecting stdout/stderr could persist the token
in a log, contrary to the explicit no-token-in-logs restriction. Inspection
confirmed neither the config nor output log exists. No command process, token
grant or auth flow started. Do not retry or change execution route to evade this
denial. A single retry needs materially new relevant authorization and a confirmed
non-executed prior operation.

Proposed narrower request: run the official 12-hour browser authentication
without saving stdout/stderr to a file, preserving the ignored owner-only token
destination, verifying identity/expiry without printing token values. This has
**not** been approved since the denial, and has not been attempted.

Installed official `modal/token_flow.py` lines86–122 prints a browser flow URL,
optional verification code, workspace and success; lines159–169 writes credentials
through the SDK config writer and prints the destination/profile. The token
secret itself is not printed by these inspected paths. This is supporting source
evidence, not authorization to retry after the explicit denial.

No Modal code upload/image build/GPU launch or cost. New OKC checks remain paused.

## Read-only investigation of “service paused”

Visible Stripe portal says `CURRENT SUBSCRIPTION Modal Starter Plan $0.00 per
month Your service is paused.` No cause or resume action was visible. No billing
details changed, no resume attempted. Modal's plan tab marks Starter current,
shows $30 remaining credits, $0 used, net spend cap $0, GPU concurrency10/current0.
No Modal pause explanation was visible in overview, plans or resource limits.

[Modal budgets](https://modal.com/docs/guide/budgets#spend-limits) says credit-covered
workloads may continue at the spend limit. A $0 cap is therefore not documented
as an unconditional all-compute pause while applicable credits remain.
[Modal pricing](https://modal.com/pricing) says Starter credits cover Functions/
Sandboxes CPU, memory and GPU compute, excluding per-token Shared Endpoints.

[Stripe](https://docs.stripe.com/api/subscriptions/object) distinguishes paused
subscription status from
[paused payment collection](https://docs.stripe.com/billing/subscriptions/pause-payment),
which leaves status unchanged. Portal wording alone does not expose Modal's
underlying field/reason or its GPU entitlement mapping. Actual effective GPU
access remains **unknown** without a supported authenticated status response or
authorized bounded execution. No support message sent.
