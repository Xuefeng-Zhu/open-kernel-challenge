The controller is prepared but has not been launched. Its default action reads
the exact frozen Git commit and compiles Python source without importing the
worker, Torch, CUDA, Modal, or authentication. Local guard tests also make no
network calls.

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python -B \
  reviews/attention-v2/sliding-unsplit-probe/runtime/controller.py \
  --frozen-head <reviewed-40-character-commit>
```

The explicit `--launch` action additionally requires `--slot-receipt` and
`--billing-receipt`. These are local records of an actual parent slot grant and
an actual supported billing observation, respectively; preparing the files
does not create authorization or establish their truth. Neither receipt is
uploaded. The slot record must contain `state: granted`,
`experiment: attention-sliding-unsplit`, `gpu: H200`, the exact
`source_commit`, `granted_utc`, and `expires_utc`. The grant must be less than
15 minutes old and remain valid for more than the 210-second watchdog.

The billing record must contain `workspace: xuefeng-zhu`,
`verified_net_spend_cap_usd: 0`, `total_spend_usd: 0`,
`changes_made: false`, `observed_utc`, `verified_credit_balance_usd`, and
`total_usage_usd`. It must be less than 15 minutes old. Supported CLI queries
independently require current actual paid spend zero, every listed app stopped
with zero tasks, and sufficient conservatively adjusted free credits.

The fixed reservation is **$0.58 in free credits**, with actual paid authority
**$0**. It is admitted only if twice the live resource rate for all 210 seconds
plus $0.02 fits within that reservation. At the previously observed H200 rate
with 2 CPU cores and 8 GiB, this model is $0.56817. Rates are queried again
before launch; this is a reserve, not a promised charge. There is no automatic
retry or additional experiment. The worker deadline is 160 seconds, child hard
limit 170 seconds, function limit 185 seconds, startup limit 20 seconds, and
app watchdog 210 seconds. A timeout retains the worker's available prefix
report and remains partial evidence.

The payload reproduces the reviewed repo-relative layout beneath
`/opt/sliding-unsplit/`. It contains only the worker, its explicit read-only
installed-FA3 identity helper, baseline, candidate,
existing harness helpers, official oracle/checks, and license notices. Existing
authentication stays in its protected original `MODAL_CONFIG_PATH` and is never
copied into the worktree or payload. No installation, billing change, credential
renewal, hardware counter collection, official check, or formal entry is performed.
The separate worker may collect its one predeclared Torch/CUPTI activity session
after the installed-API gate. Those instrumented events are not benchmark timings.

The ignored `run.local.json` ledger is created exclusively before app startup.
Its existence prevents duplicate launches. The exact returned result is bounded
to 500,000 bytes and hash checked. The controller requests stop in `finally`
and reads app state and billing independently; a `stopping` state or nonzero
task count is explicitly unclosed and requires reconciliation before any next
job. An absent result or uncertain request must be checked, never relaunched.
