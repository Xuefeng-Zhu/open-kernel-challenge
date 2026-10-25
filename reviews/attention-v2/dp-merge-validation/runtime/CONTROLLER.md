# DP multi-chunk correctness controller

This independent controller prepares one bounded H200 correctness validation of
the already selected fusion-plus-lookahead source. It does not edit either
candidate, benchmark performance, send an official check, collect counters,
install software, change credentials or billing, or create a ranked entry.
Writing this file does not launch anything.

The default command is an offline preflight. It reads the exact committed Git
objects, requires the current working files to match them, checks the fixed
candidate/control/support hashes and compiles Python without importing the
worker, Torch, CUDA, Modal or authentication:

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python -B \
  reviews/attention-v2/dp-merge-validation/runtime/controller.py \
  --frozen-head <reviewed-40-character-commit>
```

The explicit `--launch` action requires `--slot-receipt` and `--billing-receipt`.
These local records describe an actual parent serialization grant and a fresh
read-only billing observation, not an additional user approval. They are never
uploaded. The slot receipt names `state: granted`,
`experiment: attention-dp-merge-validation`, `gpu: H200`, the exact frozen
`source_commit`, `granted_utc` and `expires_utc`. It must be less than 15 minutes
old and extend more than the 210-second watchdog into the future.

The billing record names `workspace: xuefeng-zhu`,
`verified_net_spend_cap_usd: 0`, `total_spend_usd: 0`, `changes_made: false`,
`observed_utc`, `verified_credit_balance_usd` and `total_usage_usd`. It also
must be less than 15 minutes old. Supported CLI queries independently check
actual paid spend zero, all listed Modal apps stopped with zero tasks, current
resource rates and a conservatively adjusted free-credit floor.

The fixed free-credit reserve is **$0.58**, actual paid authority **$0**. The
controller admits only when twice the current GPU/2 CPU/8 GiB resource rate for
all 210 watchdog seconds plus $0.02 fits inside that reserve, and the verified
free-credit floor exceeds the reserve. There is no artificial cumulative stage
ceiling. No billing limit is raised. Bounds are 160 seconds for the worker,
170 seconds for its child process, 185 seconds for the Modal function,
20 seconds for startup and 210 seconds for the independent stop watchdog.
There is one explicit invocation, one maximum container and zero retries.

The existing image is `im-cPTbGpNiX3RarKResPgX7h`. Eight allowlisted files are
mapped beneath `/opt/dp-merge-validation/` with their original repo-relative
layout: this new worker; the unchanged sibling
`fusion-lookahead-probe/{baseline,candidate}.py`; existing `modal_worker.py`,
official reference/checks, license and third-party notices. The controller
itself is also verified against the frozen commit, but is not uploaded.
Authentication stays at the existing protected original
`local-notes/modal-auth.toml`, selected through supported `MODAL_CONFIG_PATH`
and profile `xuefeng-zhu`. Secret bytes are neither printed nor copied, and no
credential renewal or environment expansion is attempted.

The intrinsic correctness-only worker covers two generic DP-head full-B32/L1
cases: page64/scale1/64 at ctx8193 with two splits; page1/scale1 at ctx16385
with three splits and masked merge padding. Both retain the original oracle,
tolerances, candidate algorithm and source hash. Live same-address A/B/A graph
replay, whole-cache equality, exact returning-A, independent graph submissions
and post-submission isolation checks are retained. Actual hardware overlap and
internal scratch pointers are not inferred from independent submissions.

The bounded transport must be valid JSON with the exact returned SHA256 and no
more than 500,000 bytes. Claimed success additionally requires the frozen
worker hash, exact source/support hash maps, `correctness_complete: true`,
intrinsic correctness-only protocol and these exact completion counts:

| Counter | Required |
|---|---:|
| Original oracle calls, including setup/concurrent submissions | 15 |
| Exact return-A checks | 4 |
| Cross-arm whole-cache byte comparisons | 12 |
| Concurrent original-oracle calls | 2 |
| Independent graph own-A setup oracle | 1 |
| Post-concurrent isolation checks | 4 |
| Extra candidate replays for isolation | 2 |

Claimed success with timings or official admission is rejected. Failed/partial
reports are still retrieved unchanged for diagnosis. A successful correctness
report is robustness evidence, not performance admission or a new score.

The ignored `run.local.json` ledger is created exclusively before app startup;
any existing ledger/result blocks duplicate launch. The controller saves the
raw returned result before success-contract validation, requests app stop in
`finally`, and independently reads app state and billing. Initial `stopping`
or nonzero tasks remain unclosed and require parent reconciliation; receipts
are not rewritten to imply closure. An uncertain operation must be checked,
never repeated. No next job starts until this app is independently stopped
with zero tasks and the parent releases the serialized slot.

Local guard tests use mocks and no network, credentials or GPU. The parent
must freeze and review the final new worker/controller together before launch.
