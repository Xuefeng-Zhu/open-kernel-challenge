# Common-PV native validation controller

This controller prepares one bounded native correctness, frequency and paired
whole-call timing probe. The common-PV compile gates passed, but that is not
numerical or performance evidence. Current preparation is **local only**: the
parent released the previous slot to Projection P049, so a new serialized slot
is required after harness review. No launch is implied by these files.

The control remains exact working source
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The separate candidate remains
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.
The generated diagnostic is independently pinned to
`ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60`.
Original algorithms, oracle/checks and tolerances are unchanged. Diagnostic
counters are collected separately from every performance interval.

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python -B \
  reviews/attention-v2/alpha-one-common-pv-probe/runtime/controller.py \
  --frozen-head <reviewed-40-character-commit>
```

The default action checks exact committed Git bytes and fixed source/support/
diagnostic hashes, then parses Python without importing the worker, Torch,
CUDA, Modal or authentication. The guard unit test uses an explicit tiny worker
fixture; it does not replace actual preflight on the final committed worker.
There are ten offline tests, including both valid negative completion modes.

## Admission and fixed bounds

Explicit `--launch` requires ignored local `--slot-receipt` and
`--billing-receipt`. These record an actual parent serialization grant and a
fresh supported read-only billing observation, not another user-approval step.
Neither receipt enters the payload. The slot names `state: granted`,
`experiment: attention-alpha-one-common-pv-native`, `gpu: H200`, exact
`source_commit`, `granted_utc` and `expires_utc`; it must be under15 minutes old
and extend beyond the210-second watchdog. The billing record names
`workspace: xuefeng-zhu`, `verified_net_spend_cap_usd: 0`,
`total_spend_usd: 0`, `changes_made: false`, `observed_utc`,
`verified_credit_balance_usd` and `total_usage_usd`; it must also be under15
minutes old. Supported CLI reads independently verify paid spend zero,
all listed Modal apps stopped/tasks0, current resource rates and the adjusted
free-credit floor.

The fixed reservation is **$0.58 in free credits**, actual paid authority
**$0**. Twice the current H200/2 CPU/8 GiB rate for all210 watchdog seconds
plus$0.02 must fit in that reserve, and the free-credit floor must exceed it.
No artificial cumulative stage ceiling or billing-setting change is applied.
The existing image is `im-cPTbGpNiX3RarKResPgX7h`, with one maximum container,
one explicit invocation and zero retries. Bounds remain160s worker,170s child,
185s function,20s startup and210s watchdog. No install or credential renewal.

Nine allowlisted files retain their repo-relative layout beneath
`/opt/alpha-one-common-pv-native/`: the new worker, separate diagnostic,
baseline/candidate, existing modal helpers and official reference/checks, plus
license/third-party notices. The controller itself is frozen too but not
uploaded. Existing authentication remains in the protected original
`local-notes/modal-auth.toml`, selected through supported `MODAL_CONFIG_PATH`
and profile `xuefeng-zhu`. Credentials, receipts and other workspace files are
never copied; expanded credential environment variables are rejected.

## Evidence phases and valid negative outcomes

Six declared cases first require complete native correctness/replay/cache and
independent-output isolation evidence. The expected initial counts are:

| Counter | Required |
|---|---:|
| Original-oracle calls | 39 =36 ABA +1 independent setup +2 concurrent |
| Exact returning-A checks | 12 |
| Cross-arm whole K/V cache byte comparisons | 36 |
| Post-concurrent isolation checks / extra candidate replays | 4 /2 |
| Cross-arm output/new-KV comparisons during ABA | 18 |
| Separate diagnostic calls / own-source reference replays | 10 /10 |

The diagnostic's three int32 values per CTA count total/fast/slow iterations.
The worker must verify exact output and complete K/V bytes against its own
uninstrumented candidate, with unchanged source/oracle checks. Instrumented
calls carry no benchmark claim.

The parent requires a nonzero observed fast count in each scored fixture.
If either scored fixture has zero observed fast iterations, an explicit
negative frequency outcome may stop before timing: `frequency_gate.opportunity`
is false, `scored_fast_counts` contains a zero, timing counts are zero and
`benefit.admitted` is false. This is completed falsifying evidence, not an
execution failure. No arbitrary skip-frequency threshold is introduced.

Otherwise the complete native run requires64 checked uninstrumented warmup/
measured replays and32 additional cross-arm output/new-KV comparisons. The
predeclared performance question is at least2% target latency reduction for
**each** fresh seed135731/135732, all six reverse-order pairs favorable, and
unchanged full-B16/L8-control slowdown no greater than2% for each seed.
Cold-L2 eviction, input/cache restore and frequency diagnostics stay outside
timing. A failed performance gate is a valid complete negative result.

The controller checks exact frozen worker/algorithm/support/diagnostic hashes,
completion flags, full expected counts and the two timing-family records. It
accepts a completed negative benefit; it does not turn a point estimate or
`benefit.admitted` into performance/official admission. Raw per-seed samples,
calibration, control drift and gate arithmetic require independent parent
review. No official check, selection promotion or ranked entry is automatic.
Actual hardware overlap is not inferred from independent submitted streams.

## Transport and shutdown

Returned JSON is bounded to500,000 bytes and SHA256 checked. The exact raw
report is stored unchanged before success-contract validation, preserving any
failed or partial evidence. The ignored `run.local.json` ledger is created
exclusively before app startup; an existing ledger/result blocks duplication.
No timeout, denial or uncertain request is retried by this controller.

An independent watchdog requests stop, and `finally` requests stop again only
if no previous request was recorded. Supported CLI reads record app state and
billing. Initial `stopping` or nonzero tasks stay explicitly unclosed; a
separate stopped/tasks0 read is required before another job. Original receipts
must not be rewritten to imply closure. The controller returns success for a
complete, closed run even if its benefit gate failed. Missing/failed evidence
or unverified shutdown remains unsuccessful and requires reconciliation.

No services, authentication reads, network or GPU actions are performed by the
default preflight or unit tests. The parent must finish review/freeze, run the
actual offline preflight and coordinate a new slot before execution.
