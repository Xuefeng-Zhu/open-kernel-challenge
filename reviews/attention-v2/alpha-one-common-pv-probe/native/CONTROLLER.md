# Alpha-one common PV: compile-only artifact controller

The controller prepares one H200 compiler-artifact probe of the isolated
alpha-one candidate. It compiles four fixed units: baseline/candidate producer
and merge. Compilation completion is not correctness, performance or proof of
the intended machine-code improvement. No kernel execution, correctness run,
performance timing, profiler, official check or ranked entry is authorized by
this controller. The selected working source remains unchanged.

This preparation is local only while Projection owns the measurement slot.
The parent must review, commit and run the actual frozen-head preflight, then
obtain a new serialized slot before any explicit launch. There is no new user
approval requirement for routine steps already inside the standing project
scope; the slot receipt coordinates measurement serialization.

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python -B \
  reviews/attention-v2/alpha-one-common-pv-probe/native/controller.py \
  --frozen-head <reviewed-40-character-commit>
```

The default action reads the exact committed Git objects, requires current
files to match, checks fixed source/license hashes and parses Python without
importing the worker, Torch, CUDA, Modal or authentication. Offline tests use
explicit mocks, including a tiny worker fixture for the Git guard unit test;
that fixture does not substitute for preflight on the completed real worker.

The explicit `--launch` action requires local `--slot-receipt` and
`--billing-receipt` paths. Neither receipt enters the payload. The actual slot
receipt names `state: granted`, `experiment: attention-alpha-one-common-pv-compile`,
`gpu: H200`, the exact `source_commit`, `granted_utc` and `expires_utc`. It must
be less than 15 minutes old and remain valid beyond the 140-second watchdog.
The billing record names `workspace: xuefeng-zhu`,
`verified_net_spend_cap_usd: 0`, `total_spend_usd: 0`, `changes_made: false`,
`observed_utc`, `verified_credit_balance_usd` and `total_usage_usd`; it must
also be less than 15 minutes old. Live supported CLI reads separately confirm
paid spend zero, all listed Modal apps stopped/tasks0 and current rates.

The fixed free-credit reserve is **$0.40**, actual paid authority **$0**. Before
admission, twice the live H200/2 CPU/8 GiB resource rate for the complete
140-second watchdog plus $0.02 must fit inside that reserve; the adjusted free
credit floor must exceed it. At the previous $4.54/GPU-hour rate and recorded
CPU/memory rates, the model is about $0.38545; this is a reserve model, not an
invoice. Rates and guardrails are checked again before launch. No artificial
cumulative stage-credit ceiling is applied and no billing limit is raised.

Bounds are 90 seconds for the worker, 100 seconds for its child process,
115 seconds for the Modal function, 20 seconds for startup and 140 seconds for
the independent app-stop watchdog. There is one explicit invocation, one
maximum container and zero retries. The existing image is
`im-cPTbGpNiX3RarKResPgX7h`; no installs are performed.

Only five files are uploaded under `/opt/alpha-one-common-pv/` with the same
repo-relative layout: the new worker, this probe's baseline/candidate, and the
root license/third-party notices. The controller is also checked against the
frozen Git commit but is not uploaded. The source hashes are:

| Source | SHA256 |
|---|---|
| Working baseline | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Alpha-one candidate | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |

Authentication stays in the existing protected original
`local-notes/modal-auth.toml`, selected by supported `MODAL_CONFIG_PATH` and
profile `xuefeng-zhu`. Secret bytes are not printed or copied. The controller
rejects credential environment expansion and does not renew credentials.

## Bounded artifact transport

The entire worker JSON response is at most **2 MiB** and carries an exact SHA256.
Duplicate JSON object keys are rejected. The artifact bundle uses
`encoding: gzip+base64`, compressed/uncompressed hashes, uncompressed size,
and a per-file `sha256`/`size` catalog. Its deterministic gzip body expands to
JSON `{"files": {"<name>": "<base64 bytes>"}}`.

Decompression uses one bounded `zlib` call with an **8 MiB** output cap; there
is no unlimited `gzip.decompress` or flush. EOF, unused-data and unconsumed-tail
checks reject truncation, concatenated members, trailing data and expansion
bombs before JSON parsing. The decoded file set must exactly match the catalog,
with strict base64, size and hash checks for every member, a 4 MiB per-file cap
and an aggregate 8 MiB decoded-file cap. Optional compressed-size metadata is
also verified against the actual compressed bytes. The general count cap is
32 files; only these **24 exact
flat filenames** are permitted:

`(baseline|candidate).(producer|merge).(ttir|ttgir|llir|ptx|cubin|sass)`

This fixed set excludes absolute paths, directories, traversal, alternate
suffixes and archive links. No tar/zip extraction is performed. The entire
bundle is validated in memory before any files are written. Persistence creates
a fresh `native/artifacts/` directory and exclusively creates each file; any
existing destination blocks overwrite and requires reconciliation. A compact
manifest retains hashes/sizes/framing without duplicating the encoded bundle.

Claimed compilation completion requires `status: compiled`, `compile_complete:
true`, exact source/license/worker hashes, all four fixed unit records, all24
artifacts, intrinsic `protocol.compile_only: true` and integer `kernel_launches:
0`. Correctness claims, timings and official admission are rejected. Failed or
partial reports and their valid available artifacts remain diagnostic evidence;
they do not become successful completion.

## Request and resource closure

The ignored `run.local.json` ledger is created exclusively before app startup;
an existing ledger or result blocks another launch. Raw returned JSON is stored
unchanged before success validation. Any missing result, timeout or uncertain
request is reconciled rather than retried. The controller requests app stop in
`finally` and independently records app state and billing. Initial `stopping`
or nonzero tasks remain unclosed and require a separate terminal read before
another job. Receipts must not be rewritten to imply shutdown.

The controller's default action and 12 local guard/transport tests make no
service, credential or GPU calls. Passing them prepares a reviewable launch
path; it does not create a Modal app or admit the alpha-one algorithm for use.
