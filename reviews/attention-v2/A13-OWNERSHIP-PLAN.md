# A13: exact-source H200 ownership diagnostic

User approved one bounded Modal diagnostic on 2026-10-25: incremental free
credits at most $0.25, net paid $0. Existing authentication only; no renewal,
new credentials, OKC check, scored entry or algorithm change.

Frozen candidate: mixed_fa3_decode_parallel_merge.py, original source commit
7e302d2366b194c32c433f6d6f9cc7b0b59b3928, SHA256
c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0.
Official source result is A2-11 cfa8a8670e394f80; this probe creates no new score.

Existing CLI app/billing read succeeded and profile was xuefeng-zhu. On the
user's Mac Chrome, https://modal.com/settings/xuefeng-zhu/usage showed $28.85
credits, $0 net paid and $0 spend limit, with zero live apps (06:04 UTC).
No setting, permission, payment or credential was changed.

Use existing image im-cPTbGpNiX3RarKResPgX7h and eleven explicitly allowlisted
candidate/harness/reference/license files plus their generated manifest.
No repository, auth, logs, local-notes or third-party competitor code is uploaded.
The controller is local; only its self-contained bounded entry function is
serialized. Return at most 24 KiB (bounded JSON may use zlib compression), no PTX, trace or tensor payload, to avoid
repeating the earlier large-result BlobCreate failure.

Resources: one H200, CPU request/limit2, RAM request/limit8GiB, min/buffer0,
max1, scaledown2s, startup30s, function90s, internal75s, worker watchdog83s,
client stop watchdog120s. No detach/deploy/schedule/retries or mounted secrets.
Published rates at https://modal.com/pricing: H200 $0.001261/s,
CPU $0.0000131/core/s, memory $0.00000222/GiB/s. Conservative bound for
145seconds plus5s, 10% margin and $0.01 overhead is $0.2253184.
Reserve full $0.25; prior conservative cumulative $1.708917914 is preserved,
not reduced by delayed lower billing. Total with reserve is $1.958917914.
Actual app lifetime upper bound and observed billing must remain separate.

Correctness covers independent input/cache/output/private graph pools,
same-address A/B/A with changed data and actual lengths, exact official output
and new-cache tolerances, finite outputs, unchanged cache rows, and restored-A
bitwise comparison. Only checked graph replays count. Separate stream enqueue
is insufficient: summarize profiler kernel interval intersections across distinct
streams; absent overlap must be reported unverified. Do not claim performance
from diagnostic profiler timing. Warmup, fixture/reference work and compilation
are within deadline, with explicit partial/omitted coverage on timeout.

Run commands (one dispatch only after committed source and all gates):

```sh
.venv/bin/python reviews/attention-v2/ownership_modal.py build
.venv/bin/python reviews/attention-v2/ownership_modal.py run
```

After success/failure, inspect the accepted app ID, stop if needed, and verify
stopped/tasks0 via supported CLI/browser. An uncertain launch remains reserved;
never automatically retry. This document is a plan, not a GPU pass.
