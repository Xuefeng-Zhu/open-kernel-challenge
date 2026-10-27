# P042 — direct official check of unchanged P039: duplicate

The user changed the workflow to the official runner before a new optimization
candidate existed. This request therefore sent exact P039 as baseline verification,
not a new algorithm. The prepared P041 Modal worker was never uploaded or launched.

## Identity and exact operation

- Source: `candidates/projection_router_m1_fp32_gemv.py`.
- Source commit: `cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`.
- SHA256: `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
- Historical selected job: `d5636ac8d1404abd`, 1.173238998969285×.
- New request ID: **`d9b940ff4fa14a1b`**.
- Accepted: 2026-10-25 18:48:18 UTC; terminal read: 18:48:29 UTC.

```sh
OKC_CLI=<primary-workspace>/bin/okc \
OKC_GPU_SLOT=<primary-workspace>/local-notes/gpu-slot.local.json \
$PY -B cycle2_check.py send P2-06-p039-unchanged-recheck \
  candidates/projection_router_m1_fp32_gemv.py

OKC_CLI=<primary-workspace>/bin/okc \
OKC_GPU_SLOT=<primary-workspace>/local-notes/gpu-slot.local.json \
$PY -B cycle2_check.py poll P2-06-p039-unchanged-recheck
```

These commands are the historical executed request, not instructions to resend.
The wrapper reserved the request durably before its sole check call. Rules version,
existing authentication, daily quota, exact committed bytes, passing same-source
CPU evidence and the free serialized slot were checked before dispatch.

## Actual result

The service returned **`status=duplicate`**, `duplicate_of=d5636ac8d1404abd`,
`result=null`. The new ID does not represent a new GPU measurement. No source
change, comment nonce, retry, ranked submission or duplicate-bypass was attempted.

Quota before and after: Projection checks **3/20 used, 17 remaining**, scored
**0/10 used**. Reset is 00:00 UTC; these values were observed on October 25 at
18:47:35 and 18:48:29 UTC, respectively. Six cycle-two request records now exist,
but only the original five have non-duplicate result measurements; this duplicate
consumed zero additional platform quota. Historical user-count fields are not a
current quota override. Every journaled request is terminal, with no uncertainty.

**There are no new per-shape timings, score, correctness cases or confidence
interval.** A new comparison would be false. P039 retains its original 9+30 pass,
score and limitations. Its historical 30.1058% router-m1 normalized gain does not
become more certain because the same source was submitted again.

Sanitized receipt: `results/remote-P2-06-p039-unchanged-recheck.json`. Raw account
fields/local paths remain in ignored local files. Selected source and the original
receipt/30-row CSV remain unchanged. The shared slot was returned to terminal.

## Local verification and useful preparation

Twenty-one related tests passed: six P039 router tests, six P041 receipt/byte-model
tests and nine worker mock/static tests. After adding an explicit outlier B state
to the unexecuted ABA worker, its nine tests passed again. No local CUDA ran.

```sh
$PY -B -m unittest tests.test_router_m1_fp32_gemv \
  tests.test_projection_p041_plan tests.test_projection_p041_probe -q
```

The first sandboxed read-only CLI calls failed DNS resolution. A normal network
permission escalation succeeded; no upload occurred during those failed reads.
One local read-capture failed because the ignored directory was absent; it was
created and the harmless reads repeated. The check itself was sent exactly once
and completed without a denial, timeout, or uncertain outcome.

P041 provides a reproducible evidence model and unexecuted five-case diagnostic
worker. It identifies more useful multi-load opportunities than another m256-only
knob, but does not justify inventing a candidate or claiming a measured hardware
bottleneck. No second check is part of this request. A future changed candidate
needs its own precise hypothesis and local preflight before an official check.
