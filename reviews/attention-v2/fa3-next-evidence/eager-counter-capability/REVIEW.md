# Independent prelaunch review: PASS

Reviewer: `/root/tiny_counter_review`. Read-only review of worker/controller and
independent local test execution; no service, credential, GPU, browser or Git
operation was performed by this reviewer. All 17 local CPU protocol tests pass.
GPU counter access and device correctness remain untested until actual execution.

## Reviewed scope

One fixed supported-metric query (`sm__cycles_elapsed.avg`), followed directly
by one NCU session targeting the deterministic eager `okc_counter_gate` kernel.
No FA3, baseline activity parser, CUDA graph, NVTX or application replay runs.
`--replay-mode kernel`, `--target-processes application-only`, unique demangled
kernel-name filtering, `--launch-skip 1` and `--launch-count 1` select the middle
of three matching eager calls. All three calls verify deterministic FP32 integer
arithmetic exactly. Ordered pre/measured/post pass markers, a single finite
positive metric value, completed command and exit 0 are all required for PASS.
A valid metric only establishes this tiny kernel/metric capability, not any real
Attention bottleneck or speedup.

The child emits distinct startup, helper verification, vendor import, CUDA init
and per-call progress markers. Query and counter command logs are bounded,
retained with hashes, and classified separately from explicit
`ERR_NVGPUCTRPERM`. Timeout alone does not establish a permission denial.
The helper terminates its own process group on timeout/output overflow.
The whole worker/child/function/app watchdog bounds are 65/75/85/110 seconds,
with 20 s startup and zero retries. Outer app shutdown and later independent tasks 0
verification remain required even if the child is killed.

Credentials stay in the protected existing local config; token-info stdout is
captured and only sanitized workspace/UTC expiry is reported. The explicit
remote payload is worker, frozen counter helper and license notices only.
Fresh free-credit/spend-cap receipts, live rates, existing-auth validity, no active
app and a source-bound serialized slot are checked before admission. The fixed
$0.32 reserve must cover the live doubled 110 s resource model plus $0.02; billed
spend and paid cap must remain $0. No credential creation/renewal, security change,
clock change, installation, official check or candidate change is included.

The output-directory ownership is consistent: the remote entry writes its log
outside the worker-owned result directory, which the worker creates exclusively.
Controller validation uses the actual worker fields `fa3_kernel_executed` and
`selected_candidate_changed`; their prior naming mismatch is fixed.
Partial/raw summaries are saved before completion claims are validated.

## Exact reviewed files

| File | SHA256 |
|---|---|
| `worker.py` | `1c41881f3aba41325ea8cc2e45813a78567b24ca07f19594bcd827b99aa3f538` |
| `controller.py` | `0cd1546d4bbf452e8b2fc18cb8186f17a2018fbf4a6fab1dbadf8865fae55ce1` |
| `test_worker.py` | `68879e6eea5abe85c429294e9cb70d9431c8453f1ce2e3de34cc1e4a062ae38d` |
| `test_controller.py` | `43b6f5f02fd92cad43a3bd99aec2012ff485fcbdc114e9a5779af17e6a1d50ca` |

Frozen shared helper SHA256:
`d984cbd44691cfead1c889cbb2ae8b0bde2e11e0a1d703bbdd78a7ecf1b30637`.
No remaining prelaunch code blocker was identified. This review does not claim
that a remote invocation occurred or that counter access has succeeded.
