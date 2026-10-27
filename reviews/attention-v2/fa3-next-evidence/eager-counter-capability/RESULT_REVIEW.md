# Raw-result audit: setup failure, capability unknown

Reviewer: `/root/tiny_counter_review`. This reviewer inspected saved local raw
JSON/receipts and the official NVIDIA 2025.3 CLI documentation. No service,
authentication, GPU, browser, Git mutation, frozen-source edit or retry was
performed. This reviewer authored `RESULTS.md` from that raw evidence; this is
an independent audit of the run, not a second independent review of the prose.

## Findings

1. Metric query exit 0 and its output prove that the named H200 metric is listed.
   They do not establish running-kernel hardware-counter access.
2. Exactly one profiling command was invoked. It exited 1 after 0.051463 s with
   the explicit unsupported-option error `--units`. Its stdout contains no
   child marker, connection marker, profiling progress or metric value.
3. The child was not launched, pre/measured/post correctness did not execute,
   target kernel count is zero, and counter samples are zero. The raw attempt
   counter `hardware_counter_sessions=1` must be read narrowly as command attempts.
4. This is a CLI compatibility/setup failure, not a permission denial, replay
   timeout, GPU numerical failure or measured bottleneck. Capability is unknown.
5. Worker/controller/helper hashes remain bound to the frozen source. The raw
   failed result is preserved. No corrected command has been executed.
6. Exact-app shutdown is confirmed independently by stopped/tasks 0. Billing
   remains $0 paid; the observed zero account delta has a delayed-metering caveat.
7. The prelaunch tests/review did not verify the installed NCU CLI grammar and
   missed `--units`. The official release-family reference documents
   `--print-units base`; CSV already selects base units. Removing the invalid
   pair is sufficient as a future source correction, without changing the
   experiment's workload, metric or replay design. This audit does not authorize
   another launch or report that the correction works at runtime.

Source: [NVIDIA Nsight Compute 2025.3 CLI, Console Output](https://archive.docs.nvidia.com/nsight-compute/2025.3/NsightComputeCli/index.html#console-output),
accessed 2026-10-27. The release-family manual is primary-source documentation;
it is not a fresh successful `--help` invocation in the exact remote executable.

## Audited artifacts

| File | SHA256 |
|---|---|
| `result.json` | `e9837e8758f3962f2d37e72c0e689eeface7354d2b30a1eb5071e199a3871585` |
| `independent-cleanup.json` | `0c9789b6cf94993796ad861ccd632bc628eeebef38094c0bcb1f41e250f798f6` |
| `worker.py` | `1c41881f3aba41325ea8cc2e45813a78567b24ca07f19594bcd827b99aa3f538` |
| `controller.py` | `0cd1546d4bbf452e8b2fc18cb8186f17a2018fbf4a6fab1dbadf8865fae55ce1` |
| `RESULTS.md` | `29f33dcc81f3e5f392e6171a5e86efe7e460295b5db12fcf65f79b1491001d9b` |

The query stdout SHA256 is
`7ce2e9853b86c23f1c2ed9aaa007eee09fb165d3a2780c304c1301efa0a158d1`.
The complete 73-byte profiler stdout SHA256 is
`a0491e24b9d4e2b1fa5b9c1677b75ad3db2b17cf5f089f0d53f0d0d678a04c63`.
Both stderr streams are empty. The final scope is documentation-only closure of
this known failed setup; no counter capability or Attention performance claim.

Root formatting check: removed only surplus blank EOF lines from RESULTS.md; all non-whitespace bytes are unchanged. The report hash above reflects this formatting correction.
