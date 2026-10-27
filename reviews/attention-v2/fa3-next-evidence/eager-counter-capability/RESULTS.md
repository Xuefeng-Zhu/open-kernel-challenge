# Eager counter capability: CLI setup failure before target launch

One standalone attempt ran in app `ap-fGzHaSj5oSgilv71oJ8chT`, accepted
2026-10-27 04:11:11 UTC. Frozen source commit:
`757e0df3f256fdb4e56a5e18338e2fdbc0f1c207`.
The raw result is `capability_not_established`, `passed=false` and child exit 1.
No retry, permission change, candidate edit or FA3 profiling is part of this result.

## Observed phases

| Phase | Exact observed outcome |
|---|---|
| One fixed metric query | NCU exited 0 after 1.479860 s; its H200 metric list includes `sm__cycles_elapsed.avg`, type Counter, unit cycle. |
| One profiling command invocation | NCU exited 1 after 0.051463 s, rejecting an unsupported CLI option before launching the child. |
| Target child | No `TINY_PROGRESS_JSON` marker was emitted; even `child_started` was not reached. |
| Pre/measured/post correctness | Not executed. |
| Target kernels profiled / metric samples | Zero / zero. |
| Counter permissions or capability | Still unknown. No permission-denial message, timeout or hardware result was observed. |

The complete profiling stdout is:

```
==ERROR== unrecognised option '--units'. Use --help for further details.
```

The attempted command requested kernel replay, application-only process scope,
the unique `okc_counter_gate` filter, launch-skip 1 / launch-count 1 and the single
queried metric. It also included the invalid pair `--units base`. The prelaunch
CPU tests and review missed that CLI grammar defect. This setup failure does
not test whether the deterministic eager kernel is correct, whether kernel
replay can complete, or whether GPU hardware counters are accessible.

The raw `hardware_counter_sessions=1` field counts an attempted profiler command.
It must not be relabeled as one successful counter session: actual target starts
and actual profiled kernels are both zero. Raw fields and frozen code are
preserved without post-result edits.

## Supported next local correction

The official NVIDIA Nsight Compute **2025.3** archived CLI reference documents
`--print-units base`, and also states that CSV output already implies base units.
Thus a later separately frozen minimal correction can remove `--units base`
entirely, or use the documented `--print-units base`. This is a documentation
check, not a successful run of a corrected command. No replacement was executed
here. [NVIDIA 2025.3 Console Output reference](https://archive.docs.nvidia.com/nsight-compute/2025.3/NsightComputeCli/index.html#console-output)

Keep the same tiny workload, one metric, kernel replay, correctness gates and
short timeout in any later attempt. No baseline parser, CUDA graph, NVTX,
application replay, real Attention kernel, additional metric or security change
is justified by this failure. A supported metric query alone does not prove
permission to collect it from a running kernel.

## Closure and costs

The explicit app stop request returned exit 0 at 04:11:17 UTC. The controller
observed stopped/tasks 0 at 04:11:22 UTC; a separate read-only receipt again
verified this exact app stopped/tasks 0 at **04:12:35 UTC**, with every listed
Modal app stopped and zero tasks. The app's recorded lifetime was 04:11:09 to
04:11:19 UTC. All own command process groups had terminated.

The remote entry took 1.685607 s; the internal worker reported 1.541254 s.
The maximum free-credit reserve was $0.32, and the controller's conservative
free-credit floor was $27.41. Spend limit and billed spending were $0. At the
independent cleanup cutoff, the account metered total remained $2.59 and its
unrounded ephemeral breakdown remained $2.58797306, both unchanged from the
admission receipt. The observed account delta is therefore $0 **at that cutoff**;
metering may be delayed, so this is not a zero-cost or final per-app invoice
claim. No credential action was performed.

Selected Attention candidate SHA256 remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
this tiny diagnostic does not create a candidate promotion or score claim.

## Evidence

- `result.json`: raw summary, complete bounded query/profiler stdout/stderr,
  exact argument lists, command timings, markers and source hashes.
- `independent-cleanup.json`: independent exact-app/task state and billing cutoff.
- Controller launch receipt: source commit, existing-auth verification,
  paid/free-credit admission, explicit stop and initial cleanup observation.
- `REVIEW.md`: prelaunch scope/protocol review, preserved as history; the CLI
  incompatibility it missed is recorded above rather than rewriting that review.
