# Eager single-counter capability: PASS

App `ap-utcDysQOu5e1m3MSYhEZR4` ran one bounded diagnostic from frozen commit
`0922050cf384be1bc3ec1bd7db3046846e9dcd3e`. The worker completed successfully
in 7.149576 s; its NCU profiling command completed in 6.086596 s, exit 0.
This establishes the tested tiny eager kernel/metric capability only. It does
not diagnose FA3, the rejected producer, memory traffic, stalls or performance.

The installed NCU reports version **2025.3.1.0**, build 36398880. Its actual
34,543-byte `--help` was captured and hash-bound before the metric query or
profiling command. All exact NCU flags used by both commands were advertised;
the invalid `--units base` pair was absent. The help hash is
`ac75d8e8b9801ec2b43d18ffa925b9d9e6917ea5ed0c47634af7d67ae1eae7c9`.
The fixed query confirmed `sm__cycles_elapsed.avg` as an H200 Counter, unit cycle.

## Actual counter evidence

The NCU command requested one metric, kernel replay, application-only process
scope, the unique `okc_counter_gate` filter, launch-skip 1 and launch-count 1.
There was no CUDA graph, NVTX or application replay. No security, permission,
clock, credential or candidate setting was changed.

Independent parsing of the raw CSV found exactly one data row:

| Field | Observed value |
|---|---|
| Row ID / kernel | 0 / `okc_counter_gate` |
| Exact requested metric column | `sm__cycles_elapsed.avg` |
| Value / unit | **5,541.35 cycle** |
| Block / grid | `(128,1,1)` / `(1,1,1)` |
| Command result | Complete, exit 0 |

NCU includes automatic device/launch/derived fields in its wide CSV; they are not
additional requested metric experiments. The exact requested metric is column 259
of 266 columns, its unit row says `cycle`, and its sole data row contains the
value above. This independently agrees with the worker parser, rather than
accepting a metric name, query list or success marker as a counter sample.

The raw stdout includes both profiler connection and disconnection messages.
All three ordered eager correctness markers—`pre`, `measured`, `post`—are true.
The middle matched kernel is profiled; each call verifies the same deterministic
FP32 computation exactly. No timeout or permission-denial message occurred.
The two older application-replay/graph timeouts therefore do not establish that
hardware counters are globally unavailable. This result does not prove that
every metric, replay mode or real attention kernel can be profiled successfully.

## Closure and cost limits

The explicit stop request succeeded at 04:37:41 UTC. A separate read-only check
at **04:38:24 UTC** verified this exact app **stopped, tasks0**, with all listed
apps stopped and zero tasks. Recorded app lifetime was 04:37:28–04:37:43 UTC.
The maximum free-credit reserve was $0.32; the conservative admission floor was
$27.40680620. Actual paid spending remained **$0**.

The independent cutoff still showed account usage $2.59 and ephemeral breakdown
$2.59319380, unchanged from admission. This zero observed delta may be metering
delay; it is not a zero-cost or final per-app invoice claim. No further GPU job,
candidate rewrite, official check or ranked submission belongs to this result.

## Required geometry check for the next diagnostic

The current M64N8 source only replaces its admitted **L1** custom-producer routes.
On full B32/L1, the selected source calls `_fused` and the rejected source calls
`_m64n8_fused`, with their existing merge ABI. On full B32/**L8**, both sources
take the identical `_fa3_full_verify_fused` route: their `kernel_fn` and that helper
are AST-identical. Profiling those two L8 files as separate algorithms would
compare the same route. Comparing L1 producer latency with L8 FA3 latency would
also change the workload and is not a matched performance contrast.

Any next batch must preserve the actual selected whole call as primary control,
prove dispatch before launch, and keep geometry/input/precision identical within
each comparison. If FA3 is wanted as an additional L1 diagnostic arm, it requires
an explicitly reviewed, labeled adapter; it is not the selected L1 route.

Selected source SHA256 remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
rejected M64N8 source SHA256 is
`96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970`.
Both implementations and the existing ranked entry are unchanged.
