# Independent eager counter capability gate

One fixed query for `sm__cycles_elapsed.avg`, followed by one NCU session on a
deterministic eager Triton kernel. Kernel replay, application-only process scope,
unique kernel filter, skip the first matched launch and capture only the second.
Three ordered eager calls check the same exact computation before, during and
after collection. No CUDA graph, NVTX, application replay, FA3 or other diagnostic.
No clocks, permissions, security settings, credentials or candidate code change.

Preserve query and session argv, stdout/stderr, progress markers and partial
status. Query deadline 5s, session deadline 30s; worker65s, child75s, function85s,
startup20s and host watchdog110s. One H200, 2 CPU, 8GiB host memory, no retry.
Reserve at most $0.32 of verified free credit; paid cap and actual billed spend
must remain $0. Explicit app stop and independent zero-task verification follow.

Capability PASS requires the queried metric, exactly one positive numeric counter,
exit0, and all three ordered correctness markers. A timeout or missing output is
inconclusive about permission; only an actual permission error establishes denial.
A PASS establishes this small metric collection, not a production-kernel bottleneck.
