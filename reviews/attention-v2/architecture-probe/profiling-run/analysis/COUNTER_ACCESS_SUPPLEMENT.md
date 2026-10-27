# Separate final counter-access supplement

Status: `timeout_partial_logs_preserved`. This uses the remaining counter authorization separately from the original activity run and first 60-second NCU timeout. Recorded counter sessions here: 1. No new activity trace or algorithm change was made.

Frozen supplemental execution commit `052665dd2496f12952d684ee6fd5a4b0e1075228`.
Raw supplement summary SHA256 `e9a7d7b60f4ef1b5726cdd8220868b849cdaf2fbb05e8f68cd0b846836bbb177`.
Original activity summary remains `a48846ac5a4a79de771822130eee0d46705fa5350c49fafff9508125d977ba15` and trace `e49b1677de2dfbc9fcca43e63924a026cd7c101321e2bee1131f87ee41355ffc`.

The one FA3 `dram__bytes_read.sum` session timed out after 55 seconds; retained stdout/stderr bytes are hash-verified. This does not establish permission denial or a performance result.

FA3 target pre/post oracle and exact post-profile output/new-KV validation passed: `True`. Workload validation does not establish successful hardware-counter collection.

stdout retained bytes: 272 of 272; truncated: `False`; byte/hash/text verification: `True`.

stderr retained bytes: 0 of 0; truncated: `False`; byte/hash/text verification: `True`.

Preserved diagnostic lines:

> ==WARNING== Note: Running with uncontrolled GPU caches. Profiling results may be inconsistent.
> ==WARNING== Note: Running with unmodified GPU clocks. If not controlled otherwise, profiling results may be inconsistent.
> ==PROF== Connected to process 29 (/usr/bin/python3.12)

A profiler connection message establishes connection to the child process; it does not establish permission to read hardware counters or a collected metric.

Explicit pass counts observed in logs: []. Requested session count is not replay/pass count.

This supplemental session changed both the requested metric count and the parent process/CUPTI context arrangement. Even successful access cannot isolate the cause of the first session's timeout. The authorized totals are now one activity session and at most two counter sessions; no further collection follows from this analyzer.

DRAM bytes alone do not establish bandwidth. Tensor utilization, stall causes, residency and exact CTA tail behavior remain unknown. No permission bypass, further collection or candidate/model change was performed by this analyzer.

Complete retained log text/base64 hashes, source hashes and raw report checks are in `counter-access-analysis.json`; deliverable hashes are in `counter-access-manifest.json`.
