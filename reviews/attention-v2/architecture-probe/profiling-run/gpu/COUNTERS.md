# Bounded existing-NCU counter branch

Called only after the single activity session and all four unchanged-oracle
checks pass. This module does not launch a Modal job or install/modify tools.

`run_counters(root, dest, deadline_monotonic)` returns status,
`sessions_attempted` (0..2), actual queried metric selections, unavailable
semantics and per-arm records. At most eight exact supported metrics are
selected by one narrow base/suffix query. No PM sampling or exact CTA trace is
claimed. Counter duration is instrumented diagnostic data, not a benchmark.

Only two possible mainloop collection sessions, FA3 then BF16 custom, run
serially. Each has at most60 seconds and must fit the same165-second worker
deadline with cleanup margin. NCU application replay reconstructs deterministic
seed135701 B32/L8/page1/80K inputs and private ticket state. A unique NVTX range
contains one graph replay; strict/grid matching and exact mainloop filters
exclude warmups/preparation. Cache/clock control are none; no privileges or
driver settings change. Each application pass checks output/new KV and
immutable/untouched cache rows before and after the profiled replay.

Any permission denial, including ERR_NVGPUCTRPERM or
CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, stops the counter branch before a second
arm. Other failures/timeouts also stop it. Process groups are killed on timeout
or interruption. No fallback access route or retry is implemented. A valid
activity result is independent of counter availability.

Only four optional files are promoted from temporary scratch to the shared
8MiB artifact budget: `fa3.ncu-rep`, `bf16_custom.ncu-rep`,
`fa3-counter-check.json`, `bf16_custom-counter-check.json`. Query errors and
per-arm raw CSV/log output are retained in the bounded summary. Parsed metrics
retain explicit units; missing/unparsed fields are marked partial. Binary
reports are checked for size before reading or promotion.

Six pure-local tests cover metric selection, replay/permission policy,
first-denial stopping, timeout process cleanup, CSV layouts and unchanged
pre/post correctness checks. These tests execute no CUDA, Nsight or network.
