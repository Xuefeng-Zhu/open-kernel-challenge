# Independent prelaunch review

Reviewer: `/root/baseline_diagnostic_review`. **PASS for the bounded diagnostic**, not runtime correctness, counter availability, an optimization, or promotion. No service, credential, browser, GPU or Git operation was performed by this reviewer.

The scope is exactly the unchanged selected source on full B16/B32, L8, TP8, D64, page1, ctx80000, agentic/coherent inputs, seed135741. Independent local discovery ran **34 CPU tests, all passing**. Selected source remains SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The timed runner captures the original selected `kernel_fn`, keeps input/output addresses fixed, restores the same fixture and evicts L2 outside event intervals, and records three unprofiled whole-call samples before and after one activity session. Original-oracle checks precede measurement and follow activity. Every replay checks output/new-KV bytes, immutable inputs and all non-new cache rows. Activity ranges fence completion and exclude preceding restore/eviction. Full symbols/resources and raw trace are retained; interval unions avoid double-counting overlapping activity. Instrumented durations are not benchmark timings.

The separate untimed auxiliary adapter copies the selected helper AST and namespace; it does not replace any loaded function or enter timed calls. Native argument AST is identical except the public Python return flag. The live wrapper must accept that flag and prove it is absent from native arguments. Final output/new-KV and entire caches must byte-match the original runner. Opaque returns, template booleans, PackGQA, split counts and internal QK/PV types remain unknown without supported semantics. Logical 64 packed rows are not resolved dispatch. File hashes/mappings identify this Modal image, not the judge binary.

The optional counter branch runs only after core checks and only with at least50 seconds remaining. A five-second query must confirm the one fixed `sm__cycles_elapsed.avg` metric. One deterministic eager kernel is then profiled with kernel replay, an explicit name filter, application-only processes, launch-count1 and unchanged clocks. Only exit0, a positive numeric hardware-metric row and child correctness permit one unchanged FA3 target. Each profile has a20-second process-group timeout. No CUDA graph, NVTX or application replay, metric fallback, install, security change or retry is included. Full bounded stdout/stderr and hashes are embedded. A timeout remains a timeout; only an explicit permission diagnostic is classified as denial. [NVIDIA CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) supports the query, replay, filtering and clock-control options.

The controller binds a committed allowlist, existing protected authentication and fresh free-credit/$0-paid receipts, limits execution to one H200/container/invocation with zero retries, validates raw/compressed trace hashes, explicitly stops the exact app, and records shutdown and billing observations. The initial output-directory collision was fixed by putting child logs outside the worker-owned directory. Actual shutdown must still be checked independently after execution.

Reviewed SHA256 values:

| File | SHA256 |
|---|---|
| `worker.py` | `3246f1905812f591c02299d952587c7648cefad228adcabc4c9130b2e47fce40` |
| `controller.py` | `a94656000581bf099409beb6a5b1fefb1e789038341928c1ce1ddc5b1686eeec` |
| `counter_gate.py` | `d984cbd44691cfead1c889cbb2ae8b0bde2e11e0a1d703bbdd78a7ecf1b30637` |
| `test_worker.py` | `3b18e9b2f4b642ce6ed427d3723de5a54c0b6160b60155dd8b9dac744192a8a5` |
| `test_controller.py` | `acefc07c15271667e34ba690d6f37103522a0118a3970712cec956c8131b9d55` |
| `test_counter_gate.py` | `1455917d51f6e91f9af21add8f8e83affa7e86535d4b2ea284d1b607868f09a7` |
| `CARD.md` | `268d05c1e241382deead959e6041e024be6e438b4fe7854c65be6dfdff3c5fe5` |
| Existing `dispatch-identity-cpu/worker.py` | `29e4f87f294142a38f926a7bf6f57e8032d615be7ab0da4a17e2f0962a50ec92` |

Runtime data and the counter gate may still fail. Preserve that result and partial evidence, stop resources, and do not manufacture a successful dispatch or bottleneck conclusion.
