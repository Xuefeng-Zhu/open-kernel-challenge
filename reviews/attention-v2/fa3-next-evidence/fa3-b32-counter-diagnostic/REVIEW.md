# Independent prelaunch review: PASS

Reviewer: `/root/fa3_counter_review`. Local source/tests only; no service,
authentication, browser, GPU, Git mutation or candidate action was performed.
No remaining prelaunch code blocker was identified for this one bounded batch.

Reviewed immutable-source digests:

| File | SHA256 |
|---|---|
| `worker.py` | `9b65cfc93627bc5211b92e96b1f73e3eacbf6099747f05927a6b6cb2806cc056` |
| `controller.py` | `3144b5f5a44861c684a25e4bcdf026051cbf05dea6a4dea2ec3c95d0d791b11e` |
| `csv_helpers.py` | `a00f901febe900e4ed9f48680b990acf813f0c0d76680b1c798a664f017759df` |

The fixed case is B32/L8/full, TP8, D64, page1, context80000, agentic lengths,
coherent inputs, unit scales and seed135741. Source SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
dispatches through the actual `kernel_fn` first branch to
`_fa3_full_verify_fused`; it retains FP8 external Q/K/V, BF16 output, no RoPE,
metadataNone and automatic pack/split inputs. The rejected L1 producer is absent.

The protocol binds selected/reference/check/helper files before importing them.
Original oracles, output/new-KV byte invariance, full non-new cache ownership,
stable graph addresses, three cold whole-call graph samples before activity,
three after activity, and original-oracle plus three graph samples after NCU
are checked separately from instrumented measurements. The post-NCU confirmation
occurs before interpreting CSV or returning a counter failure. Host/GPU graph
attribution uses launch correlation and matching GPU annotation, avoiding the
previous cross-clock containment bug.

Installed NCU version/help validates exact flags before four fixed support
queries and at most one combined four-metric session. Kernel replay,
application-only process scope, no NVTX/application replay, unique FA3 filter,
skip1/count1, and exactly three eager selected calls are retained. The original
oracle uses einsum and does not consume the FA3 filter. Pre/measured/post
correctness and immutable-cache checks are required. The CSV reader preserves
the full symbol, geometry, units and finite numeric values, requires one actual
kernel row, and records rather than clips percentage-over-peak warnings.

Independent local test discovery passed **35 tests** after the worker's final
post-NCU changes. These include a real saved installed-NCU CSV regression,
binding its 13,518-byte fixture to SHA256
`176beb19e4618b1c52f6ba737ceaf0d6f5353ea8574a823276530a667763ebef`,
plus long comma-containing symbols, exact help/command flags, source/route,
transport and budget contracts. Final root test run also reports35 PASS.

Controller scope is one H200, 2 CPU, 8GiB, one container, zero retries,
190s worker/205s child/220s function/20s startup/250s watchdog, at most$0.70
verified free-credit reserve and$0 paid. Credentials remain local; the remote
entry is closure-free and only the explicit source/license allowlist is uploaded.
Raw partial reports/trace hashes are preserved and explicit app shutdown follows.
Fresh live admission and independent stopped/tasks0 verification remain runtime
requirements; this review itself proves neither permission nor runtime success.

Interpretation gate: exact full-name/grid/block, complete fixture fingerprint,
output/new-KV bytes and mapped-library correspondence are necessary evidence
for the eager proxy. Even a match does not prove opaque split/capture-state
equality. A mismatch must remain inconclusive. Counters cover the mainloop,
not the preparation/scheduler/combine/gate or whole call. Tensor activity is not
useful FLOP throughput; DRAM utilization and long-scoreboard stalls alone do not
prove bandwidth causation; measured active warps do not make register pressure
the cause. No candidate change or gain is justified by static register counts.
The four counters support a conditional hypothesis or an honest inconclusive
result, followed by a distinct discriminating measurement.
