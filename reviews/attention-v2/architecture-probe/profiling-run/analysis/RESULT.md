# Fixed B32 activity and counter evidence

Frozen execution source commit `e26bb0cf291dfdbfc6e70065b6d121bf315423ca`.
Activity evidence admitted: `True`. Four pre/post oracle checks passed: `True`.

This is one instrumented CPU/CUDA activity session with one unchanged FA3 and one unchanged BF16-custom graph replay. Durations below describe activity layout; they are not unprofiled latency and must not be subtracted from earlier paired benchmark results.

| Arm | GPU activity union us | GPU activity span us | CPU annotation span us | Instrumented graph-launch API union us |
|---|---:|---:|---:|---:|
| fa3 | 393.536 | 396.288 | 512.030 | 45.430 |
| bf16_custom | 589.089 | 589.697 | 630.000 | 14.263 |

CPU annotations include completion synchronization and profiler overhead. API event durations are themselves instrumented; they do not establish ordinary launch overhead. GPU span can include gaps, while union avoids double-counting overlaps.

| Arm | Observed stage | Events | Interval union us |
|---|---|---:|---:|
| fa3 | qk_norm | 1 | 2.304 |
| fa3 | fp8_conversion | 3 | 5.440 |
| fa3 | kv_cache_store | 1 | 1.536 |
| fa3 | length_schedule_prepare | 1 | 3.712 |
| fa3 | fa3_mainloop | 1 | 378.752 |
| fa3 | fa3_combine | 1 | 3.296 |
| fa3 | output_norm_gate | 1 | 2.144 |
| bf16_custom | other | 1 | 0.672 |
| bf16_custom | integrated_producer_and_merge | 1 | 588.417 |

FA3 stage attribution uses complete observed kernel names. FP8 conversion names alone do not identify query versus cache conversion. The custom kernel includes producer and merge; their internal costs are unresolved. Memset/fill destination attribution to ticket storage remains unproven. Exact event IDs, streams, pairwise stage separations and full names are retained in `analysis.json` and the raw trace.

| Arm | Adjacent recorded stages | Pairwise separation us |
|---|---|---:|
| fa3 | qk_norm → fp8_conversion | +0.416 |
| fa3 | fp8_conversion → kv_cache_store | +0.832 |
| fa3 | kv_cache_store → fp8_conversion | +0.384 |
| fa3 | fp8_conversion → length_schedule_prepare | +0.576 |
| fa3 | length_schedule_prepare → fa3_mainloop | -2.752 |
| fa3 | fa3_mainloop → fa3_combine | -0.896 |
| fa3 | fa3_combine → output_norm_gate | +0.160 |
| bf16_custom | other → integrated_producer_and_merge | +0.608 |

Negative separation means overlapping recorded intervals; positive separation is only pairwise and an earlier event may cover it. These are not inferred SM idle times, causal handoff delays or warp stalls. The complete union-frontier uncovered intervals are retained separately. In particular, overlapping schedule/mainloop/combine intervals prevent summing their durations into wall latency.

Counter branch: `unknown_or_not_collected:counter_session_timeout`.

fa3 was the only attempted main-kernel NCU session and timed out at 60 seconds. No second arm was attempted. The timeout path returned no stdout/stderr, target validation or raw NCU report; pass/replay progress and permission status cannot be determined. This is a timeout, not a recorded permission denial. No retry or extra experiment was performed.

Achieved DRAM throughput, tensor utilization, warp stalls, achieved residency and per-CTA service times remain unknown. Activity stage duration is insufficient to call the kernel bandwidth-, tensor- or occupancy-limited.

Exact source and raw artifact hashes are in `analysis.json`; deliverable hashes are in `manifest.json`. No model, candidate route, benchmark or official result was changed by this local postprocessor.
