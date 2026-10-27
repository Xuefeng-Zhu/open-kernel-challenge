# FA3 specialization: what the saved evidence establishes

This is the historical audit made before the current paired run. The subsequent
[actual B32 profiler evidence](../paired/FA3_PROFILE.md) resolves its named
scheduler type as `VarlenDynamicPersistentTileScheduler` for B32/L8/D64/page1.
Exact build/source identity, template-flag meanings and B16's scheduler remain
unverified. The earlier observations and their limits below are preserved.

This is a local, read-only audit of existing receipts. No browser, network,
candidate execution, installation, or GPU call was made. The extracted evidence
and complete source hashes are in `fa3-audit.json`.

The exact installed FA3 persistent scheduler specialization is **not verified**.
The Mac virtual environment has no `sgl_kernel` package, and no FA3 Hopper
implementation or scheduler source was found in the original `local-notes`.
The historical H200 native diagnostic recorded `sglang=0.0.0.dev0` and
`sgl-kernel=null`; this is an unsuccessful distribution-metadata lookup, not
proof that the package was absent on that GPU image. Its custom Triton native
artifacts do not identify FA3's binary. Saved upstream links and explanations
are useful hypotheses, but must not be described as installed-code evidence.

The A2-13 official receipt does establish actual length-derived preparation
inside the existing full-verify path:

| Path | Observed preparation kernel | Activity time | Mainloop activity | Combine activity |
|---|---|---:|---:|---:|
| full B16/L8/D64/page1 | `prepare_varlen_num_blocks_kernel<1, true>` | 3.1 us | 189.8 us | 3.7 us |
| full B32/L8/D64/page1 | `prepare_varlen_num_blocks_kernel<2, true>` | 3.6 us | 373.9 us | 3.7 us |

These are instrumented activity durations from historical official receipt
`4e83463a594e6eb523b8654743dc1f28794f4f409c0f707002dbe3d94b01c084`.
They are not paired latency measurements, and cannot be subtracted directly
from a complete call. The mainloop name is truncated after
`cutlass::device_kernel<flash::enable_sm90_or_later<f...`, so its scheduler
template cannot be recovered from this report. The `<1,true>/<2,true>`
preparation names alone do not identify a dynamic persistent scheduler type.

The public generator returns FA3 scheduler metadata only for full **decode**
steps, constructed outside the call from current lengths. Full verify receives
`None`. A2-13 full B16/B32 verify dispatches to `_fa3_full_verify`, passing
live `cache_seqlens`, `page_table`, and query offsets. Its current full B32
decode dispatches to the custom Triton path instead, so that path does not
consume the supplied FA3 schedule. These are source facts in the hashed
candidate and public generator, not statements about hidden judge metadata.

The next evidence required before any novelty claim is a bounded inspection
of the existing H200 image: resolve the imported `sgl_kernel.flash_attn` wrapper
and shared objects, record their hashes/build identifiers, inspect defaults and
dispatch, and obtain full (untruncated) kernel specialization names for the
actual D64/page1 L8 calls. If source/build identity remains unavailable, report
the binary hashes and observed names with that limitation. FA3 already has
persistent scheduling mechanisms upstream; a fresh scheduler must be evaluated
against the actual specialization, with assignment and live metadata processing
inside every timed call. Cross-layer taskmap reuse is not a valid comparison.
