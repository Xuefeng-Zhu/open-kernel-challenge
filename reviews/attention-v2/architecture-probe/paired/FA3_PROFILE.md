# Actual FA3 B32 verify uses a named dynamic persistent scheduler

The new bounded diagnostic captured complete CUDA event names from one
unchanged FA3 full-attention B32/L8/D64/page1 graph replay. This was a separate
profiler call after correctness and paired timing, outside the measured samples.
The report contains eight unique kernel names (2,829 UTF-8 bytes), with no
omitted names and no individual name truncation.

The mainloop name literally contains:

```text
flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true>
```

It is the scheduler template argument of `flash::FlashAttnFwdSm90`, nested in
`flash::enable_sm90_or_later` and `cutlass::device_kernel`. The same event name
contains `cutlass::float_e4m3_t`, `float`, and
`cutlass::arch::Sm90`. These are literal demangled type tokens, not a native
instruction inspection or proof of a particular accumulation implementation.

The same replay also records
`flash::prepare_varlen_num_blocks_kernel<2, true>` and
`flash::FlashAttnFwdCombine<...>`, alongside query normalization, E4M3
conversion, KV-cache stores and output normalization/gating. Thus a proposed
new dynamic persistent scheduler is **not a new mechanism relative to this
actual FA3 B32 verify specialization**. Its assign/descriptor, gather and merge
costs must be evaluated against a baseline that already uses this scheduler.

This resolves the earlier local-audit limitation for the **named scheduler type
of this B32 call**. It does not resolve exact FA3 source/build identity, template
boolean meanings, internal dispatch conditions, scheduler behavior under other
inputs, B16's specialization, or native instructions/resources. No source
matching is claimed: this offline audit used the returned complete event names
and existing frozen worker only. The worker's conservative
`actual_scheduler_verified=false` field is left unchanged; the observed type
name is reported here with its precise limits.

`result.json` is the immutable primary artifact. Its
`fa3_profiler.kernel_names` field retains the full event strings; the observation
is not reconstructed from the historical truncated official profile. No network,
browser, GPU call, result edit, or analysis-file edit was performed for this
offline follow-up.

Provenance SHA256:

```text
result.json:
b55b87101004742d43d33272a632250d094a2345f1a3e37fce64589f76d931e9
Full mainloop event name (UTF-8, no newline):
33239a4fe9825c98df5c44acd8bc412ee2c441fab7b704ffa6124fb68de96711
```
