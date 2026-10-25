# Sliding B16/B32 unsplit candidate: source-ready, installed behavior unverified

The actual baseline executes combine and scheduler preparation on both target
paths. A target-only `num_splits=1` override is a coherent single-variable
hypothesis, but the local record does **not** establish the exact installed API
or prove that the override removes combine. This audit admits a conditional
candidate for source review, not GPU execution or an official check.

## Exact candidate isolation

The baseline is the newly validated fusion source from private commit
`05c9cc08948c42fed9e09aae1cd2b41edf966c74`. Its SHA256 is
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
The conditional candidate SHA256 is
`586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf`.

Independent AST comparison of the actual files passes: after removing one
conditional unpacked keyword from the preserved fused-sliding FA3 call, their
entire ASTs match. The added argument is:

```python
**({'num_splits': 1} if batch in (16, 32) and length == 8 else {})
```

The existing entrypoint restricts this helper to sliding L8/window2048/RoPE500000.
The empty dictionary preserves the original omitted argument on B1 and nearby
unaffected shapes; always passing `0` there would instead assume an installed
default. No preparation, query/cache precision, causal/window expression,
softmax scale, descale, scheduler metadata, output gate or other FA3 call changes.
This does change attention decomposition and may change reduction order; exact
output-byte equality between algorithms is not assumed. Original official
numerical tolerances and self-consistent replay checks remain mandatory.

## Actual baseline evidence

The completed official fusion check `2f5f1fb54b48486d` contains these two target
profile records. They are executed-baseline evidence rather than upstream
predictions. The report truncates mainloop/combine template names.

| Sliding verify target | Mainloop activity | Combine activity | Scheduler preparation | Total activity / GPU busy |
|---|---:|---:|---:|---:|
| B16 | 10.5 µs | 4.2 µs | 1.8 µs, `prepare_varlen_num_blocks_kernel<1, false>` | 19.7 / 18.3 µs |
| B32 | 14.1 µs | 4.4 µs | 1.8 µs, `prepare_varlen_num_blocks_kernel<2, false>` | 24.2 / 22.7 µs |

Each record also contains the fused preparation and output gate. PDL/stream
overlap is reported: stage durations include waits and cannot be subtracted
from whole-call latency to predict a saving. The preparation template integers
are not established split counts. The baseline's exact split counts, grid,
packing and sliding scheduler specialization remain unknown from these records.

Source-bound extraction and original hashes are in `source-audit-evidence.json`.
Its official receipt SHA256 is
`46f9818a029a68995d6bad8b99c79a72f390ab09e464c01eb9a79b64bb160001`.
No new profiling, Modal job, official job, browser control or credential action
was performed during this audit.

## Fixed upstream semantics, not installed identity

At Dao-AILab commit `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, the public
KV-cache wrapper exposes automatic splitting as default0 and passes a positive
override through the API. This establishes the intended parameter contract,
not this image's installed callable. [Wrapper](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_attn_interface.py#L609-L640).

That revision's C++ API uses the requested positive split count, selects an
unsplit mainloop for1, and calls combine only above1. However, varlen batches
up to992 still allocate length-derived dynamic metadata independently of that
override. Thus setting1 does not itself remove scheduler preparation. It also
leaves the full causal/local attention domain and precision controls intact;
it changes partitioning, not permitted keys. [API selection and metadata](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_api.cpp#L887-L920),
[combine condition](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_api.cpp#L1051-L1073).

The launch code runs the varlen preparation whenever its metadata pointer is
present and no caller-provided schedule skips it. Verify supplies None, so
preparation is expected to persist under that implementation. Hopper varlen
work remains persistent; unsplitting need not reduce the launched grid to the
number of useful tiles. [Launch conditions](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_fwd_launch_template.h#L54-L67),
[metadata launch](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_fwd_launch_template.h#L137-L152).

## Installed identity gap and parallelism risk

Saved image inventory identifies SGLang `0.0.0.dev0`, metadata SHA256
`e345ccfcb7bc6df9fb5a55037f03dc75fa243770524a9f7f6cfef6e1ba6e7128`;
`sgl-kernel` distribution lookup is null. It saved neither the installed
`sgl_kernel.flash_attn` wrapper nor a FA3 extension/source hash. Bounded searches
of the local Mac caches and saved artifacts found no such source. Null package
metadata does not mean the module is absent: previous native calls succeeded.
This is an unavailable local identity, not an access denial.
The official profile and Modal inventory are different runtime environments;
do not infer a shared extension or wrapper hash from their similar kernel names.

One previous full-B32 trace captured a complete scheduler type with M64; the
fixed Dao revision's FP8/D64 tile selector instead contains M192. They are
different evidence scopes, not a source-to-binary match. Neither determines
the installed sliding specialization. [Fixed tile source](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/tile_size.h#L37-L49).

If packed GQA is retained and its M tile is at least64, L8×8 grouped queries
provide one useful query tile per KV head: B16/B32 yield32/64 logical tiles.
These are **not measured launched CTAs, occupancy or residency**. With132 SMs,
removing KV partitions may sacrifice useful parallel work and length balancing.
Packing, tile selection or persistent scheduling may also change implicitly
when split changes; record them rather than introducing another override.

## Fail-closed native admission after a separate slot grant

Before candidate invocation, record the actual callable's module/file, visible
signature and defaults (`inspect.signature(..., follow_wrapped=False)`), readable wrapper source hash, package metadata and
loaded FA3 extension path/hash. Require an explicit supported split parameter;
a generic `**kwargs` wrapper alone is not proof. Stop if its forwarding or
backend semantics cannot be identified safely. Do not install, renew credentials
or expand permissions to obtain that evidence.

Native admission also requires at least one relevant already-loaded extension
and complete SHA256s for every observed relevant extension, with no relevant
path omitted. The bounded inventory covers `sgl_kernel`, `flash_attn`,
`sgl_flash_attn` modules and relevant existing `torch.ops.loaded_libraries` paths.
Canonical aliases are deduplicated; full-file hashes stream within4GiB/file
and8GiB/total bounds, with the160s worker deadline unchanged and propagated.
Unavailable/lazy-provider identity fails closed before candidate calls; no
fallback or automatic second job is allowed. This verifies observed extension
files, not every transitive dependency or a source-to-binary equivalence claim.

After the identity gate, run unchanged-tolerance boundary, cache, causality and
fixed-address A→B→A checks. Then use one separate activity session over labeled baseline/candidate graph
replays for both targets. Retain full names, graph association and counts for
mainloop, combine, scheduler preparation and any replacement work. Reject the
elimination hypothesis before timing if the control lacks combine or candidate
combine persists. Confirm and report whether scheduler preparation remains.
Ordinary activity cannot prove per-CTA utilization; do not claim it does.

Only after identity, correctness and mechanism admission, run separate unprofiled
cold-L2 paired whole-call timing. Useful logical work may decrease even while
launch count decreases; a complete-call regression rejects the candidate.
Projection owns the current slot. No compute or official request is admitted
by this source audit.
