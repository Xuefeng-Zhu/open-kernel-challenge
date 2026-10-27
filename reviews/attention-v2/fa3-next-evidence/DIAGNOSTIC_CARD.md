# Selected-attention mechanism diagnostic: same fixture, before/after activity

**Question.** Which FA3 specialization, packing/scheduling preparation and
auxiliary allocation shapes actually execute for the selected page1 routes,
and can their activity durations be related to the same graph's unprofiled
whole-call time? This is a causal-uncertainty diagnostic, not a performance
candidate. It changes no precision, split policy, algorithm or selected source.

The saved full-B32-verify trace proves an installed dynamic persistent FA3
specialization, but predates preparation fusion. Current official names are
truncated and lack resource fields. Full-B16-decode benchmarks stayed near
194.3µs while forward profile activity changed185.7→297.3µs across jobs.
Do not infer a recoverable60% loss or transplant a different architecture.
See `LOCAL_SPECIALIZATION_AUDIT.md` for exact evidence and qualifications.

The CPU inventory `image-inventory/result.json`, SHA256
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`,
found the installed public wrapper and Gluon async APIs, not exact FA3 C++ source
in its fixed five candidate locations. This is a bounded search result, not
proof of source absence everywhere. FlashInfer XQA eligibility excludes page1;
none of these findings establishes a FA3 page1 optimization gap.

## Smallest useful fixed stratification

One fixture per row, seed135741; TP8=16Q/2KV heads, D64, ctx80000, page1,
unit scales. Use the original generator. B1 rows use fixed lengths, batched
rows use agentic lengths. Full cases use coherent inputs, sliding cases soft.
Save actual lengths/config/page-table and supplied metadata shape/dtype/hash.

| Case | Exact selected dispatch | Purpose |
|---|---|---|
| Full B16/L1 decode | `_fa3_full_verify_fused` → public FA3 → norm/gate; supplied decode metadata | Large decode specialization and prior profile discrepancy |
| Full B16/L8 verify | Same fused helper; metadataNone, normal in-call schedule preparation | Large verify, fewer requests |
| Full B32/L8 verify | Same fused helper; metadataNone | Large verify, comparison with older complete trace |
| Full B1/L1 decode | Same fused helper; supplied decode metadata | Small-batch full FA3 stratum |
| Sliding B1/L1 decode | `_e2_kernel_fn` → custom `_fused`/merge; PAGE_LOOKAHEADFalse, RoPE/window unchanged | B1 sliding custom stratum; do not mislabel as FA3 |
| Sliding B1/L8 verify | `_fa3_sliding_verify_fused` → public windowed FA3 → norm/gate | B1 sliding FA3 stratum |

These six fixed rows cover different dispatch mechanisms without a broad
shape sweep. B1/full verify and batched sliding are not claimed covered.
All chosen paths are timed as complete `kernel_fn` calls, not subkernels.
No task-map cache, pack_gqa/num_splits override, KV repack or precision change.

## Measurement protocol

1. Hash-bind selected source and original helpers/oracle/checks. Capture each
   row once with its own default graph pool and private input/cache/output
   allocations. Keep the exact graph and tensor bytes for every later phase;
   record stable addresses. Generate decode metadata outside capture as the
   original generator does; leave all work normally inside `kernel_fn` there.
2. Run one original oracle check per row. Then two warmups and three
   unprofiled whole-graph CUDA-event samples per row. Before each replay,
   restore that fixture and read256MiB to evict L2, outside the event interval.
   Check byte-exact output/new-KV against the row's own oracle-checked result,
   immutable inputs and untouched cache rows. No replay-cost subtraction.
3. Collect **one** existing Torch CPU/CUDA activity session containing exactly
   one fenced, annotated graph replay per row. Restore/evict/synchronize before
   each range, synchronize completion inside it. Export the full raw trace and
   untruncated kernel names, graph/node IDs, grid/block, register/shared fields,
   device properties and stage intervals. Use interval unions, not overlapping
   duration sums. CPU ranges and instrumented kernel durations are diagnostic.
4. Check the original oracle again on each activity result, then collect three
   more unprofiled samples per row using the same graph/fixture/L2 protocol.
   Retain all samples, before/after medians and ranges. There are12 original
   oracle checks,48 checked warmup/pre/post replays and6 activity replays,
   excluding capture warmups. These are invariance checks, not an optimization
   benchmark or a confidence interval for an improvement.
5. Only **after** those measurements, make one separately labeled, untimed
   validating auxiliary-return replay for each of the five FA3 rows, using its
   own captured adapter graph; capture warmups are excluded from that count.
   Keep the same capture mode and all fwd arguments as the selected call.
   A small reviewed
   harness adapter must reproduce the selected fused preparation, FA3 arguments
   and norm/gate exactly, adding only supported `return_softmax_lse=True` and
   recording returned tensor shapes/dtypes. It must byte-match that row's own
   oracle-checked final output/new-KV and entire K/V caches. Do not replace or
   monkeypatch any loaded candidate/public wrapper function.

The saved installed wrapper unpacks `out, softmax_lse, *rest` from the operator;
`return_softmax_lse` changes only its Python return container, not fwd arguments.
Record the installed operator schema where exposed and raw auxiliary metadata.
If a `rest` tensor's meaning cannot be proved from that schema/source, keep the
field unnamed/unknown. No guessed “split count” from a dimension, combine grid,
`prepare_varlen<1/2,true>` or template integer. If the API returns no interpretable
auxiliary data, preserve that negative result; no alternate implementation.

## Exact reuse and implementation state

| Existing source | Functions to reuse | SHA256 |
|---|---|---|
| Selected `fusion-lookahead-probe/candidate.py` | `kernel_fn`, unchanged full/sliding fused helpers | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| `modal_worker.py` | `clone_inputs`, `restore`, `unchanged`, `capture`, `timed` | `9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e` |
| `official/reference.py` | `generate_input`, `collect`, `check_implementation`, decode metadata generation | `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b` |
| `official/checks.py` | Original precision/output/cache tolerances | `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055` |
| Architecture profiling `gpu/worker.py` | Pure `interval_union`, range fencing/raw trace export and compact event extraction | `07c6c36b06995df46b49b90208b0454eba1aa3efb17787c7a2fbdf67b1efa326` |
| `fusion-lookahead-probe/runtime/controller.py` | Reviewed serial admission, protected existing config, bounded child/watchdog and independent shutdown pattern | `0b78f5a5147030f67701dcaf1298ba5077b947e133c6485478931e938014cbf1` |

The architecture worker is in the independent `attention-architecture` worktree.
Reuse its pure parser/fencing functions, **not its main program or optional
counter branch**. The parser must become dispatch-aware: selected fused prep is
`_fa3_prepare_fused`, decode may consume supplied scheduling metadata, and the
B1 sliding-decode row expects custom kernels rather than a FA3 mainloop. Do not
require every historical stage name for every row or silently truncate symbols.

**Not implementation-ready yet.** This card writes no GPU harness. Required
local work is a six-range adaptation, the separate auxiliary-return adapter,
strict byte/shape/path bounds, synthetic parser/dispatch tests and independent
review. The auxiliary schema/field mapping is unresolved. Existing activity
collection succeeded historically, but current live admission must still verify
supported activities, selected-source bytes, image and existing access. There
is no known current permission denial; CPU inventory alone is not a fresh GPU
activity-permission test. Failure does not authorize installs or counter retries.

## Time, free-credit reservation and stop rules

Use the existing image `im-cPTbGpNiX3RarKResPgX7h`, one H200,2CPU,8GiB,
one serialized app, zero retries:160s worker/170s child/185s function,
20s startup/210s watchdog. No unrelated parallel GPU or official job.
Bound raw trace at4MiB, compact summary at256KiB and encoded transport at2MiB;
retain per-file and whole-transport hashes. Oversize means an explicit failure,
not truncating the evidence needed for attribution.

Using the previously captured combined rate$0.00130496/s,210s nominal active
resources cost$0.27404160; a conservative `2×210×rate+$0.02` is$0.56808320.
Reserve$0.58 from existing **free credits only**, after checking fresh rates,
balance and paid cap$0. This historical-rate calculation is not a current quote
or exact invoice. No reservation or launch is created by this card. Independently
stop the app, verify tasks0, record observed usage and release the slot promptly.

Stop on source/schema mismatch, changed fixture/address, oracle/cache/byte
failure, unsupported collection, transport bound, watchdog, access denial,
unexpected app or failure of free-credit/$0 paid admission. No tolerance change,
credential renewal, paid resources, NCU, permission expansion or fallback route.
Before/after timing drift or changed specialization is a useful **negative**
observation: retain it and stop attributing activity duration to normal latency;
do not rerun until a preferred result appears.

## What the result can and cannot justify

It can identify the current executed specialization/launch resources, confirm
which scheduler/preparation/combine stages actually run for each stratum, and
test whether the earlier profile disparity persists on the same graph/state.
Supported auxiliary metadata may narrow runtime split/allocation uncertainty.
It cannot measure actual per-CTA residency/tail, DRAM throughput, tensor
utilization, warp stalls or the exact page1 memory-instruction path. Static
register/shared estimates and persistent grid size are not those measurements;
the older zero occupancy estimate remains invalid as a hardware conclusion.
Exact FA3 C++ source/instruction mapping is still absent. Gluon API presence
and XQA ineligibility do not fill that mechanism gap.

No latency gain is predicted. As a conditional leverage check, reducing each
of the three large FA3 full paths by5% would model+2.077% overall;10% each
models+4.290%;3% overall requires about7.125% per path with all other terms
fixed. None is an observed recoverable loss. Build another candidate only
after this or other admissible evidence identifies a concrete avoidable cost.
No official check is justified for unchanged source, and no ranked entry,
selection change or serving-concurrency claim follows from this diagnostic.
