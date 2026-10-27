# Conditional installed-API, mechanism and whole-call diagnostic

Prepared locally only. No installed FA3 API, native execution, CUDA activity,
correctness or speedup has been established by these files. The parent must
grant the serialized slot and verify live free-credit/$0 paid-cap guardrails
before any launch. An idle slot is not admission. No official check is sent
automatically by this worker.

The baseline is the validated fusion candidate, SHA256
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
The conditional candidate SHA256 is
`586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf`.
Its only change is the target-only `num_splits=1` keyword on the existing
sliding fused-preparation FA3 call for B16/B32, q_len8. The original A2-11
selected state, validated full-path fusion, preparation, cache writes,
causal/window masks, quantization and output gate remain unchanged.

## Admission comes before candidate calls

The worker verifies exact candidate/control, official oracle/checks, existing
buffer-helper and identity-helper hashes. It imports the installed baseline
and candidate wrappers, then inspects the visible actual FA3 callable using
`inspect.signature(..., follow_wrapped=False)`. Admission requires an explicit
keyword-capable `num_splits` parameter, not merely `**kwargs` or a positional-only
parameter. Both candidate modules must refer to the same installed callable.
This gate runs before CUDA availability queries, fixtures, candidate invocation
or profiling. Failure is a precise unsupported/unverified API observation, not
proof that every possible installed interface lacks the feature.

The worker also requires `exact_observed_extensions_verified`: at least one
relevant already-loaded extension must be observed, every observed relevant
extension must have a complete file SHA256, and no relevant path may be omitted.
It inspects already-loaded `sgl_kernel`, `flash_attn`, `sgl_flash_attn` modules
and the relevant paths in existing `torch.ops.loaded_libraries`, without
importing another package. Unavailable inventories, lazy providers, missing
paths, unreadable files, omitted paths or incomplete hashes fail closed before
candidate calls. There is no fallback or automatic second job.

`identity.py` records the callable module/name/signature and bounded installed
source, source-file/function SHA256s, package versions, and already-loaded
relevant extension paths/sizes/hashes. Source text is limited to64KiB per
record, files inspected for function text to1MiB, and streaming full-file hashes
to4GiB per file/8GiB total. Canonical resolved paths deduplicate module/library
aliases so a large file is not rehashed. The160s worker deadline remains
unchanged; timeout exceptions propagate through provenance inspection.
It performs no recursive filesystem scan or GPU-package import. Every omitted
or unavailable hash is identified and blocks strict observed-extension admission.
Verified hashes identify only the observed extension files, not every possible
transitive dependency or an exact complete installed build. Pinned upstream
source is a causal lead, not this runtime identity.

## Native correctness before activity or timing

For both B16/B32 sliding q_len8, TP8/D64, window2048, RoPE500000, seeds135701/
135702, prepare three fixture families:

- Scored: context80000, page1, scale1, agentic lengths, soft distribution.
- Boundary: page64, scale1/64, actual histories0/1/63/64/65/2047/2048/2049,
  repeated to the batch then reversed. Update every position, physical new
  cache location and FA3 length at fixed shape/capacity. Make physical slot0
  live by a bijective page swap, retain scattered pages and mark unused
  page-table entries negative. This soft distribution does not claim planted
  edge keys survive metadata mutation.
- Dedicated edges: unchanged official generation with context2048,
  `spread=fixed, dist=edges`, page1, scale1. Its oldest-visible and first-excluded
  keys are planted by the official generator for the actual history and are
  not invalidated by any length mutation. Histories match across A/B here;
  QKV and physical pages still change.

An unchanged full B16/q_len8/context80000/page1/scale1 coherent/agentic case
is the control. All verify metadata remains None. Both input versions are
fresh fixtures; work buffers and outputs keep fixed addresses between replays.

Before each of the six sliding cases, directly compare the same native fused
preparation in both source modules for both versions: FP8 Q and entire
post-store K/V bytes must be exact, queries finite, immutable inputs and
untouched cache rows intact. These12 groups contain36 Q/K/V byte comparisons.
This is separate from FA3 and not a preparation microbenchmark.

Capture both complete entrypoints independently for all seven cases. Replay
A→B→A with the unchanged official output/cache oracle on every arm:42 oracle
calls and14 exact second-A return checks. Compare entire post-store K/V cache
bytes across the two arms after every replay label. Attention outputs need not
be bit-identical between split policies: each arm must satisfy the original
oracle/tolerances and reproduce its own first-A bytes exactly. No tolerance or
precision flag is changed. Two additional private candidate graph captures for
the two boundary cases use disjoint writable caches and separate-stream
submission, adding4 oracle checks. This tests independent concurrent submission
correctness; actual hardware overlap is not measured or asserted.

`--correctness-only` stops here with no mechanism/performance admission. It
does not authorize an official check and supplies no speedup claim.

## One separate activity session establishes the mechanism

After all oracle/cache gates, profile exactly one replay of each target graph
and arm in one Torch/CUPTI CPU+CUDA activity session:4 graph replays. Restore/
validate inputs before collection, use synchronized named CPU intervals for
attribution, and retain every observed CUDA event's full name/start/end within
a128KiB exact event-list bound. Attribution ambiguity, missing markers, unknown
mainloop classification or transport overflow stops timing rather than
truncating names and manufacturing an absence claim. The report retains raw
event intervals and classifications; no hardware counters, installs or NCU
sessions are used.

For each target, require an observed FA3 mainloop in both arms, a baseline
combine kernel and no candidate combine kernel. The installed candidate may
still launch `prepare_varlen_num_blocks`; report actual preparation counts.
Scheduler preparation removal is not assumed and is not a success criterion.
All profiled replays must still match their own oracle-checked bytes and
preserve caches/inputs. Activity durations are instrumented evidence only:
they are never benchmark samples, cannot be summed/subtracted into a predicted
gain and do not reveal exact CTA occupancy/residency or hardware stalls.

## Clean whole-call timing and predeclared failure gates

Only after correctness and actual mechanism gates, measure the two scored
targets and unchanged full B16 control. For each, retain4 warmup two-arm
samples and12 measured two-arm samples, grouped as6 reverse-order pairs.
Each pair uses the same fixture version; versions alternate by pair. Restore
all inputs/caches, read256MiB to evict L2 outside the event interval, and time
the complete `kernel_fn` graph device work including per-call preparation,
cache stores, all FA3 prep/mainloop/combine and output gate. Replay reuses
captured graph memory; host allocation, fixture work, validation, eviction
and activity collection are outside timing.

All96 warmup/measured replays must exactly reproduce their own source/input's
oracle-checked output/new-KV bytes and pass address/immutability/untouched-row
checks. Keep every sample and empty-graph calibration before/after. Raw times
are primary; separate unclipped empty-graph subtraction is not official judge
calibration. No profiled sample enters this set.

An interpretable completed run sets `passed` for correctness/mechanism and
reports performance separately. Admit consideration of one official check only
if **each** sliding target has median reverse-order pair mean saving≥2µs and
all6 pair geometric ratios favor the candidate, and the unchanged control's
median paired candidate slowdown is≤5%. A reverse-order pair's mean saving is
the mean of the two baseline-minus-candidate times for the same fixture. The
ratio is the geometric mean of its two control/candidate ratios. Failure of
materiality rejects this combined candidate even if combine disappeared.

On the latest official costs,2µs saved in both complete targets conditionally
models about+0.578% overall;4µs models about+1.171%. These are sensitivity
models, not forecasts or activity subtraction. Unsplit work may lose useful
parallelism; the source's32/64 packed query-task estimate is conditional on
installed packing/tile geometry, not a measured CTA grid or occupancy claim.
One session is not independent-session repeatability or an official score.

## Bounds and resources

Keep worker160s, subprocess170s, function185s, startup20s and app watchdog210s,
one H200/2CPU/8GiB invocation, no retry, bounded400,000-byte worker report and500KB
controller transport. No long trace or extra artifact channel is required.
The reviewed reserve is$0.58 in verified free credits only, actual paid$0.
Timeouts retain the available prefix; do not extend a run or relaunch to fill
quota. The parent/controller must stop the app and independently verify tasks0
before releasing the slot or sending a later official check.

Offline tests import only standard-library protocol code:

```sh
python3 -B reviews/attention-v2/sliding-unsplit-probe/runtime/test_identity.py
python3 -B reviews/attention-v2/sliding-unsplit-probe/runtime/test_worker.py
```
