# Source-bound native gate and whole-call paired timing

Prepared locally; no GPU, official check, profiler or performance result is implied.
The unchanged A2-13 control is `../baseline.py`, SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
The isolated candidate is `../candidate.py`, SHA256
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
Only the full-attention fused-preparation dispatch guard changes. FA3 computation,
arguments, precision, scheduler metadata, split choice, cache format, output gate,
and every helper body remain preserved by separate source tests. Selected A2-11
is unchanged. No loaded candidate module is patched by this harness.

The worker verifies all source/support hashes before importing GPU code. It uses
the already installed Torch/Triton/FA3 stack in the existing H200 image, with no
installs, credentials, permission changes, network or reference import in the
candidate. The harness imports the unchanged official oracle solely for validation.

## Predeclared gates

For each newly routed `(B,L)=(16,1),(16,8),(32,8)`, exercise two cases:

1. Scored family: full attention, TP8/D64, context80000, page1, scale1,
   agentic lengths and coherent cache, seeds135701/135702.
2. Boundary family: full attention, TP8/D64, page64, scale1/64, soft cache,
   actual histories0/1/63/64/65/127/128/129 repeated to batch size, then reversed.
   Append crosses page boundaries for L8; L1 exercises last/first slots around a
   boundary. Physical page0 is explicitly live, pages are scattered, and unused
   page-table entries are negative. Metadata for every new token is rebuilt.

Both page sizes and both scales reach all three changed routes; this is two
predeclared combinations, not a claim of exhaustive cross-product coverage.
All A/B inputs change actual lengths, QKV values and physical page mappings.
The decode's serving-supplied FA3 schedule is generated from each actual length
vector outside capture, as in the official input generator. Its tensor is copied
into fixed-address graph inputs between replays; the candidate forwards it.
Verify's scheduler metadata stays None. Any metadata shape mismatch fails safely.

Before any complete-call timing, compare the ordinary and preserved fused native
preparations on both versions of all six cases. Capture neither FA3 nor output
gate in this isolated diagnostic. Q FP8 bytes and the entire post-store K/V cache
must be exact, finite queries and immutable inputs/untouched rows must pass.
**Any byte mismatch stops the experiment before timing.** A mismatch is an
equivalence investigation, not grounds to relax precision or tolerances. RMS
reduction rounding is the principal uncertainty despite preserved BF16 boundaries.

Then capture both complete `kernel_fn` entrypoints independently for each case.
Replay A→B→A at fixed addresses, calling the unchanged official oracle on every
replay, checking immutable inputs and every untouched cache row. The second A
must reproduce its own first-A output/new-KV bytes exactly. This is36 oracle
checks across12 graphs. Every native preparation and replay gate must pass.

For each boundary target, make a second private candidate graph capture with
disjoint writable caches and concurrently submit the two graphs on separate
streams using A and B. Check both with the unchanged official oracle and cache/
immutability/address requirements:6 additional oracle checks. This tests independent
calls submitted concurrently; it does not measure or prove hardware overlap.
No allocation pool is explicitly shared. No input-derived cache crosses calls.

`--correctness-only` exits successfully after these gates and reports that timing
was omitted. It produces no performance claim. Otherwise proceed as below.

## Timing

For each of the three scored targets, retain4 warmup two-arm samples then12
measured two-arm samples, grouped into6 adjacent reverse-order pairs. Each pair
uses the same oracle-validated input version; pair versions alternate. Restore
the whole input/cache before each replay and read256MiB to evict L2 outside the
CUDA event interval. Measure the entire captured `kernel_fn` device work,
including preparation, stores, FA3 scheduling/compute/combine and output gate.
Capture includes each call's allocation structure; replay reuses graph memory,
so the event interval does not include host allocator work.
No fixture work, validation or profiler enters the interval.

Every warmup and measured replay must exactly match its own source/input's
oracle-validated output/new-KV bytes, with immutable inputs and untouched rows
checked. Retain every raw sample and its order/version/check. Report raw medians,
all ratios and median reverse-order geometric pair ratio. Separately report
empty-graph calibration before/after without clipping negative differences;
raw timing is primary and this is not judge calibration. One session with six
pairs is not six independent sessions and supplies no formal score or interval.

This first session falsifies a small preparation hypothesis. No improvement,
failure of exact preparation, any correctness/cache failure or an unexplained
>5% target regression rejects promotion. A material useful gain needs a later
independent session and complete official coverage before adoption; the assumed
5–7µs gain and +0.84–1.19% model are not forecasts or measurements.

## Admission and closure

No GPU starts until the parent explicitly grants the serialized slot and live
free-credit/$0 paid-cap guardrails are verified. Proposed maximum bounds are
worker160s, subprocess170s, function185s and app watchdog210s, one invocation,
one H200, no retries. At the recorded H200+2CPU+8GiB rate
`(4.54 + 2×0.0473 + 8×0.008)/3600 = $0.0013051667/s`, a conservative210s
wall allowance charged twice plus$0.02 slack is$0.56817; round the reserve to
**$0.58 free credits**. This is a planning bound, not an invoice, and actual
platform time must be reconciled. A shorter correctness-only run keeps the same
safe ceiling. The launcher may tighten bounds after review, never silently
extend them. All apps must be stopped and independently verified tasks0 before
releasing the serialized slot, and delayed workspace usage must be qualified.

This harness deliberately omits16-prerequisite/12-scored full-suite coverage and
profiling. Unchanged-route preservation is source evidence. A complete official
check is separate, follows parent slot authorization, and does not constitute a
formal contest entry. No third-party Pro message or upload is part of this task.

Local stdlib validation:

```sh
python3 -B reviews/attention-v2/fused-preparation-probe/runtime/test_worker.py
```
