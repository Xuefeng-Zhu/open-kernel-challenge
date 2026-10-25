# Composition replication: original correctness, then paired whole-call timing

These files are local preparation. No GPU, Modal app, official check, profiler,
network, credential operation or installation has been performed by this task.
A new parent serialized-slot grant and live free-credit/$0 paid-cap admission
are required before a launcher may execute the worker. It never submits an
official check automatically.

The exact control is validated preparation fusion, SHA256
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
The composition candidate is SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
It transplants the exact measured A2-15 `_fused` and `_e2_kernel_fn` functions
into that control. All other function bodies, fusion dispatch, FA3 arguments,
original precision, split policy, geometry, output gate and KV ownership remain
unchanged. Automatic FA3 splits remain in place. The source audit proves which
config selects PAGE_LOOKAHEAD; this worker imports the two complete entrypoints
without patching either loaded module.

The prior A2-15 experiment measured a5.189% target cut in one session against
A2-13 and missed its original10% goal. The new question is whether that source
component still helps after composition with validated fusion, using a new
session and fresh predefined seeds **135711/135712**. These differ from the
historical135701/135702 fixtures. This is not evidence that page-ID scheduling
alone caused the old gain; the source edit also changed generated buffering.
The historical10% goal remains unmet, independent of the new replication gate.

## Frozen input and ownership gates

Before GPU imports, verify both complete source hashes and the unchanged
`modal_worker.py`, official reference and official checks hashes. Require native
H200 execution and record GPU/capability, Python, Torch, CUDA and Triton versions.
No new FA3 API or hardware-profiler path is introduced by this composition.

Five cases use two fresh generated versions each:

| Case | Config and purpose |
|---|---|
| Changed target | Full B32/L1, TP8/D64, context80000, page1, scale1, agentic/coherent; the scored family |
| Changed target boundary | Full B32/L1, TP8, page64, scale1/64, context8193 capacity; histories0/1/63/64/65/8191/8192/8193 repeated four times, then reversed |
| Unchanged custom fallback | Full B4/L1, TP8, page64, scale1/64, context8193 capacity; histories0/63/64/8193, then reversed; PAGE_LOOKAHEAD remains false |
| DP smoke | Full B32/L1, DP heads, page64, scale1/64, context65 capacity; histories0/63/64/65 repeated eight times, then reversed; exercises the generic true guard and one historical-chunk capacity |
| Unchanged fusion control | Full B16/L8, TP8/D64, context80000, page1, scale1, agentic/coherent |

The three boundary fixtures update every live length, position, cache length,
physical new-token cache location and scheduler tensor consistently. Physical
page zero is made live through a bijective table-page swap, and unused page
suffixes are negative. Only fixture construction synchronizes or reads GPU
values; none of it enters a timed call. Real serving-supplied full-decode
scheduler metadata is regenerated from each actual length vector outside
capture; the custom route may ignore it as the contract allows, while it remains
immutable. The verify control keeps metadata None.

Both implementations must retain last-split ownership of new K/V, including
zero histories and empty historical chunks. The two-split target boundary and
the DP smoke's one8192-token chunk capacity cover different ownership paths.
These source-derived capacities are not measured occupancy or CTA claims.
No historical key, value or empty split is deleted to obtain a speedup.

For each case, capture both complete `kernel_fn` entrypoints independently and
replay A→B→A at fixed tensor/output addresses. Every replay calls the unchanged
official output/cache oracle and verifies immutable inputs and all untouched
cache rows: **30 oracle calls**. Each second-A must exactly reproduce its own
first-A output/new-KV bytes: **10 exact return-A checks**. After each label,
compare the entire post-store K/V caches across arms:30 byte comparisons.
Attention outputs need not be bit-identical across arms; each must satisfy the
original oracle and its own exact replay check. No precision or tolerance is
relaxed.

For the target-boundary case only, make a second private candidate graph with
disjoint writable caches. Restore A and B, then submit the two independent
graphs on separate streams; check each with the original oracle, stable
addresses and cache/immutability requirements. This adds **two oracle calls**,
for **32 total**. No allocation pool is explicitly shared. This checks
independent concurrent submission correctness, not actual hardware overlap.
The fallback and DP smoke do not claim this concurrency coverage.

All five cases and cross-arm cache checks must pass before timing. Explicit
`--correctness-only` mode stops here, omits performance and does not admit an
official check. Any failure or timeout retains the available report prefix and
forbids a performance claim; do not automatically relaunch.

## Paired timing and the new replication gate

Time only the scored B32 target and unchanged full-B16-verify control. For each,
retain four balanced warmup two-arm samples and twelve measured two-arm samples,
grouped into six adjacent reverse-order pairs. Each pair uses the same fixture
version; versions alternate by pair. This is **64 checked warmup/measured graph
replays** across the two timed families.

Restore all inputs/caches and read256MiB to evict L2 outside the CUDA-event
interval. Time the complete captured `kernel_fn` device work: preparation,
cache stores, custom producer/merge or unchanged FA3 scheduling/compute/combine,
and output gate. Graph replay reuses captured memory; host allocation, fixture
generation, metadata construction, restoration, validation and eviction are
outside timing. There is no profiler session, counter collection or overlap
measurement in this protocol.

Every warmup and measured replay must exactly match its own source/input's
previously oracle-validated output/new-KV bytes and preserve immutable inputs,
untouched cache rows and addresses. Keep all raw samples, order and version.
Report six reverse-order geometric pair ratios and their median, raw medians
and ranges. Empty-graph calibration is retained before/after with unclipped
subtraction; raw timing is primary and this is not judge calibration.

The **new composition-replication materiality gate**, declared before compute,
requires target median balanced latency reduction≥4%, all six target pair ratios
favoring composition, and unchanged-control median paired slowdown≤5%.
Reduction is `1−1/median(pair geometric ratios)`; the control uses the reciprocal
ratio to express slowdown. Missing/incomplete families or nonfinite/nonpositive
timing fail the gate. The target threshold is supported by the prior measured
component range but does not rewrite its failed original10% ambition.

An interpretable run can have `passed=true` for native correctness while
`official_check_admitted=false` when benefit is insufficient. A passing benefit
gate only supports parent consideration of one complete nonranking official
check, not automatic submission, selected-version promotion or a formal score.
One session with two fixtures and six pairs is not independent-session
repeatability or a confidence interval. Small whole-score transfer models
remain conditional; retained source and measured evidence determine the next
decision, not remaining quota.

## Bounded execution envelope

Keep worker160s, subprocess170s, function185s, startup20s and app watchdog210s,
one H200,2CPU,8GiB, one invocation and no retries. The worker report is bounded
to400KB and the controller transport to500KB. At the previously recorded rate,
the conservative full-watchdog double-rate allowance plus$0.02 is$0.56817,
covered by the reviewed **$0.58 free-credit reserve**, actual paid authority$0.
The launcher rechecks rates, free-credit coverage, paid cap/spend, existing
protected configuration and stopped/tasks0 state before admission. It must stop
and independently reconcile its app before releasing the slot. No new
credential, billing limit, installation or permission is authorized here.

Offline protocol tests:

```sh
python3 -B reviews/attention-v2/fusion-lookahead-probe/runtime/test_worker.py
```
