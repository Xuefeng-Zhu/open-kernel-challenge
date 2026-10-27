# Conditional sliding split-combine experiment: local preparation

The candidate adds one shape-dependent `num_splits=1` argument to the existing
fused sliding FA3 call for B16/B32, L8. It preserves the validated three-path
preparation fusion, full attention, B1, other dispatch, original precision,
causal/window bounds, cache stores and output gate. Original A2-11 remains
selected. This is one mechanism test, not a split-count sweep or an assumed
speedup from fewer launches.

The validated fusion baseline is remote commit
`05c9cc08948c42fed9e09aae1cd2b41edf966c74`. Its result permalink was verified
through the authenticated GitHub contents API:
[fusion results](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/05c9cc08948c42fed9e09aae1cd2b41edf966c74/reviews/attention-v2/fused-preparation-probe/RESULTS.md),
Git blob `2d87f01ecffe6b8047ec459898ad3df4bbd4b828`.

The independent worktree and branch are `attention-sliding-unsplit`. Its frozen
control is a byte-exact copy of the validated fusion candidate. Source SHA256s:

| File | SHA256 |
|---|---|
| `baseline.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| `candidate.py` | `586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf` |
| Preserved selected A2-11 | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |

```diff
         scheduler_metadata=data['scheduler_metadata'],
+        **({'num_splits': 1} if batch in (16, 32) and length == 8 else {}),
         **descale)
```

This is **conditionally source-ready**. The actual installed wrapper signature,
source/extension identity and force-one kernel behavior are unknown locally;
no access denial occurred. The saved official baseline profiles establish that
both target paths currently execute mainloop, combine and varlen scheduler
preparation. Pinned upstream code explains how one split omits combine but can
retain preparation. It does not establish the installed implementation's
identity. The first phase of any future authorized job must inspect the actual
visible callable and reject an unsupported or opaque API before invoking the
candidate; a later activity diagnostic must establish actual kernel counts.
The source audit records these distinctions and bounded evidence search.

The intended mathematical operation and precision are unchanged, while split
reduction order can change numerical output. Native output must pass the
original official oracle; bit equality between the two attention outputs is
not required. Unchanged preparation/new-cache bytes must remain exact, and
each arm must reproduce its own fixed-address A→B→A output/cache bytes.
CPU host models check only contracts and mathematical masks, not native
splitting, scheduling or GPU numerical behavior.

At the current official weights, saving 2µs in both complete target calls models
only +0.578% overall; 4µs models +1.171%. Observed combine activities of 4.2/4.4µs
overlap other work and are not predicted savings. Conditional useful query
tile counts of 32/64 concern logical work, not measured CTA grids or occupancy.
Lost KV parallelism can outweigh launch removal. Scheduler preparation may
persist without invalidating the hypothesis; useful whole-call benefit is
the acceptance criterion.

The planned native gate covers six sliding cases plus an unchanged full-B16
verify control, both scale/page combinations, correct generated window-edge
keys, live slot zero, changing metadata, original oracle, untouched cache,
fixed-address replay and independent disjoint-cache submissions. One bounded
Torch/CUPTI activity session checks that baseline combine exists and candidate
combine disappears, recording scheduler preparation separately. Unprofiled
whole-call timing follows only after all mechanism/correctness gates pass.
Both targets must save at least 2µs by the median of balanced absolute pair
differences, with all six ratio pairs positive and no unexplained >5% unchanged
control regression, before proposing one official check. Details are in
`PLAN.md` and `runtime/PROTOCOL.md`.

No GPU, Modal app, official check, credential operation or billing operation
has run in this preparation stage. New compute cost is $0. A future job needs
a new explicit parent serialized-slot grant and fresh free-credit/$0 paid-cap
admission. Proposed bounds remain worker160s, child170s, function185s,
startup20s, watchdog210s, reserve $0.58 free credits. There are no automatic
retries, counter attempts, installs, permission expansion or external Pro sends.
Readiness does not authorize launch, and remaining official quota is not a
reason to test an unsupported or unhelpful candidate.
