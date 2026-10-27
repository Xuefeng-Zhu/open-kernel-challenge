# Independent review: conditional local readiness passes

The candidate, local tests, model, native protocol and controller have no
remaining source or harness blocker. This review approves an immutable local
preparation freeze. It does not establish the installed FA3 API, actual unsplit
behavior, native correctness or performance, and does not admit a GPU or
official job. A separate parent slot grant and current free-credit/paid-zero
guardrails remain required.

## Reviewed source and isolation

| File | SHA256 |
|---|---|
| `baseline.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| `candidate.py` | `586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf` |
| `runtime/worker.py` | `953537670113411a0de1445864bbaa52d4b3cfecb276da5a01ca5fdf0ee6d812` |
| `runtime/identity.py` | `46fd1727b4744ce74f4eb4514db704710451dffba527d28abefe2c8800e9e4a0` |
| `runtime/controller.py` | `a175c6265151aa8465c51f75c132f2f3f372c62d8a281663b5ad64c896e0a5d9` |
| `test_source.py` | `24efb99362626831a25125a9067e85d61078dd925c55db121eb696ffb6e96636` |
| `runtime/test_worker.py` | `f99b1f3546e24cfe133dabdb51a429f895b153d6699be04704fd60731efddb72` |
| `runtime/test_identity.py` | `75cf807734e1047be52cc443b0a330125e3a0a7f2639b84b4449d1eae4570c41` |
| `model.py` | `678e960c1610893c43c69aba603dbaa4de7eee57a2a211e5f1a50d4394f2b3ce` |

Independent AST normalization confirms the complete source differs only by
one conditional unpacked keyword in `_fa3_sliding_verify_fused`:
`num_splits=1` for B16/B32, L8. Removing that keyword yields the exact baseline
module AST. The existing entrypoint restricts the helper to sliding
window2048/RoPE500000. The original omitted argument is retained on B1 and
nearby shapes, without assuming a default value. Preparation, BF16/FP8
boundaries, softmax scale, causal/window expressions, live tensors, descales,
output gate and all full-attention fusion remain unchanged. Partitioning can
change reduction order, so original native tolerances remain necessary.
Selected A2-11 remains byte-identical at SHA256
`c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`.

## Local validation and evidence assessment

The reviewer independently passed the source/CPU host-model tests (3), score
model tests (4), identity guards (10), runtime guards (9) and controller guards
(7): 33 test methods. CPU stand-ins exercise parameter forwarding, original
oracle/cache contracts, fixed-address A→B→A, scales/pages, boundary keys and a
live slot-zero target. They execute no installed FA3 or CUDA code and establish
no native scheduling or precision result.

The score model uses the current official fusion receipt and weighted
complete-call times, reconstructs its six-load score, and correctly models
conditional two-target savings. Two microseconds saved in each target models
about +0.578% overall; four models about +1.171%. The rounded combine and
scheduler activities overlap other work and are not recoverable latency.
The 32/64 logical query-task estimates are conditional on packing/tile
geometry, not measured CTAs or occupancy. These uncertainties are retained in
the plan and source audit.

## Native admission and measurement gates

The worker verifies source/support/identity hashes before GPU work. Before
candidate invocation it inspects the visible installed callable with
`follow_wrapped=False`, requiring an explicit keyword-capable `num_splits`
parameter and the same callable in both source modules. Generic `**kwargs`,
positional-only and unavailable signatures fail closed. The stricter extension
gate inventories already-loaded `sgl_kernel`, `flash_attn` and `sgl_flash_attn`
modules plus relevant `torch.ops.loaded_libraries`, without importing another
package or traversing directories. Canonical paths merge aliases before
streaming full SHA256 hashes, bounded to 4GiB per file and 8GiB total. At least
one observed relevant extension must be fully hashed, every observed extension
must have its hash, no path may be omitted, and the loaded-library inventory
must be available. Missing, denied, over-budget or omitted identities stop the
worker before a CUDA query, fixture or candidate call. The observed-file scope
does not assert discovery of every possible transitive binary. TimeoutError is
re-raised through source, hash, metadata and library-inventory error handlers,
preserving the unchanged 160-second worker alarm. No tool installation or
credential change is part of this gate.

The seven cases cover both target batches in scored, soft boundary and
officially planted fixed-context2048 edge families, plus an unchanged full-B16
verify control. The worker preserves valid edge placement, updates all live
boundary metadata, exercises physical slot zero and negative unused pages,
and retains verify metadata None. Twelve direct preparation comparisons require
exact Q/K/V bytes. Complete-call A→B→A entails 42 official oracle calls and
14 exact second-A checks; two additional disjoint-cache graph pairs add four
oracle checks. Cross-arm cache bytes must agree, while attention outputs are
checked under original tolerances and self-replay equality.

Only after those gates does one separate Torch/CUPTI session observe the four
target graph/arm replays. Synchronized CPU markers attribute full CUDA names
and intervals. The control must have recognized mainloop/combine work, and the
candidate must have recognized mainloop work with no recognized combine.
Scheduler preparation is reported separately and may persist. Other kernel
names remain in the exact bounded event list for review; unassigned activity,
missing structure or overflow stops timing. This supplies no precise CTA
residency or hardware-counter claim.

Clean cold-L2 whole-call timing then retains four warmup and twelve measured
two-arm samples for each target and the unchanged control. All 96 replay
results are checked. Pair order and fixture versions are balanced. The benefit
gate uses the median of six paired arithmetic-mean absolute savings, requires
at least 2 microseconds in each target and all six geometric ratios positive,
and permits at most 5% paired slowdown in the unchanged control. Materiality
failure rejects this combined candidate without an automatic official check.
One session does not provide independent-session repeatability.

## Controller and remaining gates

The nine-file payload includes the identity helper, exact candidate/control,
existing validation support and licenses. Repo-relative remote paths match
the worker's imports. Offline preflight reads committed bytes and parses source
without importing Modal or touching authentication. Launch additionally
requires exact reviewed HEAD, fresh explicit slot and billing receipts, live
paid-zero/rate/free-credit checks and stopped/tasks0 app state. Existing
protected configuration is used through its supported path and is excluded
from the payload.

Bounds remain worker160s, child170s, function185s, startup20s and watchdog210s,
one H200/2CPU/8GiB invocation, no retry, with a $0.58 free-credit reservation
and actual paid authority $0. Transport is bounded and hash-checked. Stop and
independent closure are mandatory; stopping/nonzero tasks or uncertain outcomes
do not authorize a duplicate launch. Commit freeze and the actual offline
committed-payload preflight are still required after this review. Native API,
mechanism, correctness and benefit gates remain unexecuted.

This review changed only this report. It ran no GPU, Modal, official, browser,
network, installation or credential action.
