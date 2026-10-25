# Composition source audit: fusion plus measured A2-15 producer

The composition is source-feasible and does not already exist in the validated
fusion control. The two changes descend from the same exact A2-13 bytes and
modify disjoint functions. The actual parent-created candidate was independently
verified as the exact two-function mechanical transplant below. This supports
testing integration/reproduction; it does not establish combined native
correctness or an additive performance gain. No candidate implementation,
GPU/Modal job, official request, browser action, installation or credential
operation was performed by this audit.

## Exact source and historical chain

| Source | SHA256 |
|---|---|
| A2-13 parent | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| Measured A2-15 | `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c` |
| Validated fusion control | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Actual composition candidate | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Preserved selected A2-11 | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |

The validated fusion report commit is
`05c9cc08948c42fed9e09aae1cd2b41edf966c74`. Its complete module AST equals
A2-13 after restoring only `kernel_fn`: its first guard additionally routes
full B16 decode and B16/B32 verify through the existing fused preparation.
That does not affect full decode B32's custom route.

The original A2-13 source at
`21fefb946524a25d8a06562dd28b631d49cd5175` matches the current parent bytes.
A2-15 first appeared at
`5b3a769a3471dbacda51a0ccb72dcaf4bbe73675`; its Git parent
`73d9d1acfa21722415bb0957f3ec9197c75c5855` contained A2-13 and no A2-15 file.
Do not infer source ancestry solely from a commit title: the existing complete
AST reconstruction test also removes exactly the declared lookahead edits and
recovers all A2-13 code. The rejected A2-14 deferred denominator is absent.

Both parent and A2-15 files match their exact Git bytes at native freeze
`31010ce6b69d54767dc322a53b0cf6894b03d955` and paired freeze
`5cb8db07702dd9a69bd632b2e425198c0dfc5e91`. The paired payload binds worker
`8f7cfd52daf2af5f7b4ea23cb07b88b88668dc4e910ef63c47e07f4d5b7287fc`
and raw result
`92b5ba197689c3e17ec3c86cefdefdb1edd4bfe0222171b0bd63bd67f000e867`.
These hashes were recomputed from local retained bytes. A separate reviewer
verified all25 native artifacts, totaling1,894,074 bytes.

## Smallest mechanical transplant

Start with the exact validated fusion control. Replace only the source segments
for `_fused` and `_e2_kernel_fn` with their exact measured A2-15 counterparts.
Retain the fusion `kernel_fn` and every other function unchanged. Equivalently,
start with exact A2-15 and replace only its `kernel_fn` with the exact validated
fusion entrypoint. Appropriate composition comments can identify the new file;
they do not authorize a further algorithm edit.

Independent AST comparison gives:

| Variant versus A2-13 | Changed functions |
|---|---|
| Measured A2-15 | `_fused`, `_e2_kernel_fn` |
| Validated fusion | `kernel_fn` |

The function sets and all other function ASTs agree. In particular `_merge`,
`_num_splits`, `_fa3_prepare_fused`, `_fa3_full_verify_fused`, and
`_fa3_sliding_verify_fused` are identical across all three sources. No rejected
unsplit source is a composition parent, and no new FA3 `num_splits` override
belongs in this candidate. Preserve baseline automatic splitting everywhere.

For the actual candidate, byte reconstruction from the control with exactly
these two `ast.get_source_segment` replacements matches the whole file. The
entire module AST recovers the validated fusion baseline after restoring the
two definitions; both replaced definitions, including decorators, equal the
measured A2-15 ASTs. Syntax compilation passed without importing a candidate.
No header, module docstring, import, global, license text, FA3 helper or
entrypoint was changed by the parent's transplant.

A2-15's host guard is exactly
`batch == 32 and length == 1 and window == 0 and not rope`. It supplies that
shape/config-only boolean as the new `PAGE_LOOKAHEAD` constexpr. The entrypoint
still sends this full decode B32 path to `_e2_kernel_fn`; the new fusion paths
go to FA3 and never execute this producer. Other custom fallback configurations
receive false. Pruning the false constexpr branches recovers the parent's
historical-loop operations. This is source/semantic isolation, not proof that
every unchanged specialization compiles to identical machine bytes.
An independent evaluation over680 batch/length/window/RoPE configurations
enabled only `(32,1,0,False)`. The guard intentionally retains A2-15's generic
head/page/scale behavior; it is not restricted to scored TP8/page1 inputs.
Prior native evidence did not cover every such nearby configuration.

## Numerical, ownership and lifetime invariants

Only the historical page-address schedule changes. Bootstrap the current
tile's masked int64 physical rows; load current K; issue the next historical
page-ID lookup; perform unchanged QK, scalar softmax update, V load and PV;
then widen/clamp next page IDs to int64 and carry their derived rows. The next
page value's first source consumption is after the current PV. There is no
new K/V payload prefetch, new-token pre-read, host synchronization, input-derived
cross-call cache or new global scratch. Source order alone is not asynchronous
overlap evidence.

Q remains normalized then BF16-rounded then E4M3; K cache remains E4M3 for QK;
both QK sites retain FP32 destination and explicit
`max_num_imprecise_acc=0`. PV remains BF16, softmax/partials/scalar denominator
FP32. QK scaling, masks, cache quantization, all new-token operations and output
gate are unchanged. Tile widths, warps4/stages3, chunk8192, split capacity,
shape-derived allocation, separate L1 merge and its arguments remain exactly
the measured A2-15 variant. Do not mix the rejected deferred denominator,
BF16-verify custom producer, unsplit FA3 flag or another tuning choice into it.

Every historical page read is masked by `t < s_hi <= seq`; final lookahead
reads are fully masked when no next tile exists. On an empty historical split,
the bootstrap is masked and the loop does not consume its row vector. Valid
negative-page clamping and 64-bit row arithmetic match the parent; negative
unused suffix entries must remain unaccessed. Testing deliberately invalid
live pages is only a parent-parity robustness model, not an official input.

The last split still owns every new token's normalization, FP8 cache stores
and causal contribution, even if that split has empty history. It cannot be
deleted as an empty task. Lookahead touches historical page addresses only;
it does not change physical `cache_loc` ownership or the page64 append offsets.
Per-invocation output and FP32 partial allocations are unchanged. L1 uses
`SEPARATE_MERGE=True`, so the producer's ticket-completion branch is inactive;
the unchanged parallel `_merge` consumes those private partials. No new ticket
or resource lifetime is introduced. Native disjoint-cache graph overlap still
needs testing; source inspection does not establish concurrency correctness.

## What the old measurements establish

The saved native run recorded96 producer registers and raw driver spills0 for
both sources, while shared memory increased12,544→20,480 bytes. Merge SASS was
byte-identical. Native scheduling and K/V buffering changed, and the originally
intended strict source-order issue sequence was not literally preserved.
Both saved producer PTX files use FP16 QK MMA and BF16 PV MMA; neither contains
native FP8 MMA or WGMMA. These are historical source-bound artifacts, not the
new composition's compiler/resource report or an exact installed compiler Git
revision. A source-exact transplant carries the measured variant, including
its compiler-induced buffering tradeoff, not a uniquely isolated page-ID cause.

The paired historical run recomputes to1.054731138 control/candidate ratio,
or5.189108% target latency reduction. All12 measured comparisons favored it;
the six balanced pair ratios range1.050148–1.057979. Raw medians were
312.032→295.904µs. Its predeclared10% target was missed. The evidence comprises
one session, two B32/L1 cases (scored page1/unit scale and page64/scale1⁄64
boundary), eight oracle calls, four exact return-A checks and32 checked
warmup/timing replays:44 checked replays total. It has no complete official
suite, unchanged-path control, independent session or overlapping-call test.
The paired run did not capture driver or extension hashes; do not substitute
another run's identity. The old artificial$2 accounting ceiling is history,
not the current authorized free-credit scope.

## Required composition gates

Before compute, bind the actual new candidate hash and prove it has exactly
the A2-15 producer/host plus fusion entrypoint, with no further modified math,
dispatch, lifetime or split setting. Reuse source-bound address tests over
scattered page1/page64, physical page0, masked suffixes, 64/8192 tails, empty
and last-owned splits, changing lengths/page mappings at fixed addresses.
The three historical AST tests were rerun successfully during this audit;
they do not discharge these new-candidate tests or native gates.

Native validation must compare complete composition versus current fusion,
not rejected unsplit, with unchanged official tolerances/cache checks and
A→B→A, nearby/boundary/page64/scaled cases, independent calls and unchanged
fusion controls. Whole-call cold-L2 paired timing must include preparation,
producer, cache stores, partials and merge. Record the new environment and
compiled resource/code evidence if collected; prior native artifacts cannot
stand in for a new binary. A reproduced gain remains conditional and small in
the weighted whole-score model. If the prior gain does not reproduce or gates
fail, stop this hypothesis without an automatic official check. A separate
parent slot and admission are still required; this audit starts no job.
