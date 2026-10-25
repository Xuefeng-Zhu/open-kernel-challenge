# Fusion plus A2-15: composition and replication probe

This candidate starts from the officially checked preparation-fusion source,
not the rejected sliding-unsplit variant. It transplants exactly A2-15's
`_fused` and `_e2_kernel_fn` source segments. Every other function and module
statement remains unchanged, including the fusion dispatch, automatic FA3
splits, precision, output gate, geometry and KV ownership. Lookahead is true
only for full B32/L1 without RoPE. Its inherited generic head/page/scale scope
is tested with an additional small DP smoke case.

| Source | SHA256 |
|---|---|
| Validated fusion control | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Measured A2-15 donor | `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c` |
| Composition candidate | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |

The independent worktree/branch is `attention-fusion-lookahead`, based on
verified private commit `05c9cc08948c42fed9e09aae1cd2b41edf966c74`.
[The fusion report](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/05c9cc08948c42fed9e09aae1cd2b41edf966c74/reviews/attention-v2/fused-preparation-probe/RESULTS.md)
has been verified through the GitHub contents API. Original A2-11 remains
selected and unchanged. The rejected split experiment is separately preserved
at private commit `51bd5db1c7836e67aa65a4e1843bb78826eb23e5`.

The source audit verifies exact A2-13→A2-15 native/paired receipt ancestry,
all25 saved native artifact hashes, and the actual composition's two-function
isolation. Removing the lookahead constexpr branches and launch flag reconstructs
the complete fusion-control AST. Integer address models cover fresh mappings,
page1/page64, empty historical chunks, tile/chunk tails, live zero, masked
padding and64-bit offsets. These are source/CPU proofs, not new native evidence.

Historical A2-15 lowered its target by5.189% in one session, with twelve favorable
comparisons and six reverse-order pairs. It missed its original10% target.
Native changes included page-load scheduling and K/V buffering; that timing did
not isolate the contribution of page addresses alone. Fusion and A2-15 changed
different functions from A2-13, but that does not prove composition correctness
or reproduce the old gain. This probe asks those two questions on fresh seeds
135711/135712, rather than reusing the historical135701/135702 fixtures.

The predeclared new replication gate is≥4% complete target latency reduction
from six balanced pair ratios, all six favoring composition, with unchanged
full-B16-verify control slowdown≤5%. This gate does not relabel the old10% goal
as met. On current official weights, transferring the old cut conditionally
models+0.752% overall; a4% cut models+0.576%. Neither is a forecast or an
achieved new score. Remaining official quota is not the acceptance criterion.

The native protocol first checks all five cases through complete entrypoints:
scored full B32 decode, scaled page64/chunk/empty-history boundary, a B4 false-
lookahead fallback, a small B32 DP true-lookahead/single-chunk smoke, and unchanged
full B16 verify fusion control. It requires30 A→B→A original-oracle checks,
ten exact return-A checks, cross-arm cache-byte equality, plus two independent
disjoint-cache concurrent-submission oracle checks. Only then may64 checked
warmup/timing replays measure target and control, cold-L2 and unprofiled.
Last-split new-token ownership, private partials and merge remain intact.

This preparation stage performs no GPU/official job, profiler, installation,
credential renewal, payment or billing-setting change. Execution requires the
parent-coordinated serialized slot, exact committed sources and fresh free-
credit/$0 paid admission. Reviewed bounds are160s worker,170s subprocess,
185s function,20s startup and210s watchdog, one H200/2CPU/8GiB invocation,
no retries, $0.58 free-credit reserve and actual paid$0. A successful local
benefit gate supports a complete nonranking official check only after slot
coordination; it does not create a ranked entry or promote selected code.

See `SOURCE_AUDIT.md`, `PLAN.md`, `runtime/PROTOCOL.md` and the independent
`REVIEW.md`. The preparation manifest records this frozen pre-execution scope;
any later results must retain their own exact source, receipts and resource
closure evidence.
