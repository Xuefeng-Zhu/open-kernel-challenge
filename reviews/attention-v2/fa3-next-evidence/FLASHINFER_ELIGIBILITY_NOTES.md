# FlashInfer / XQA eligibility: page1 blocks the direct XQA route

**Do not implement or benchmark a direct XQA adapter for this Muse workload.**
The inspected public source rejects page1. FlashInfer availability/version in
the fixed image is **unknown**, not proven absent. The ordinary planned-wrapper
route also lacks an established way to perform its input-dependent planning
inside the required graph-timed call. No eligible drop-in route is admitted by
this audit; this is not a claim that every FlashInfer backend is impossible.

No GPU, service, installation, credential, kernel/candidate edit or commit was
performed. Only saved local receipts and public primary sources were read.

## Installed evidence versus public reference

The saved CPU-only image inventory at
`../attention-architecture/reviews/attention-v2/architecture-probe/profiling-run/inventory/result.json`
has SHA256
`f7f114b91ef6fded836f4502e15dbd4e0972a2872ac3e1142c20ad8ad847c382`.
Its distribution query covers Torch, Triton, SGLang, sgl-kernel and two CUPTI
packages. It never queries FlashInfer or its module paths. Therefore absence
from that list is not absence from the image. It reports SGLang0.0.0.dev0 and
sgl-kernel distribution metadata null even though other receipts prove its
FA3 module exists; metadata null must not be treated as import failure.

Public references below use the official **v0.7.0 tag**, which is available
and was read on2026-10-25. This is a fixed public source version, **not an
installed-version match**. Current `main`/latest docs were used only to locate
the interfaces, then the decisive constraints were checked at that tag.

## Contract matrix

| Muse requirement | Public source evidence / missing proof | Decision |
|---|---|---|
| Existing supported package/API in fixed image | Inventory did not query FlashInfer; no package path, version or wrapper hash recovered here | **Hold: installed identity unknown** |
| page1 without full KV repack | v0.7.0 XQA generator permits only16/32/64/128 and raises before build for1 | **Direct XQA reject** |
| D64, GQA8, SM90 | XQA generator accepts head dimensions divisible by16 in16..512, group ratio is specialized, and builds SM90 code for dimensions≤256 | Format alone is plausible; does not override page1 rejection |
| E4M3 KV with original Q quantization | XQA accepts E4M3 KV but public Q input is FP16/BF16. E4M3-rounded Q could be widened before passing it, but internal Q handling/scales must be audited at the actual installed version | **Precision compatibility unproven** |
| Original PV/FP32 softmax/accumulation contract | Python dtype acceptance does not prove internal QK/PV rounding, P format, accumulation order or tolerances | **Unproven; no precision substitution** |
| B16/B32 decode and L8 verify | v0.7.0 XQA has speculative `q_seq_len>1` code and mask handling; it is not universally decode-only. Exact causal L8 path/rounding still needs installed-source proof | Coverage not admitted; page1 already fails |
| Live GPU page table / lengths on replay | Direct XQA takes page tables and GPU length tensors, avoiding an ordinary wrapper's explicit `plan()` API | Plausible direct contract, but page1 and installed identity block it |
| Per-call input-dependent plan within timed graph | Standard decode/prefill wrappers document a separate cached plan; prefill implementation copies input metadata to CPU | Naive plan-then-replay adapter **reject** |
| No unsupported architecture fallback | v0.7.0 direct TRT-LLM decode `auto` chooses XQA on SM90; CuTe-DSL/Prims-TS decode alternatives identified in this source target SM100+ | No demonstrated H200/page1 escape route |

The decisive XQA guard is at
[`jit/xqa.py` v0.7.0 lines74–77](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/jit/xqa.py#L74).
The same file binds input/KV formats, dimensions and speculative specializations.
The public [XQA interface](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/xqa.py)
describes Q/cache/metadata shapes. A page-size argument cannot simply be changed
to64: Muse page1 mappings may scatter adjacent logical tokens across unrelated
physical rows. A view or coarser page table is not generally equivalent, and
full KV repacking is explicitly outside this proposal.

The [v0.7.0 decode implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/decode.py#L3051)
selects XQA for Hopper in direct `auto` dispatch, while its ordinary wrappers
separate plan from execution. It also labels CuTe-DSL and task-scheduled
Prims-TS decode as SM100-family paths. Forcing another backend string is not
evidence that the installed H200 kernel supports this exact contract.

The [v0.7.0 paged prefill planner](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/prefill.py#L2348)
creates reusable auxiliary state and explicitly disallows capture of `plan()`;
its implementation reads query/page metadata on CPU. Thus moving that planner
outside `kernel_fn` on actual live lengths would omit required per-call work
and retain input-derived schedule state. The standard wrapper cannot be
silently treated as a graph-capable GPU planner.

## Limits of this rejection and smallest useful falsifier

These findings reject the concrete direct XQA and naive planned-wrapper
adapters, not all possible FlashInfer kernels. A documented installed public
GPU planner or direct execution API might have a different contract. A purely
shape-derived capacity schedule could also require a separate proof: its task
map must not depend on a prior call's lengths, and every replay must consume
live GPU lengths/page mappings safely. None of that is established by enabling
`use_cuda_graph=True` or by a cached successful first plan.

The cheapest next step is **not a GPU timing run**. Recover a bounded read-only
package/module inventory from existing saved source evidence or a separately
authorized CPU inspection: package version, import path, public API signatures
and exact wrapper/source hashes. No installation follows from a missing module.
Compare those bytes with a pinned public release/commit. If its XQA page guard
still excludes1, stop that route immediately without compilation or upload.

For any different proposed public API, require a written source proof first:
SM90+D64/page1+E4M3 KV/GQA8; caller-preserved original Q quantization/scales;
legal PV/accumulation precision; all keys and causal L8 visibility; GPU-live
length/table replay; per-call descriptor/plan costs included; private outputs/
scratch; no host synchronization or cached input-derived task map. Only after
those checks could one representative B32 decode and B32 verify correctness/
replay test become useful. Successful library import or a published H20/page64
benchmark would not satisfy these gates or predict whole-score improvement.

No latency or weighted-score forecast is justified for an ineligible/unknown
adapter. A large kernel-family opportunity remains conditional on exact source
eligibility; this audit supplies the cheap blockers before implementation.
