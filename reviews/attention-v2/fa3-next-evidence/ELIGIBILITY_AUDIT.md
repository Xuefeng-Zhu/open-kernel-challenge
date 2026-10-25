# High-upside attention routes: eligibility before implementation

**No new all-four-path implementation is admitted by the available evidence.**
Public XQA rejects Muse page1. FlashInfer installation/version and installed
Gluon submodule availability were never queried in the saved inventory.
Existing FA3 is eligible and already uses dynamic persistence; relabeling that
as a new scheduler is not an optimization. Public Triton3.8 Hopper primitives
make a genuinely overlapping page-gather/WGMMA prototype plausible, especially
for natural-M64 verify, but do not establish a faster replacement for FA3 or
an efficient M8 decode mapping.

This is a local/public-primary-source audit only. No service/GPU request,
browser/Pro operation, credential access, install, kernel/harness implementation
or commit was performed. The working source remains fusion plus lookahead,
SHA256`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The completed alpha-one family is rejected; this audit does not reopen it.

## Actual installed evidence and its limits

Fixed image`im-cPTbGpNiX3RarKResPgX7h` has saved evidence for H200 SM90,
Torch2.14.1+cu130, CUDA13.0, Triton3.8.0 and SGLang0.0.0.dev0. The CPU inventory
SHA256 is`f7f114b91ef6fded836f4502e15dbd4e0972a2872ac3e1142c20ad8ad847c382`,
in sibling `attention-architecture/.../profiling-run/inventory/result.json`.
It queried only Torch, Triton, SGLang, sgl-kernel and two CUPTI distributions.
It did **not** query FlashInfer or `triton.experimental.gluon` module paths.
Therefore neither is proven installed or absent. Distribution metadata null
is not import failure: the same inventory reports no sgl-kernel distribution,
while later saved evidence proves its FA3 module and extension are present.

Later same-image FA3 identity receipt SHA256
`46aed5e3fba5b978d3206de15106ce7bd5961fd63b1bc8e76ac887073fb4f0ab`
binds wrapper`sgl_kernel/flash_attn.py` to
`610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`
and loaded`flash_ops.abi3.so` to
`cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`.
Its actual upstream source revision and all transitive GPU binaries remain
unknown. These Modal receipts must not be equated to the official judge image
or its different driver. See`LOCAL_SPECIALIZATION_AUDIT.md` for exact paths,
untruncated trace symbols, resource interpretation and the missing split data.

## Contract matrix

| Route | D64/page1/FP8 KV/GQA8 and quantization | B16/B32 decode + verify | Live graph metadata/planning | Decision / exact blocker |
|---|---|---|---|---|
| Installed public SGLang FA3 wrapper | Proven by saved whole-call checks; caller supplies existing E4M3-rounded Q/scales/cache | Existing full B16 decode and B16/B32 verify pass; B32 decode currently custom | Supplied decode metadata is an allowed live input; other FA3 preparation remains in the call | **Eligible control/introspection**, not a demonstrated new mechanism; installed C++ revision and current per-path runtime splits unknown |
| Public FlashInfer v0.7.0 XQA | D64/GQA8/E4M3 KV plausible, but explicit page whitelist16/32/64/128 rejects1; Q API FP16/BF16 would also need exact round-then-widen proof | Speculative L>1 exists, so not universally decode-only; exact L8 contract unproven | Direct page-table/GPU-length interface avoids ordinary wrapper planning | **Reject this direct adapter for page1**; installed FlashInfer identity additionally unknown |
| Ordinary FlashInfer planned decode/prefill wrappers | Some paged backends may cover the format; no installed matching source recovered | Backend-specific, not established for all four targets | Ordinary `plan()` is separate/noncapturable and may read metadata on CPU; caching actual-length plans outside the timed call violates the required contract | **Do not implement the naive adapter**; a different documented GPU-live public API needs its own proof |
| Triton3.8 Gluon Hopper pointer async-copy + WGMMA | Public pointer copies can gather page1 rows; strict E4M3 encoding then exact BF16 widening can preserve the established QK contract; BF16 PV/FP32 state required | L8×G8=M64 is natural; L1×G8 has only8 valid rows, so decode needs a separate costed reuse/parallelism mapping | Can express GPU pointer gathers, explicit stages/waits and live-length masks without a per-call host planner | **Conditional prototype only**; installed Gluon path/source, precision lowering and useful overlap unverified |
| Hopper affine TMA as a general page1 gather | Arbitrarily scattered page1 addresses are not an affine tensor map | Does not solve the common paged contract by a view | Public gather example inspected for v3.8 uses Blackwell API | **Reject blind TMA/SM100 port**; use legal pointer copies or prove another supported Hopper primitive |

Any accepted future route must keep original Q normalization→BF16→E4M3
rounding before any widening, scales exactly once, FP8 cache/new-KV semantics,
all historical keys, causal L8 visibility and no lower arithmetic precision.
Input-dependent host synchronization, prior-call/task-map caching, untimed new
planning, unsupported private entrypoints and competitor-private source are
excluded. Reusing an official supplied GPU metadata tensor is distinct from
inventing and caching a new input-derived plan.

## Pinned primary-source checks

The [FlashInfer v0.7.0 XQA generator](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/jit/xqa.py#L74)
rejects page1 before build. Its speculative path does not cure that constraint.
Changing the page-size argument to64 is not a valid adaptation: page1 mappings
can scatter adjacent logical tokens into unrelated physical rows. Full KV
repacking is outside this proposal. The tagged
[prefill planner](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/prefill.py#L2348)
also does not establish graph-timed per-call planning. These are fixed public
reference sources, **not an installed-version match**. Detailed qualifications
and direct-interface checks are in`FLASHINFER_ELIGIBILITY_NOTES.md`.

The [Triton v3.8.0 Hopper API](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py#L95)
exposes asynchronous warpgroup MMA, explicit outstanding-group waits and a
precision parameter. Pointer-based async copies are imported from its Ampere
interface. This makes an explicit page-gather pipeline expressible without a
full KV repack; it does not prove the experimental module exists in this image
or that a requested strict FP8 accumulation mode lowers as intended. Use the
already quantized operands with exact BF16 widening if necessary, never lower
precision to obtain WGMMA. The version-specific limits and TMA distinction are
documented in`GLUON_ELIGIBILITY_NOTES.md`.
An actual prototype must explicitly request`max_num_imprecise_acc=0` rather
than rely on the public API's default, then prove the resulting precision in
generated code; accepting a parameter is not a native-precision certificate.

Public [FA3 v2.8.3 interface](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/flash_attn_interface.py#L678)
documents arbitrary page sizes and public GPU cache-length/scheduler interfaces.
Its [dispatch source](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/flash_api.cpp#L376)
has non-TMA paged handling, automatic GQA packing and dynamic scheduling.
This is consistent context, not proof of this installed extension's exact
revision. Existing executed symbols already identify a dynamic persistent FA3
scheduler. The installed public wrapper's `return_softmax_lse=True` can expose
auxiliary shapes through its normal return path without changing fwd arguments;
the meaning of returned extra fields still needs an actual schema mapping.
Do not replace this with direct undocumented operator/kernel entry calls.

## Whole-score scale: require a family mechanism

Using calibrated official5090 rows, the full-path share of each load is
82.9176% decode16,84.0262% decode32,75.9700% verify16 and82.8085% verify32.
For fractional cuts`r_i`, holding all other paths fixed, modeled score factor
is`product((1-f_i*r_i)^(-1/6))`.

| Conditional cut in all four full paths | Modeled aggregate factor | Score from current1.233633 |
|---:|---:|---:|
| 20% | 1.125840 | 1.388873 |
| 30% | 1.205370 | 1.486984 |
| 50% | 1.417341 | 1.748478 |

These are scenarios, not observed recoverable losses. The parent's public
leader snapshot2.9097 at22:57UTC is about2.35864× the current point score;
it is provisional and cross-run. Even a broad family improvement cannot be
treated as a forecast of closing that gap. As a deliberately optimistic
planning scenario, substituting the official nominal63.2/126.4/63.4/126.8µs
full-path floor figures gives about2.04189 aggregate with other paths frozen.
Those nominal streaming/compute models are not measured HBM bounds or a proof
of an impossible score. They show why decode-only tweaks, tiny epilogues and
one-path frequency wins are insufficient as the central strategy.

## Smallest useful next falsifiers

1. **Resolve eligibility cheaply before kernel code.** Existing saved data
   cannot answer whether FlashInfer or Gluon is installed. A separately scoped
   CPU-only inspection of the existing image should record module origins,
   package versions, public signatures and exact wrapper/source hashes, with no
   install. If absent, stop that route. This audit performs no such service call.
   If installed XQA still excludes page1, stop before compilation. Require a
   documented alternative API if claiming graph-capable GPU planning.
2. **Recover current FA3 work before attributing losses.** The separate trace
   audit found that current official full-decode-B16 benchmark is stable near
   194µs while its profile forward activity changed from185.7 to297.3µs. It does
   not establish a60% scheduling opportunity. Use one future same-fixture,
   same-graph, unprofiled-before/after activity/auxiliary-shape diagnostic to
   obtain complete current symbols, actual metadata and split/task evidence,
   without counters or changing timed algorithms. See its precise falsifier in
   `LOCAL_SPECIALIZATION_AUDIT.md`.
3. **If Gluon is present, isolate true overlap rather than repeat WGMMA
   existence.** Start with natural-M64 verify, D64/page1/GQA8 and a few live-
   mapped KV tiles. Compare equal-geometry, equal-precision immediate-wait and
   explicit double-buffer/async QK-next-tile versus current softmax/PV schedules.
   Include all gather/conversion/buffer management in both arms. First compile
   must prove multiple useful outstanding groups, safe buffer lifetimes, real
   wait placement and no precision reduction/spills—not merely WGMMA opcodes.
   Only then original-oracle/tail/mask/replay checks and a small native mechanism
   comparison can test overlap. A full-call comparison must still beat current
   FA3 on fresh B16/B32 verify inputs before any dispatch change.
4. **Decode is a separate admission problem.** M8 padded into M64, grouped
   unrelated requests or block-diagonal packing can add substantial masked
   arithmetic. A common architectural family is a goal, not an established
   four-path implementation. Require explicit useful-work, gather/I/O, padding,
   split/merge and residency cost models plus a same-geometry raw control before
   coding a decode mapping. Do not transfer verify gains or H20/D128/page64
   benchmark numbers to this contract.

The prior M64 BF16 producer's immediate-wait pipeline failed its tested
whole-call comparison; that rejects that pipeline, not all asynchronous Hopper
architectures. Conversely, public async capability and a larger tile are not
evidence that a new implementation will beat FA3's already asynchronous,
persistent mainloop. No next GPU candidate is ready from this audit alone;
the exact blockers and cheap tests above prevent another unsupported leap.
