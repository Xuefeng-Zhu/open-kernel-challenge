# Per-call compact sliding KV: prospective experiment

Status: local source preparation only. No native eligibility, numerical,
timing or official result is claimed by this card. At most one initial free
official nonranking check may be used after local source/model review. No
ranked entry, credential action or paid spending is part of this experiment.

## Question and relevant control

Can exact, per-call compaction of the small live sliding-window union into
contiguous FP8 scratch improve the complete L8 sliding call enough to pay for
its gather/store launch, scratch allocation and metadata work?

The actual best validated control is selected fusion-plus-lookahead source
`../fusion-lookahead-probe/candidate.py`, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
official nonranking check `5090b83f1d63407b`, score1.2336330588627955.
Original A2-11 remains preserved as rollback. Prepared candidate SHA256 is
`c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594`.
Reverify these exact bytes after review and before admission; no unrecorded
candidate revision is admitted.

The control already has GQA reuse, exact window pruning, FP8 Q/K encoding,
fused normalization/RoPE/cache append and automatic FA3 scheduling. Historical
Modal sliding B16/B32 L8 executions show tile64x128x64. They do not establish
the current judge binary or a missing TMA/packing feature. This is a different
data representation and address path, not a claim of new attention sparsity,
missing GQA, or a24x/3x runtime-padding loss.

This is also different from the rejected full80K cache repack: its live union
is at most2055 tokens per request. No full-context staging is proposed.

## Exact isolated change

Target all batch sizes with L8, TP8 (HQ16/HKV2), D64, page1, window2048 and
RoPEtheta500000. All other configurations retain exact selected dispatch.
The scored changed rows are sliding verify B1, B16 and B32; full attention
and all three decode loads are unchanged controls.

1. Run the existing selected fused preparation unchanged: global-position
   RoPE, original BF16/FP8 rounding, FP8 Q and live append to the original
   K/V caches. Global cache mutation and scale semantics remain unchanged.
2. Allocate invocation-local K/V scratch `[B,2055,2,64]` in E4M3 and an int32
   length vector. One packing kernel gathers the valid union from the live
   GPU page table into scratch, using32-token by128-byte work units. Copy
   encoded bytes through UINT8 views; do not requantize values. The kernel
   writes compact lengths on the GPU in the same call.
3. Call the same public FA3 with `page_table=None`, contiguous scratch K/V,
   compact lengths and unchanged Q, cumulative Q lengths, max q_len8,
   softmax scale, descales, `causal=True` and `window_size=(2047,0)`.
   Keep requested automatic split/packing policy; no override or sweep.
4. Run the selected output norm/gate unchanged. All packing and scratch work
   stays inside the complete timed call. No input-derived cross-call cache,
   saved task map or outside-call preparation is introduced.

Changing representation and effective cache capacity may also change FA3's
automatic specialization, split count or planner work. Those are coupled
consequences of this experiment. A result cannot be attributed specifically
to TMA, a page-load stall or one scheduler decision without further evidence.

## Mathematical and API eligibility gates

Let `s=cache_seqlens[b]` include the new L8 block, and let
`a=max(s-2055,0)`, `n=min(s,2055)`. Copy original absolute cache positions
`a..s-1` to relative positions `0..n-1`. Query j has original causal endpoint
`s-8+j` and compact endpoint `n-8+j`. Keeping the2047 left window maps its
compact accepted keys back exactly to
`max(s-8+j-2047,0)..s-8+j`. Since the union starts at the earliest query's
window edge, no required key is removed. **Do not disable the window:** later
verify queries would otherwise attend to as many as2055 keys.

RoPE is applied before copying with the original absolute positions; relative
scratch indices are never substituted for RoPE positions. For short/empty
history, compact lengths still include all eight new tokens and causal order
is unchanged. `cache_seqlens` and page-table interpretation must follow the
same live contract as selected FA3, with no host `.item()` or value branch.

The saved installed wrapper explicitly documents contiguous K/V-cache shape
when `page_table` is absent and forwards None to native. It accepts the same
E4M3 Q/K/V tensors and descale arguments used by the selected source. Its hash
is `610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`.
That is an API-surface eligibility fact, not proof that the current SM90 FP8
contiguous specialization is available, capture-safe or faster. No native
instruction, TMA route, resolved packing or P dtype is established here.

Cheap local falsifiers before any upload:

- Prove exact key-set equality for every query across empty/short histories,
  tile tails and2047/2048/2049 boundaries; include80K and nearby lengths.
- Model shuffled page1 mappings, partial final packing tiles, slot0, masked
  page suffixes, unique scratch stores and stable global-cache ownership.
- Audit byte-copy addressing/strides, live GPU lengths and capacity masks;
  no scratch padding may be read as a valid key and no gather may use a
  masked `-1` page index. Global cache writes remain only in unchanged prep.
- Prove fallback/source preservation and automatic FA3 policy arguments.
  Reject the source on any semantic/API mismatch; do not spend a check to
  discover an already visible mistake or relax the local checker.

## Coverage and cost arithmetic, not a prediction

Using the selected run's own fixed references, changing only the three
verify-sliding complete-call times gives:

| Uniform target latency reduction | Modeled whole score | Modeled score increase |
|---|---:|---:|
|10%|1.255616369|1.781998%|
|25%|1.291524828|4.692787%|
|50%|1.361342122|10.352273%|
|Impossible zero target time|1.567915451|27.097393%|

To model+3%,+5% or+10% whole-score improvement, all three target times would
need uniform reductions16.4704%,26.4926% or48.5887%, respectively. Unequal
gains must be evaluated with the exact six-load model. The zero-time result
is a mathematical fixed-control limit, not a hardware ceiling. This experiment
alone cannot reach the historical2.9097 leader, even under that impossible
assumption; no current leaderboard or winning forecast follows.

K+V scratch capacity at full windows is526080 /8417280 /16834560 bytes for
B1/B16/B32, plus4B per request for lengths, output tensors and existing Q.
Packing adds one launch and two scratch arrays plus a length allocation.
The fixed32-token tile gives65 CTAs per request,2080 at B32. The implementation
writes all2055 scratch rows on every call, zeroing padding: short/empty-history
calls still pay the full scratch-store capacity and launch grid, even though
their original-cache reads are masked. This is a cost, not free initialization.
Compared with directly reading the original cache, an illustrative one-gather,
one-store, one-attention-read accounting has three logical traversals:

| Batch | K+V scratch bytes | Three logical traversals /4.8TB/s |
|---|---:|---:|
|1|526080|0.3288us|
|16|8417280|5.2608us|
|32|16834560|10.5216us|

These are **not measured HBM bytes, strict latency lower bounds or predicted
costs**. The scratch can fit in H200 L2; writes/reads may hit cache, scatter
loads may transfer extra lines, and native splitting/GQA may reread operands.
Launch, allocation, synchronization, metadata and compute costs are omitted.
Do not subtract instrumented FA3 activities from calibrated whole-call times
to manufacture a savings estimate. The extra work can erase all benefit.

## First native gate and prospective acceptance rules

Modal authentication is independently unavailable at preparation time. A
native paired selected-versus-candidate session is therefore unavailable.
If authorized official-check access remains valid, **one** free `mode=check`
can first falsify SM90 FP8 contiguous eligibility, native safety/correctness
and descriptive complete-call performance. This is not a formal entry and
does not imply a same-session comparison against selected3da.

The same functional candidate includes one bounded import-time provenance
print: Torch/CUDA version strings and the public registered
`torch.ops.sgl_kernel.fwd.default._schema`, when exposed. It reads no inputs,
does not invoke `fwd`, starts no process and makes no CUDA call. A4096-byte
UTF8 schema limit and constant failure/oversize markers bound the output.
This may establish actual runtime API registration; an input schema is not
resolved PackGQA/split output, QK/PV instruction dtype or a binary hash. The
service log may omit/truncate earlier output; missing metadata stays unknown.

Before that sole upload, freeze source, pass local tests and independent
review, verify problem/quota and preserve source/receipt hashes. Original
16 prerequisites and12 scored rows must all pass without tolerance changes,
including page1/64 fallbacks, nonunit scales, nearby shapes, window-edge keys,
causal new tokens and the original output/cache checks. Valid slot0 is covered
by the local address model and source audit, not guaranteed by the random
official fixtures. Unless a native receipt observes it, targeted native slot0
coverage remains outstanding. A crash, timeout,
unsupported native route or original-oracle failure rejects this candidate.

For performance, compare against selected5090 only when evaluator identity,
contract and compatible reference measurements permit the comparison. Record
both complete evaluator dictionaries, image/code/file hashes and reported GPU
environment. A changed evaluator is an explicit comparability flag; absent a
relevant new control, performance classification becomes inconclusive rather
than assuming reference normalization removes the difference.

For row p, use the descriptive historical reference-normalized ratio
`r_p=(C_new/R_new)/(C_selected/R_selected)`. Construct a fixed-reference model
with only the three target times changed to `C_selected*r_p`; freeze all nine
untargeted times and all selected references. Prospective screening gates:

- All correctness gates pass; no target normalized latency regression exceeds3%.
- Fixed-reference modeled whole-score gain is at least3%.
- Flag any of the nine unchanged rows with more than3% absolute normalized
  drift. Such drift blocks clean performance attribution; report it instead
  of silently absorbing it into the candidate's benefit.
- If all targets fail to improve, or the modeled score gate fails, reject the
  performance hypothesis. No threshold/split/tile sweep or extra quota follows.

The new run pairs candidate with its own SGLang reference, **not with selected
3da**. Historical normalized comparisons and point scores are descriptive;
overlapping/nonoverlapping marginal intervals are not a paired significance
test. Passing these gates is ready-for-confirmation, not automatic promotion.
Keep selected1.233633 unchanged pending later actual-best paired confirmation
or an explicit review of a large effect surviving controls and source checks.
Fresh-fixture graph mutation/cache/isolation coverage remains necessary before
claiming serving robustness; an official pass alone is not that evidence.

## Stop and resource rules

No GPU launch or official upload is performed by this card's author. Root
coordinates the serialized service slot, exactly one initial check, terminal
receipts and quota accounting. Official run timeout includes compilation as
specified by the service. No Modal app, credential creation/renewal, install,
paid resource, visibility change or ranked submission is requested. If a tool
denies an operation, stop that operation and report its exact status; do not
route around it or edit source solely to evade duplicate detection.

Decision after the first result is one of: native eligibility failure,
correctness failure, performance rejection, inconclusive comparison, or
descriptive gain requiring confirmation. Never equate compilation, changed
dispatch, a missing combine or local prototype speedup with a production win.
