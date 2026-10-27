# Independent source audit of the fused preparation candidate

The requested fusion is absent from the three target dispatch paths at
`ebb80b91c58032641bf5341927261eb64929f14e`. The existing full-attention fused
helper is batch-generic and already serves B1 decode and verify. The smallest
candidate changes only the first dispatch guard to add `(B, L) = (16, 1),
(16, 8), (32, 8)` under `window == rope_theta == 0`.

This audit inspected the actual standalone `candidate.py`, not a proposed diff.
It made no candidate, baseline, helper, GPU, browser, credential or billing
change. It does not establish native numerical equivalence or performance.

## Frozen sources

All repository source hashes below were computed from the authoritative commit,
except the explicitly identified standalone candidate. The upstream source
copies were read locally and their hashes match the saved
`E5-native-contract-audit.json`; they are pinned source evidence, not a fresh
identity check of the installed GPU environment.

| Source | SHA256 |
|---|---|
| `candidate.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Parent A2-13 `mixed_fa3_fp8_qk.py` | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| Selected A2-11 `mixed_fa3_decode_parallel_merge.py` | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |
| `official/baseline.py` | `37f51207283f85a65eb393b1dfb1344759d7df0bdec5450e34d24e7139f56ecc` |
| `official/reference.py` | `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b` |
| `next-plan/PRO_RESEARCH_REVIEW.md` | `d67554ae533c5df7c5de3118cefd75b1e686191a452fb00559759f43add43f7e` |
| Pinned SGLang `fused_qk_norm.py` | `99ed08f1f35bbd4174c4a40becfc7978b7a5ba0a055818e79470254e5c8a2b67` |
| Pinned SGLang `kvcache.py` | `fe75ca2abad09ee4911f1049f1c054d93f0470578f2a3e234452d19980e0ad81` |

The SGLang source is pinned to
[`a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py).
Its cache-store wrapper is
[`kvcache.py`](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/kvcache/kvcache.py).

## Exact isolation and route coverage

The complete text diff against A2-13 has one removed and one added line:

```diff
-    if (cfg["batch"] == 1 and cfg["q_len"] in (1, 8)
+    if ((cfg["batch"], cfg["q_len"]) in ((1, 1), (1, 8), (16, 1), (16, 8), (32, 8))
             and cfg["window"] == 0 and cfg["rope_theta"] == 0):
```

An independent AST check confirmed the function-name sets are identical,
every function other than `kernel_fn` is unchanged, and the complete
`kernel_fn` AST differs only in its first `if` test. Evaluating only the two
entrypoint ASTs with sentinel adapters checked 1,020 configurations: batches
1 through 34, lengths 0/1/2/7/8/9, and five full/sliding/mismatched window/RoPE
pairs. Exactly the three requested full paths changed from ordinary to fused
preparation. This is a host dispatch test, not attention execution.

Full B32 decode remains on A2-13's existing custom path. The selected A2-11 file
is preserved. A2-11 and A2-13 have identical ASTs for the four relevant functions
`_fa3_prepare_fused`, `_fa3_full_verify_fused`, `_fa3_full_verify`, and `kernel_fn`.
Thus this is the same three-target preparation hypothesis for either parent;
the explicit parent choice retains A2-13's pre-existing B32 decode behavior.

The unchanged source-segment hashes in both parents are:

| Function | SHA256 of `ast.get_source_segment`, without an added newline |
|---|---|
| `_fa3_prepare_fused` | `219f54d445b09c85df13f5ff96d5cfe3fdefae3df38404b5a310a28eff80712c` |
| `_fa3_full_verify_fused` | `491f530b10392582d5b9fdbd2724783106604b5cd00ea2eb4c8b409a2036222e` |
| `_fa3_full_verify` | `4dbd21a3ae4c318371a3a5a51a72de71299b2f5ff3a48c05e2c5d097b565764d` |
| Parent `kernel_fn` | `5e799721e2a142fbe063cad2e6422fc08614cf14114ad822a24eb614278b857a` |

## Intended numerical equivalence

For a BF16 input row `x`, both ordinary and fused preparation express
`n(x) = BF16(FP32(x) * rsqrt(sum(FP32(x)^2) / D + 1e-5))`.
For full attention, RoPE is disabled and the fused kernel's constexpr false
branch does not read `positions` or its placeholder cosine/sine pointer.
Both routes then express:

| Quantity | Unit cache scale | Non-unit cache scale |
|---|---|---|
| Query | `E4M3(n(q))` | same |
| New key | `E4M3(n(k))` | `E4M3(BF16(FP32(n(k)) / k_scale))` |
| New value | `E4M3(v_BF16)` | `E4M3(BF16(FP32(v_BF16) / v_scale))` |

The extra BF16 boundary after scaling is essential. The ordinary Torch division
has BF16 tensor output for these scalar FP32 scale tensors; the existing fused
kernel makes this boundary explicit before FP8 conversion. The scale branch
uses the same `config["kv_scale"] != 1.0` condition and reads the same live
`k_scale`/`v_scale` tensors. The query factor is folded into the unchanged FA3
softmax scale, rather than applied before query quantization.

Both paths use direct FP8 casts. The candidate introduces no clipping, coarser
quantization, accumulation-policy change or altered softmax. In particular it
does not invoke SGLang's optional fused KV-store feature: that feature reserves
slot zero and clamps FP8 stores, which would differ from this problem's ordinary
path. The established custom preparation writes live slot zero and retains the
ordinary direct-cast intent.

These formulas and dtype boundaries establish the intended source semantics.
They do not prove native bit identity: the installed normalization backend,
compiler reduction/reciprocal lowering and FP8 conversion behavior must still
be checked on GPU. Compare the actual query and new K/V bytes against the
ordinary preparation in addition to the unchanged official output/cache
tolerances. A CPU model cannot discharge this gate.

## Cache ownership and live metadata

The fused kernel derives `(token, head)` from rows of `HQ + 2*HKV` and writes
each Q element or new K/V element once for the contract's unique physical
`cache_loc` rows. Its destination is `(cache_loc[token]*HKV + kv_head)*D + col`.
There is no assumption that a logical page is contiguous with the next page.
The preparation does not consult `page_table`; the unchanged FA3 call consumes
that live tensor. Unused `-1` page-table entries are therefore a FA3 input
boundary, not a new preparation shortcut. All actual new-token `cache_loc`
values generated by the allowed reference are nonnegative and within capacity.

The fused store additionally masks negative or out-of-capacity locations. This
audit makes no equivalence claim for invalid `cache_loc` outside the specified
input contract; test negative *unused page-table entries* without manufacturing
invalid live cache destinations. Include physical slot zero, scattered pages,
page64 appends crossing a page boundary, and unchanged nonwritten cache rows.
Changing actual lengths must restore all corresponding positions, live cache
locations, FA3 lengths, page tables and scheduler metadata at fixed addresses.

The new query allocation and preparation occur inside each `kernel_fn` call.
The added dispatch guard depends only on config and introduces no cache of
inputs, outputs, schedules or intermediate values. The fused helper has no
global writable scratch. Independent overlapping calls still require a native
test with disjoint writable caches and graph allocation pools; source inspection
alone does not prove stream/concurrency behavior.

## FA3 and output-gate invariants

AST comparison of ordinary and fused full helpers confirms every FA3 keyword
expression except `q` is identical: caches, page table, cache lengths, query
offsets, maximum query length, softmax scale, causal flag, full window,
`scheduler_metadata`, and expanded scale descales. Neither helper supplies a
new split count, scheduler override, tile parameter or precision option.
The sole query-expression difference is that the fused helper already owns
the FP8 contiguous query tensor. The output `rms_norm_gate` call and final four
statements match exactly.

Full decode B16 must exercise a real generated non-None scheduler tensor on
GPU. The reference builds it from that step's lengths before the timed kernel;
the candidate must forward the current tensor without replacing, reusing or
recomputing it. Verify fixtures use `None`. A synthetic CPU sentinel proves
host forwarding only and is not a valid native scheduler encoding.

## Performance uncertainty and admission

No new preparation kernel or configuration is needed. Its fixed `BM=4` launch
has 80, 640 and 1,280 CTAs for the three TP8 targets respectively. Pinned ordinary
QK normalization chooses a larger row block for the two verify shapes, so this
fusion also changes preparation scheduling and the count of rows processed per
CTA. That is part of reusing the existing helper, not proof of a speedup or an
invitation to sweep blocks. Separate casting/store launches disappear, but
their instrumented durations cannot be subtracted from complete latency.

Admit compute only after source/hash-bound local checks and independent harness
review. Then validate all three targets on native preparation, official output
oracle, cache bytes/untouched rows, page1/page64, unit/non-unit scales, zero and
short histories, causal L8, same-address A→B→A changing metadata, and disjoint
independent calls. Keep all preparation inside the measured complete call.
Only then run balanced paired cold-L2 whole-call graph timings. Stop on a
semantic mismatch; reject the performance hypothesis if the complete call does
not improve. This audit does not request or start GPU compute, an official
check, a profiler retry or an external Pro message.
