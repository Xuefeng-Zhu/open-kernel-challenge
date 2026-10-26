# Independent B1 sliding SIMT source review

**PASS for the source/protocol gate; no concrete source-level blocker found.**
This is not native compile, numerical correctness, resource or performance
evidence. Only this review was written; no candidate/control/test, old artifact,
selection, checker or algorithm was edited. No GPU, compiler, service, network,
credentials or Git action occurred.

## Exact reviewed bytes

| File | SHA256 |
|---|---|
| `candidate.py` | `f5161be2e91b05fceb01e28564f96eecff471244704d036fa4cc7950dbe0ef34` |
| `test_source.py` | `a43d723e2d931aa7ffda37dfa74d23bf5f614371c7058750cdedb8471c9f000f` |
| `test_geometry.py` | `67afd5419565856ff29cd9bae3c110694901d07206d8b8ae8866e703e906978f` |
| `EXPERIMENT_CARD.md` | `7672a57a349b84f48f5e9c09ed8ecb480daad2514a792d9d0e1b0852c643f501` |
| Selected control `../fusion-lookahead-probe/candidate.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |

All14 original function ASTs and imports are retained; the prior entrypoint
is renamed `_baseline_kernel_fn`. Source tests reconstruct the entire original
producer after replacing only SIMT helper calls and partial-row stride names.
Thus normalization, RoPE, rounding, scale placement, history partition, online
softmax, new-KV ownership and final gate/merge formulas are directly bound to
the selected source rather than merely asserted equivalent.

## Isolated dispatch and SIMT arithmetic

The new guard at828–832 requires B1/L1, HQ16/HKV2, D64, window2048,
RoPE500000 and page1 or64. All other inputs use the unchanged selected route.
The host retains S16/N64, producer grid(1,2,16), four warps/three stages and
the old per-head merge. Only the producer query tile becomes M8 instead of
computing the old16-row padded tile; physical partial stride stays16.

`_simt_qk` (284–295) operates on Q[8,64] and K[N,64]. Its gather indices
[1,64] extract one valid query head, then a rank2 [N,64] FP32 product reduces
over D. `_simt_pv` (299–310) gathers P[8,N] with indices[1,N], multiplies one
head's [N,1] probabilities by V[N,64], and reduces N. The static eight-head
loops assemble [8,N] scores and [8,64] outputs; there is no [G,N,D] temporary.
Both N64 history and N16 new-token calls match their actual operand shapes.
All global K/V payload loads remain outside these head loops in source.
Native lowering can still replicate work or spill; source shape inspection
does not establish its register/shared-memory cost.
The final card correctly replaces the earlier shared-buffer/lane-layout sketch
with distributed TL arrays and compiler-chosen layouts. Its prospective
<=192-register/<=64KiB-shared/zero-local-spill/>=2-legal-CTA budget at128 threads
is an engineering gate declared before native counts, not measured residency
or a performance prediction.

Q retains selected BF16→E4M3 encoding before the helper, then exact BF16
widening. Historical/new K remains E4M3 before BF16 widening. Both operands
are converted FP32 for QK multiplication/reduction; P and V retain explicit
BF16 boundaries before their FP32 product/reduction. Softmax/max/sums/partials
stay FP32. The QK factor and live K scale are applied exactly once, and live
V scale remains in the unchanged final merge. No coarser quantization or
tolerance change is introduced. SIMT reduction order differs from tensor-dot
lowering, so bitwise equality or the original numerical bound is **not**
established by this source review.

## Cache, window and partial ownership

The cloned producer preserves `_norm_rope`, query rounding and raw/new KV
normalization/scale/cast operations exactly, including their existing dtype
boundaries. New KV computation/store remains in `split==15`, outside the
historical loop. It cannot disappear when that split has no history. The sole
valid L1 new row is stored by each KV head once; cache slot0 remains live.
Page1/64 addressing, masked physical-page loads, sliding visibility and
partial tile masks are unchanged. History covers `[max(seq-2047,0),seq)`;
adding the one causal new token gives the complete2048-or-shorter window.

All16 splits write8 valid FP32 numerator/max/denominator rows even when their
historical loops are empty. Empty non-owner splits produce0/−inf/0; the last
split includes the new token and supplies a nonempty denominator. Producer
addressing uses PARTIAL_M16; the original merge uses stride16 but reads only
g0…7 for each head/split. The256 padded rows g8…15 may remain uninitialized
because no merge load addresses them. Valid writer and reader sets are equal:
256 rows across2 KV heads×16 splits×8 query heads. The unchanged allocations
hold512 rows and135,168 total FP32 partial bytes. No counter-zero launch or
mutable global ticket is introduced on this L1 path; the unused ticket
argument aliases output under the compile-time separate-merge branch.

## Independent local evidence and remaining gates

Independently ran all5 source tests and16 geometry tests: **21 PASS**, without
importing the candidate, Torch or Triton. A second reviewer independently ran
the same21 and agreed with the host/ownership audit. Additional direct CPU
enumeration covered16 contexts (including0/55/56/63/64/65/129/2047/2048/2049,
80K and1M),10 empty-last-history cases and25,940 page1/64 address translations.
It confirmed exact history/new-token coverage, all256 valid partial writes
matching merge reads, and no padded-row read. These are symbolic/CPU checks,
not native execution.

The next falsifier is native compilation of the exact target specialization:
confirm rank2 `tl.gather` legality, actual FP32 SIMT arithmetic, generated
payload-load/reduction work, registers/shared/spills/local traffic and usable
resources. None is known from these tests. If it compiles acceptably, original
oracle tolerances, same-address fresh replay, page/window tails, scale/cache
bytes, empty-history ownership and private outputs/scratch still require a
reviewed native harness before any uninstrumented complete-call timing.
Source-private allocations alone do not prove concurrent runtime safety.
No winning latency or score is predicted, no official check is admitted, and
the selected control/A2-11 rollback remain unchanged.
