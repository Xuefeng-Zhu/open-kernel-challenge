# Strict alpha-one historical numerator rescale: source preparation

Local source preparation passes. The next gate is a native compilation and
generated-code audit, not a correctness or performance claim. No GPU, service,
official request, credential action or source promotion was performed here.

## Exact scope

| Artifact | SHA256 |
|---|---|
| Working fusion-plus-lookahead control / `baseline.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Prepared `candidate.py` | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |

Only one historical-loop numerator assignment in `_fused` changes. Under the
existing `PAGE_LOOKAHEAD` constexpr, reduce
`row_ok & (alpha != 1.0)` across the query-row axis. If the count is zero,
use `acc + tl.dot(...)`; otherwise retain the exact original
`acc * alpha[:, None] + tl.dot(...)` expression. The constexpr-false branch
also retains that original expression. No new dispatch/configuration field,
buffer, input-derived persistent state or precision selector is introduced.

The existing host flag is true exactly for batch32, query length1, full
attention and no RoPE, independent of head/page/scale configuration. This
includes generic DP as well as scored TP8. Runtime predicate reduction must
be uniform across the CTA. The source uses query `row_ok`, not the probability
or attention weight, to exclude padded rows. Valid rows with NaN/non-unit alpha
take the slow branch. It compares the computed alpha exactly to1; it does not
infer equality from unchanged maxima or use an approximate threshold.

The first variant intentionally skips only the historical **numerator**
rescale. Both `l_i = l_i * alpha + tl.sum(p, 1)` assignments remain byte-exact,
including the denominator FMA/reduction opportunity. The entire last-split
new-token tail is byte-exact. Row maxima, safe maxima, alpha/P exponentials,
QK and PV operands/precision, `max_num_imprecise_acc=0`, lookahead addresses,
chunk/split policy, output/partial ownership, merge and all FA3 paths remain
unchanged. Every historical key/value is still loaded and included.

`test_source.py` passes six offline methods. It verifies the exact control
hash, reverse patch recovering every baseline byte, both literal slow paths,
all other function ASTs, byte-exact denominator/tail, existing dispatch flag,
predicate cases including padded NaNs and values adjacent to1, and a host
IEEE scalar identity model. No Torch/Triton/CUDA import or native claim is
part of these tests. The scalar model does not establish GPU equivalence.

## Existing generated-code evidence makes the compile gate concrete

The working control's producer is an exact A2-15 transplant. Saved A2-15
artifacts are relevant historical evidence, not a substitute for compiling
the new candidate with the installed compiler and the same specialization.

| Historical artifact | SHA256 |
|---|---|
| `../a215/native_artifacts/a215.producer.ptx` | `df4a5b499b5a4116bc35d0864187c2339d3ac5b62d213955384da4174b1f19e2` |
| `../a215/native_artifacts/a215.producer.sass` | `44dc14bbc9a1d2ab246146c7c76d76c9cb7f13b929ad4c900f18c3f3f021e8de` |
| `../a215/native_artifacts/a215.producer.ttgir` | `345f1ede702bdf065cd7002bcd637af5519ec86721aee8344a54e6280d9ab32a` |

PTX lines1512–1519 contain eight old-accumulator `mul.f32` instructions;
lines1539 onward use those registers as the C accumulator of BF16 PV MMA.
The denominator at lines1151–1152 instead uses `fma.rn.f32`. Thus the saved
compiler did not already remove all numerator rescaling, and it did not lower
PV to a separate dot from zero followed by a final scalar add. TTGIR expresses
the rescaled old numerator as the dot accumulator. Preserve that carry and
MMA order in both new paths. The source deliberately duplicates the identical
PV expression in branches rather than computing PV separately before branch
selection; generated IR must prove whether the compiler merges it correctly.

The compile-only probe must compare control and candidate TTIR/TTGIR/LLVM,
PTX, SASS, registers/shared/spill metadata and loop/dataflow, using identical
geometry/options/inputs. It must show the true path actually bypasses the
old-accumulator multiply while preserving the same PV MMA sequence and carry.
The false path must retain old behavior; the non-target specialization must
remain equivalent. Reject if the branch is predicated away into unconditional
rescaling, creates a separate PV-from-zero/final-add order, duplicates harmful
compute, introduces spill/local traffic, or breaks the established lookahead
buffering/dependency schedule. A branch in source is not sufficient evidence.

## Floating-point limits and required native checks

For finite FP32 values under matching IEEE handling, multiplication by exactly
one is an identity, including signed zero and representable subnormals. That
algebra alone does not prove this Triton source has bit-identical GPU behavior.
Compiler contraction, tensor-core carry placement, fast-math transforms,
denormal modes and NaN handling must be checked in the actual generated code.
NVIDIA documents the rounding difference between FMA and separate operations
and explicit FTZ behavior; retain the actual modifiers and instruction order
rather than assuming an abstract real-number identity. See the
[NVIDIA floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html)
and [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

The slow branch is selected for any valid NaN alpha; invalid lanes are ignored
by the predicate and their partial bytes may differ but may not influence a
valid output. Legal full-attention historical iterations have at least one
visible historical key. Empty histories/splits skip the loop; the first
nonempty tile ordinarily has alpha0 and takes the original path. Entirely
masked key columns, partial final tiles and changed live length/page tables
must still be exercised explicitly. Padded-row behavior may not be used to
admit a valid non-unit row.

Native validation must include cancellation/low-magnitude states, sign/zero
cases, original-oracle output/new-KV checks, exact cross-arm cache bytes,
fixed-address A→B→A replay, immutable/untouched rows, page1/page64,
unit/non-unit scales and empty/partial chunks. Synthetic maxima positioned
in the first/middle/last historical tile must force both predicate outcomes
without relying on a frequency model. Any output difference requires diagnosis;
passing unchanged tolerances is mandatory and cannot establish universal
bitwise identity. This variant is not the approximate deferred-max algorithm.

## Primary-source inspiration, not Hopper performance evidence

[FlashAttention-4, arXiv2603.05451v1](https://arxiv.org/html/2603.05451v1)
studies conditional rescaling as part of Blackwell-specific algorithm/pipeline
co-design. Its B200 results do not predict H200 gains. The current
[official CuTe softmax source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py)
has an SM100-specific class with an optional positive threshold that can retain
an older maximum. That changes the deferred-rescale algorithm; it is not copied
into this strict variant. The source was read as mutable `main` on2026-10-25;
the experiment derives from the locally frozen control, not a vendored FA4
implementation or an SM100 port. No approximate threshold or FP8-PV change is
part of this preparation.

The independent `SEMANTICS_REVIEW.md` supports a compile-gated probe while
explicitly withholding native bitwise-equivalence and benefit claims.
