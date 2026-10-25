# Common-PV exact-alpha lowering: local source audit

This is a separate, uncompiled source variant. The failed alpha-one source and
its negative native result remain unchanged. No GPU, service, credential,
network, official request, native harness copy or commit is performed by this
preparation.

| Source | SHA256 |
|---|---|
| Working control / `baseline.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| New common-PV `candidate.py` | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |
| Preserved failed variant | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |

## Single-site change and unchanged scope

Start directly from the selected working control, not from the failed variant.
Replace only the historical-loop numerator assignment in `_fused`. Under the
existing `PAGE_LOOKAHEAD` constexpr, the exact valid-row predicate decides
whether to execute `acc = acc * alpha[:, None]`. A single unconditional
`acc = acc + tl.dot(...)` follows that conditional. The constexpr-false branch
retains the exact old `acc * alpha + tl.dot(...)` expression.

The true target path therefore has one common PV consumer and no PV operation
inside its multiply-only conditional. It has no separate temporary PV-from-zero
result assignment. The intended native carry is old acc for the alpha-one
edge and the original alpha product for fallback, then the same PV MMA chain.
That intended lowering must be proven in fresh artifacts; the source form is
not a native carry-order certificate.

The predicate remains
`tl.sum((row_ok & (alpha != 1.0)).to(tl.int32), 0) == 0`.
Any valid non-unit/NaN alpha selects rescaling, padded query lanes are excluded,
and no approximate threshold or rowmax-equality substitute is introduced.
The existing flag continues to admit only full B32/L1 without RoPE, including
the inherited generic head/page/scale scope. No host dispatch or signature
change is made.

Both denominator expressions and the complete last-split new-KV tail remain
byte-identical. Rowmax, alpha/P exponentials, reduction, QK/PV precision,
`max_num_imprecise_acc=0`, page lookahead, chunk/split/warp/stage policy, partial
ownership, merge, FA3 helpers and all other functions are unchanged. Every key
and value is still included; only an exact identity numerator multiply can be
bypassed.

## Why this is a distinct lowering question

The rejected variant's actual artifacts, from frozen request
`99185609d7924d18bb07de950c3d689a9ec9e649`, showed a correct branch/MUL bypass
but96→128 registers and loss of two-slot asynchronous V buffering. Its
TTGIR338–354 contains a PV dot inside each `scf.if` branch; V is synchronously
loaded and materialized in two layouts before the branch. Its loop carries
larger KV-address/mask state and only K async tokens. Merge SASS remained exact.
See`../alpha-one-rescale-probe/native/ARTIFACT_REVIEW.md` and the retained raw
artifacts.

An unconditional common PV consumer may let the compiler's loop consumer and
pipeline analysis preserve V staging. This is supported as a falsifiable
source-to-lowering explanation, not established as the unique reason for
register allocation. No allocation map identifies which value caused the
32-register increase. This variant does not relax either failed gate.

## Offline verification and remaining gates

Seven focused tests pass without Torch/Triton/CUDA imports:

- Exact working-control SHA; reversing the single source patch recovers every
  control byte, and all other function ASTs are equal.
- Folding the existing constexpr false reconstructs the complete control AST.
- The target path has exactly one PV call outside a multiply-only conditional;
  its direct old-acc addition form is retained without a separate PV result.
- Predicate checks cover exact1, adjacent values, valid NaN/non-unit fallback
  and padded NaN/non-unit lanes.
- Denominator, new-KV tail, dispatch, merge and strict precision sites remain
  unchanged; the failed candidate hash remains intact.
- A host FP32 carry-recurrence model covers signed zero, finite low-magnitude
  values and subnormals. It is not a GPU/FTZ/MMA equivalence proof.

The original semantic limits still apply: scalar multiplication by1 being an
identity does not establish native equivalence after compiler reassociation,
tensor-core carry folding, rounding/FTZ or NaN handling. The split source
multiply must lower to the original fallback carry and the common dot must
retain its original accumulation order. No compiler flag or precision change
is proposed to force a favorable result.

Readiness requires independent source review before any native harness is
copied or frozen. A subsequent single compile falsifier, if coordinated, must
show actual MUL bypass, original K **and V** asynchronous double-buffer stages,
no more than the same control's96 registers, no new spills/local accesses,
unchanged denominator/tail and exact merge SASS identity. Any failure rejects
this source; compilation alone cannot admit correctness, frequency, timing,
official checks or working promotion.
