# Independent semantics review: exact-one historical numerator rescale

Local source/artifact review supports preparing a compile-gated experiment,
with no native correctness or performance claim yet. Scope is deliberately
narrow: only the historical-loop numerator `acc` in the existing full-B32/L1
custom producer. Denominator `l_i`, row maxima, exponentials, P conversion,
new-token tail and all other routes remain literal control code. No GPU,
network, credentials, source mutation or service action was performed for this
review; this report is the only file written by the reviewer.

## Exact predicate and source contract

The verified control SHA256 is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Its historical update is at lines163–170; `row_ok = offs_m < L*G` is defined
at line99. Use the already computed FP32 `alpha`, with exact equality to1.0,
and require every valid query row to satisfy it. A scalar reduction such as
`sum((row_ok & (alpha != 1.0)).to(int32)) == 0` excludes padded rows. Comparing
`m_new == m_i` is not the same predicate: the existing approximate exponential
may round a small nonzero exponent to exactly1.0. No epsilon or logit threshold
is part of this strict variant.

For the allowed L1/G8/BLOCK_M16 specialization there are eight valid query rows;
the other eight must not control the predicate. Invalid historical columns are
already excluded by `t_ok` in the unchanged logits. A loop iteration exists
only when `s_lo < s_hi`, so it has at least one historical key. An empty split
does not run this loop; its partial and last-split new-KV rules stay unchanged.
On the first ordinary finite tile, `m_i` starts at−Inf and `alpha` is0, so the
original rescale path is taken. A new maximum on any valid row also forces the
original path unless its *computed* alpha is exactly1.0.

The proposed fast source expression `acc + dot(P,V)` and the unchanged fallback
`acc*alpha + dot(P,V)` are mathematically identical for finite valid-row carry
when the predicate holds. This alone is not a proof of identical generated
floating-point operation order. The compiler gate below is necessary.

## Existing compilation makes the carry order concrete

The archived A2-15 `_fused` definition is byte-identical to the working control's
definition after extracting the undecorated source segment: SHA256
`df3aff9cac1cd345a44b8b80f7b983d0a524a41250cb880315745f460d2a1ccc`.
These archived artifacts therefore provide relevant existing lowering evidence,
but not a fresh compile of the new conditional branch.

* `../a215/native_artifacts/a215.producer.ttgir:356–360` multiplies old carry
  `%arg23` by broadcast `alpha`, then passes that result as the **C operand**
  of the PV `tt.dot`. It does not compute PV from zero and add the old carry
  afterward.
* `../a215/native_artifacts/a215.producer.ptx:1504–1512` contains eight
  `mul.f32` carry-rescale instructions per thread in this specialization.
  Subsequent BF16 PV MMA at line1539 and onward uses those same registers as
  its accumulation C inputs.
* Denominator source `l_i*alpha + sum(P)` lowers to `fma.rn.f32` at PTX
  lines1151–1152. This experiment leaves that source line unchanged.
* The archived carry `mul.f32` has no explicit `.ftz` modifier. The existing
  alpha and P exponentials use `ex2.approx.ftz.f32`; the experiment does not
  alter them. Do not infer a universal floating-point mode from these samples.

| Archived artifact | SHA256 |
|---|---|
| A2-15 producer TTGIR | `345f1ede702bdf065cd7002bcd637af5519ec86721aee8344a54e6280d9ab32a` |
| A2-15 producer PTX | `df4a5b499b5a4116bc35d0864187c2339d3ac5b62d213955384da4174b1f19e2` |
| A2-15 producer LLIR | `da82162dfa21a5f04e2a6540b938e40ae5a1d10e4bd9cb37634d798a5b470768` |

For finite FP32 values, multiplication by positive1 is exact, including the
sign of zero and representable subnormal values under gradual underflow. If
the fast path supplies the identical carry bits to the same PV MMA sequence,
removing only that multiplication has a strong strict-equivalence basis. This
does not justify changing the dot accumulation C to zero plus a final scalar
addition: intermediate FP32 rounding would then differ. NaN payload/signaling
behavior and mismatched flush modes also prevent a universal bitwise assertion;
the official finite-output contract and native checks remain necessary.

Padded-row carries may differ because their alpha is deliberately ignored by
the predicate. Those rows have no output stores; the separate merge reads only
the valid G rows. Existing per-row max/softmax/dot operations do not reduce
across query rows. The only new across-row operation is the predicate, where
`row_ok` must be applied explicitly. This is why all-row equality without the
mask is an incorrect source contract even if present fixtures often make padded
alphas harmless.

## Required cheap compiler falsifier

Compile control and candidate with the same installed compiler, geometry,
precision, page/scale specialization and launch choices. Preserve complete
TTGIR/LLIR/PTX/SASS and resource metadata. Before any timing, verify:

1. The true path bypasses the old-carry multiplication; an equality test that
   still executes all rescale instructions does not establish the mechanism.
2. Both paths retain the baseline PV matrix instruction order and carry-C
   semantics. The fast C is old `acc`; the fallback C is `acc*alpha`. Reject a
   standalone PV-from-zero followed by final addition, duplicated executable
   PV work, changed BF16 conversion or changed dot accumulation precision.
3. The denominator FMA/reduction, logits/P/max and new-token tail remain
   unchanged. PAGE_LOOKAHEAD-false specializations reconstruct the original
   source and generated arithmetic, apart from harmless naming/locations.
4. Record reduction/shuffle/CTA-barrier, branch/predication, code size, registers,
   shared bytes and spill changes. Eight skipped per-thread multiplies are a
   small saving; a scalar all-row predicate can cost more. Correctness alone
   does not admit a timing experiment.

The new branch has not been compiled in this review. Source scope is defensible,
but the carry-order, real skipped-work and resource checks are still pending
native falsifiers. No source identity or archived artifact proves a speedup.
