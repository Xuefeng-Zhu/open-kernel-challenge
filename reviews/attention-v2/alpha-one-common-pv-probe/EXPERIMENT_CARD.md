# Card: multiply-only conditional with common PV

**Evidence.** The first strict-alpha source produced a real MUL bypass and
correct structural PV carry, but its actual compile artifacts failed both
resource gates: producer96→128 registers and removal of the control's two-slot
V async buffer. PV appeared in two conditional regions, with two V layouts and
large loop-carried pointer/mask state. This source is preserved as rejected;
none of its kernel code was executed.

**Single causal hypothesis.** Put only the old-numerator multiply inside the
exact-alpha conditional and leave one PV consumer outside it. This may allow
the installed compiler's V consumer/pipeline analysis to retain its previous
asynchronous staging and live-state footprint. Actual dataflow supports this
question; it does not prove a unique register-allocation cause.

**Exact isolated change.** From working control3da22a…0a7cd8, candidate
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`
uses the same padded-row-safe exact alpha predicate. Its target historical path
conditionally multiplies acc, then has one unconditional `acc + tl.dot(...)`.
No PV dot appears in the inner conditional. Denominator, new-KV tail, precision,
lookahead, geometry, ownership and dispatch remain unchanged. This is not an
approximate deferred-max algorithm or a threshold/register/geometry sweep.

**Optimistic latency/weighted-score bound.** No new native result exists and
recoverable rescale critical-path time is unknown. Current full-B32-decode is
304.031893µs, with weight16 and load share0.8402623. A conditional10% target
cut would save30.4032µs and model1.4735% overall with all other paths fixed;
this is a sensitivity scenario, not a forecast. Even the unrealistic removal
of the whole target caps the single-path model at35.7580% overall. The changed
source does not itself establish any practical recoverable-time upper bound.

**Costs/resource risks.** The exact predicate reduction and branch may still
cost more than saved MULs. A conditional carry phi can still inhibit staging,
increase live state or become unconditional multiplication plus select.
No register cap or compiler-precision workaround is permitted. This local
preparation reserves no GPU credit; a separately reviewed bounded compile-only
invocation would use existing tools/image, paid cap$0 and the parent serial slot.

**Same-geometry control.** Compile exact working source and this separately
frozen candidate on the same installed compiler/image with M16/N64,
CHUNK8192/SPLITS11, four warps/three stages, strict QK/BF16 PV and the unchanged
BS16 merge. Fresh same-job control is primary; the failed variant's artifacts
explain the question rather than substitute as control.

**Cheap falsifier and predeclared gates.** Seven offline source/predicate/carry
tests must pass, followed by independent review. Only then prepare one bounded
compile/disassembly probe. Require actual alpha-one MUL bypass, PV old-carry C
and original accumulation order, original K **and V** two-slot asynchronous
buffering/token stages, no more than control96 registers, no new spill/local
access, unchanged denominator/tail/precision and exact merge SASS. Equal total
shared bytes alone do not pass the buffering gate. Reject on any failure; do
not rescue by threshold, register-cap or geometry changes.

**Later gates stay separate.** Compile success establishes only this lowering
mechanism. Actual fast/slow frequency would need a separate instrumented
diagnostic without timed claims. Original-oracle/cache/replay/isolation checks
and uninstrumented cold-L2 paired complete-call timing remain mandatory on
fresh inputs. Retain the first card's proposed≥2% target cut per new fixture,
six favorable balanced pairs and unchanged-control slowdown≤2% per fixture,
with fresh-input confirmation before promotion/check. No results have been
observed against those gates. No unchanged-source official rerun or deduplication
bypass follows from this card.
