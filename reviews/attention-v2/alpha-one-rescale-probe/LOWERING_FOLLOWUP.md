# Conditional MUL, common PV: local follow-up card only

**Current decision remains rejection.** Candidate`e66f50…94f764` generated a
real fast branch but increased producer registers96→128 and lost the control's
two-slot asynchronous V buffer. Its source/results must remain unchanged as a
negative experiment. This document proposes a distinct source-lowering test;
it creates no candidate, reserves no compute and authorizes no new launch.

**Actual evidence and one causal hypothesis.** Candidate TTGIR338–354 places
one PV dot in each`scf.if` branch. V is loaded/materialized in two layouts
before the branch instead of retaining the control's V async copy tokens.
The loop additionally carries large KV-address tensors. These changes are
consistent with conditional PV consumers disrupting the compiler's pipeline
analysis and increasing live state; no compiler diagnostic or allocation map
proves that as the unique cause.

The falsifiable hypothesis is that a conditional region containing **only the
old numerator multiply**, followed by one common unconditional PV consumer,
allows the installed compiler to retain the original V buffering and register
allocation while still bypassing MUL when exact alpha is1 for every valid row.

**Exact future primitive.** Starting again from working control3da22a…0a7cd8,
retain the same target constexpr and padded-row-safe exact predicate. Express
the target historical loop conceptually as:

```python
if PAGE_LOOKAHEAD:
    if not all_alpha_one:
        acc = acc * alpha[:, None]
    acc = acc + tl.dot(p.to(tl.bfloat16).to(DOT), v)
else:
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(DOT), v)
```

There is only one target-path PV expression after the conditional multiply.
The slow edge must still lower to the same alpha-product C input; the fast
edge must use the unchanged old C. Denominator, maxima, exponentials, P,
precision, geometry, lookahead, new-KV tail and merge remain byte-identical
where source structure permits. No deferred maximum, rescale threshold,
approximate predicate, explicit register cap or manual precision change.

**Latency and score ceiling.** This reformulation has no measured savings.
It seeks to avoid the observed codegen regressions, not to claim an additional
algorithmic reduction. Actual recoverable rescale critical-path time remains
unknown. With current full-B32-decode304.031893µs and its load share0.8402623,
even an optimistic10% path cut (30.4032µs) models only1.4735% overall; removing
the entire target has the deliberately loose35.7580% mathematical ceiling.
The successful compile would not establish either scenario or a bottleneck.

**Same-geometry control and cheap falsifier.** Compile the exact working
control and separately frozen follow-up source once with identical M16/N64,
CHUNK8192/SPLITS11, strict QK/BF16 PV, four warps/three stages and unchanged
merge. Use the existing bounded compile/disassembly harness after local source
and harness review. No instrumented timing or algorithm execution is required
for this question. Existing90/100/115/20/140-second bounds and at most$0.40 free-
credit reserve are a reviewable envelope, not a new reservation; paid cap$0
and the parent serialized slot remain mandatory.

**Predeclared compile gates.** The producer must have **no more than the
same-job control's96 registers**, no new spills/local accesses, and the original
K **and V** two-slot async buffering/token schedule. The branch must actually
bypass MUL rather than becoming unconditional multiply plus select. PV must
retain one common logical consumer and the original C accumulation order;
denominator/tail/precision must remain intact and merge SASS byte-identical.
Record any barriers, layouts or other resource changes rather than accepting
equal shared bytes alone as preserved buffering. If any gate fails, stop this
reformulation; do not tune thresholds, register caps or geometry to rescue it.

Only a pass would justify separately counting actual fast/slow visits, then
the original exact-source oracle/cache/replay and fresh uninstrumented paired
whole-call gates. The existing proposed2%-per-fixture target/six favorable
pairs/control≤2% gates must be frozen before those results and confirmed with
fresh inputs before promotion or a full official check. No unchanged-source
official rerun, deduplication bypass or success claim follows from this card.
