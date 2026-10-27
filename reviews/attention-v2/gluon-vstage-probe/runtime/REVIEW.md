# Independent prospective runtime semantic review

**PASS for the local worker/card contract; no concrete semantic blocker found.**
Twelve CPU tests and the default source-bound CPU preflight passed independently.
This is preparation evidence only. Actual current-source native contrast,
resource/publication/lifetime review, final frozen payload, bounded controller
and fresh serialized admission remain prerequisites to numerical execution.
This review does not infer those gates from compilation or authorize a job.

Only this report was written. No Torch/Gluon/driver import, GPU, service,
network, credential, Git operation, controller or algorithm edit was performed.
The source and support hashes were checked by the real worker helper locally;
all original helper/reference/checks and working3da bytes remain unchanged.

## Complete-call comparison

The two prototypes share the exact adapter. Its fused preparation invocation
matches working3da `_fa3_full_verify_fused`: original BM4 Q RMS/BF16-to-E4M3
rounding, scaled new-KV conversion and live cache stores, including slot0.
ROPE=False uses qkv only as the unused COS_SIN placeholder. Cache capacity is
the actual physical pool, independent of padded page-table width.

FP8 Q is widened to BF16 **inside** the captured call, then passed to unchanged
kernel6a5. Its BF16→E4M3→BF16 round trip preserves those finite E4M3 values;
the core applies QK factor and live K scale once, and V scale once. Separate
BF16 raw output feeds the same working `rms_norm_gate` with the original gate,
head dimension and epsilon. Preparation, new-KV append, cast, core and output
gate device work are inside capture/replay/timing. No input-derived taskmap or
attention result is prepared outside the call.

The actual baseline is `working.kernel_fn`, source3da, whose B32/L8/full branch
calls unchanged fused preparation plus actual FA3 and output gate. The worker
does not patch the entrypoint, add the prototype cast to FA3, replace its
automatic split/scheduler or time a serial surrogate as FA3. Prototype math
differs from FA3, so original oracle tolerances are applied separately rather
than demanding equal FA3/prototype attention bytes.

## Fixtures, page/cache semantics and replay

The three fixed TP8 B32/L8/full/page1 cases are unit and1/64-scale soft
boundaries at nominal129 context, plus original agentic/coherent unit-scale
nominal4096 context. Fresh seeds135751/135752 supply A/B QKV, maps and caches.
Boundary lengths `[0,1,55,56,63,64,65,129]*4` and reverse exercise zero history,
causal new-token visibility and partial final tiles.

Boundary mutation consistently updates int32 history/cache lengths, positions,
new-cache locations and unused table suffixes. Swapping two entries in the
original unique physical permutation makes the zero-history request's first
new token own physical row0 while retaining unique physical ownership. Prefix
truncation removes only unused logical keys; it does not alias live rows.
The unchanged preparation body explicitly accepts loc>=0, so row0 is live.

Outside capture, legal int32 page tables are padded to32×86008 with−1. All
live entries through history+8 are range-checked against the unchanged physical
pool; every unused suffix is checked−1 and cache_loc is checked against actual
`reference.new_slots`. History/PageTable pointers are int32, cache_seqlens
includes all eight new tokens, query offsets remain length8, and verify
scheduler metadata is None. No historical KV is manufactured by padding.

All three arms receive the same legal padded shape. This can affect FA3's
shape-based specialization and is explicitly a **padded nearby4096** diagnostic,
not an unpadded4096 or scored80K measurement. No score extrapolation follows.

## Correctness-first counts and independent ownership

The original `collect/check_implementation` and tolerances are unchanged.
Before any timing, the worker requires:

| Evidence | Declared count |
|---|---:|
| Original oracle calls |30 =3 cases×3 arms×3 ABA +1 own-A setup +2 submissions |
| Exact own returning-A checks |9 |
| All-arm whole K/V byte comparisons |36 |
| FA3/prototype new-KV comparisons |18 |
| Prototype-pair complete output/new-KV comparisons |9 |
| Post-submission isolation checks / extra replays |4 /2 |

Each replay restores complete fresh inputs into stable addresses, checks finite
outputs and unchanged immutable/history/unused rows, then calls the original
oracle. Returning-A bytes must match that runner's prior oracle-checked A.
Cross-arm whole-cache/new-KV equality is required; prototype-pair full raw-byte
equality is required, while FA3 output is governed by its oracle.

The independent candidate graph establishes **its own** oracle-checked A before
separate-stream A/B submission. Submission seeds its own B expectation. First
then mutates toB while second retainsB; second mutates toA while first retainsB.
All four checks therefore use the owning graph's already checked expectation;
there is no transferred/relabelled reference or uninitialized non-oracle check.
Distinct output/cache bases and exposed graph-pool handles are asserted.
Actual hardware overlap and internal scratch addresses remain unmeasured.
Any correctness, byte, immutability, address or isolation failure exits with a
bounded partial report before the timing section.

## Timing gates and bounds

Only the already oracle-checked third case is retained for timing. Four warmup
and twelve measured three-arm rounds give48 checked replays,16 prototype-pair
byte checks and32 FA3/prototype new-KV checks. Every timed replay must match its
own previously oracle-checked raw output/new-KV. Each consecutive measured pair
uses one fixture and reverses the complete arm order; three cyclic rotations
balance position in three pairs per seed/six pairs total.

The unchanged helper evicts256MiB before CUDA events and times only complete
graph replay. Restore, fixture generation/padding and validation stay outside.
There is no replay-cost subtraction, profiler, instrumented sample or tuning
sweep. Per-fixture median paired reduction>=2% and all six favorable pairs are
required **separately** for before-vs-after and before-vs-actualFA3. CPU tests
verify a pooled gain cannot hide a failed seed and one comparison cannot stand
in for the other. Both signals only support prospective follow-up; production
replacement and official admission remain explicitly false.

The worker enforces160s and500000-byte atomic partial JSON, with native imports
after exact source/installed pins. Its default is CPU preflight; --execute
depends on external technical/admission gates. The card declares a separate
one-job170/185/20/210 child/function/startup/watchdog envelope, retries0 and$.58
free reserve/paid0; controller implementation/admission is outside this semantic
review.
Broader precision,80K, other shapes and serving behavior remain untested.

## Final SHA256

| Reviewed artifact | SHA256 |
|---|---|
| `worker.py` | `bc93f136f4943e28ce3fd6be91cf67639ee4172ea9195b715bc762ac80a13d83` |
| `test_worker.py` | `df17fa88d7d71244b5603be5c35fc5b6ca0b28cef25195ed1612ce016798fec5` |
| `CARD.md` | `2a23d84264fb50d6f996df77a8a05cd0fe6f6fc5d612d7ce24e198fb0ef5adc6` |
| `../kernel.py` | `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e` |
| `../config.py` | `686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0` |
| `../../fusion-lookahead-probe/candidate.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |

All three original support files and both licenses also passed the complete
hashes pinned by worker.source_gate; none was imported by the reviewer.
