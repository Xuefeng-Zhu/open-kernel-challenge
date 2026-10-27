# Two KV heads per CTA: correct, performance rejected

One official **off-board check** `56ff7ccdaf66426f` completed on H200.
All **16 prerequisites and 12 timed shapes PASS**, with no flags.
Score: **0.531819231x**, 95% interval [0.523841942, 0.539796519].
The selected champion remains **1.233633059x**; this candidate is not adopted.

| Full path | New candidate, us | Saved champion, us | Reference-normalized latency change |
|---|---:|---:|---:|
| Decode B1 |36.732|27.031|+35.54%|
| Decode B16 |481.488|194.272|+147.57%|
| Decode B32 |876.010|304.032|+187.57%|
| Verify B1 |47.091|29.569|+56.79%|
| Verify B16 |1076.699|197.602|+445.69%|
| Verify B32 |1921.433|380.381|+403.47%|

Each new candidate/reference comparison is the official paired, fresh-input,
cold-L2 graph benchmark. Champion comparisons above are historical reference
normalizations, not same-session champion A/B measurements. Evaluator identities
match; six unchanged sliding controls drift by at most 2.8811% after normalization.
The six weighted loads reconstruct the reported score exactly.

The implementation pairs two independent KV heads in each CTA, reads adjacent
128-byte head pairs using the actual HKV*D row stride, and initially targets one
132-SM wave: TP B1/B16/B32 splits132/9/5. This establishes address contiguity,
not measured transaction coalescing or useful occupancy. All six full profiles
contain only `_dual_full` and `_dual_merge`; the batched producer accounts for
99.5–99.7% of their instrumented kernel sums. These profiles identify the costly
stage, not its hardware cause, and are separate from benchmark timings.

13 CPU contract/host tests and independent source review passed. An initial
3D batched-dot form spilled; its source is retained as `kernel_batched.py`.
The checked explicit-2D form passed the tested offline SM90 zero-spill screen:
L1 240 registers, L8 254; ordinary merge27, S132 merge96. These are generic-page
CPU compiler results, not proof of the judge's runtime resources. Cache writes,
scales, DP, page1/page64 and long-context correctness were then checked by the
official original tolerances. Extra simultaneous-graph/ABA stress was not run.

**Stop this implementation.** The bundled mapping/split change failed; neither
zero spills nor SM-count arithmetic predicted useful speed. This does not
separately disprove every two-head mapping or prefetch design. No further sweep,
ranked entry, Modal GPU job, credential action or paid spending followed.
Offline containers were independently confirmed removed.

Frozen source commit: `f877c25c6983e006b70b9b528b79c2adabab2af6`.
Candidate SHA256: `48ea78ddcd8384fad8ef6fe0a60f27cec1e9ca5fca9fb1d1b658c4c2e5fa7e91`.
Champion SHA256: `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
`official-result.sanitized.json` preserves the result; only local output/profile
paths were made relative. The immutable raw receipt remains local, hash-bound
in `analysis.json`. `compile_results.json` and `offline/receipts/` retain the
resource evidence. `offline/v2/compile_kernel.py` is the CPU reproducer using the
existing pinned wheel described in `docs/OFFLINE_SM90_COMPILATION.md` on the
Projection branch. No claim of new explicit payload prefetch is made.
