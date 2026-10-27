# P066 — evidence checkpoint after closing P065

Decision: **no new performance candidate clears the feasibility/value gate yet**.
Keep P046 selected at 1.1841399409×. Close P043–P065's compressed/direct-load
consumer branch. This checkpoint changes no candidate and launches no GPU,
Modal or official job. It recalculates the existing selected receipt, not new timing.

## Actual selected costs

Source SHA `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`,
job `41ba778763a5433b`. Values below are calibrated microseconds for one of each
family. Each load multiplies every family by 62; six load speedups are geometrically
combined. Reproduction: `python -B projection_p066_decision.py`; source/receipt
hashes, all timings, assertions and conditional calculations are in the output
`results/projection-p066-decision.json`. Score reproduces exactly.

| Load | Pooler | QKV | Unpooler | O | Router | Sum |
|---|---:|---:|---:|---:|---:|---:|
| Decode 1 | 18.310 | 11.951 | 4.073 | 6.897 | 5.130 | 46.362 |
| Decode 16 | 18.043 | 14.837 | 18.346 | 6.871 | 9.379 | 67.475 |
| Decode 32 | 18.310 | 15.258 | 18.625 | 6.945 | 9.648 | 68.785 |
| Verify 1 | 17.559 | 14.040 | 7.098 | 6.457 | 7.453 | 52.607 |
| Verify 16 | 23.158 | 18.193 | 20.208 | 7.575 | 10.731 | 79.866 |
| Verify 32 | 28.910 | 19.860 | 24.140 | 9.034 | 11.153 | 93.096 |

Mean within-load shares: pooler **31.05%**, QKV **23.46%**, unpooler **21.30%**,
O **11.10%**, router **13.10%**. The first three total **75.80%**. These are
score sensitivities, not a claim of HBM/computation bottlenecks.

## One retained research mechanism: work-centric Stream-K

[The original paper](https://arxiv.org/abs/2301.03598) partitions aggregate
output-tile × K-loop work across workers, rather than assigning an identical
K split count to every output tile. Boundary tiles require partial merging;
complete tiles need not. This is genuinely different from our fixed S8/S32
kernels, not an orientation, representation or tile-size sweep. No published
speedup transfers to this H200 workload.

[NVIDIA CUTLASS v3.9.2's SM90 scheduler](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp)
provides a persistent work decomposition with deterministic reduction support,
workspace barriers and cluster-aware assignment. It is not Blackwell CLC;
`IsDynamicPersistent` is false in this implementation. Public source establishes
API/design availability, not installed judge headers, successful native compilation
or absence of equivalent behavior inside cuBLASLt. No third-party source was
executed or copied. The linked source is BSD-3-Clause; any future reuse must retain
its notice. The existing documented native compilation exception remains narrow.

**Known versus unknown.** P046 profiles visibly show main+reducer for QKV
M8/16/32/128, pooler M128 and unpooler M8. Its small pooler M8/16/32 already has
one full-K BMM without a reducer. P059 recovered an existing useful vendor
split family; it did not test work-centric scheduling or prove exact vendor internals.
P051/P052 fixed the partition and excluded persistent scheduling. Saved custom
PTX cannot answer how the strong vendor baseline assigns work.

**Value bar.** The six visibly split paths cover **22.69%** of mean load time.
Uniform 10%/20% cuts there model **+2.34%/+4.84%** score; reaching +5% requires
**20.63%** reduction across that entire scope. Extending to all 15 M>1 paths of
the three big families would make 10%/20% cuts worth **+6.82%/+14.76%**, but this
coverage is speculative and is not credited to the proposal. A QKV-M128-only
20% cut gives **+0.780%**, not a strategy-level result.

**Cheap falsifier already applied.** Reusing the old custom M16/BN64/BK64/S8
pooler gives 64 output tiles × 128 K iterations = 8192 work units. Under a hypothetical
132-worker equal-cost one-CTA-per-worker model, fixed S8 requires four waves of
16 iterations, versus ceil(8192/132)=63 balanced iterations: only **1.5625%**
ideal critical-path reduction from 64. This ignores synchronization/cache/issue
cost and is not measured occupancy or a physical performance bound. P046 already
removes that pooler reducer. Therefore do not port this scheduler onto the losing
P051/P065 mainloop and call it a new high-value candidate.

## What must be learned before implementation

The minimum useful next *diagnostic*, if a later measurement stage proceeds,
is exact selected QKV M128 and pooler M128 work assignment: tile/K mapping,
actual split/work-unit count, cluster layout, mainloop versus synchronization
and finalization. Bind library executable/SASS and launch metadata to the exact
P046 call; distinguish static instruction behavior from observed timing. Existing
symbols and a 96-CTA launch are not proof of underutilization, missing overlap
or absence of Stream-K-like behavior. NCU timeouts supplied no counters and should
not be repeated unchanged.

A performance prototype is eligible only after a supported local compile/interface
and partition model demonstrate a material distinction with bounded workspace;
its fixed-scheduler control must use the same BF16/FP32 mainloop, geometry and
epilogue. Original correctness, fresh/ABA graphs and private per-call barriers
must pass. Compare both against actual P046, retain all samples, and reject if
only the inferior custom control improves. Require a credible multi-shape path
toward the +5% score bar before broad implementation. No current measured evidence
supports that condition, so **do not spend a check yet**.

P065's compiler fix worked while its whole call regressed. That result is the
reason to require baseline-bound evidence, not to invent a second weak idea.
No compression, generic more-tactics search, no-split retry, reducer-only rescue,
zero-bias wrapper, orientation or arbitrary tile sweep is reopened here.
