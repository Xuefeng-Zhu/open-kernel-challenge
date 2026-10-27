# Independent result review

**Valid completed diagnostic; prospective decision DROP.** Both fixtures show
a consistent ~4.8% locality effect, below the required 15% reduction.

| Seed | Random median (us) | Contiguous median (us) | Paired latency cut | Favorable pairs |
| --- | ---: | ---: | ---: | ---: |
| 135831 | 391.632 | 372.544 | 4.8148% | 6/6 |
| 135832 | 394.336 | 375.808 | 4.7931% | 6/6 |

Independently recomputed all reversed-pair ratios, median-derived cuts and the
KEEP/DROP decision from raw samples. Ordering and all 12 pairs are complete.
Local controller validation passes. Four original oracles, 16 warmups, 48
measured replays and two activity replays give **70 checked replays**.
Full-capacity bijections, post-call full-cache inverse equality, write ownership
and stable graph addresses pass; cross-layout outputs are also bitwise equal.

Raw trace decompression/hashes pass. Reprocessing through the frozen correlation
and GPU-annotation helper exactly reproduces the saved summary: five kernels
per arm and `observed_equal` full symbol/node/launch identity. One captured graph
serves both fixtures. All nine local payload/controller hashes match frozen
commit `19313efb186ba91a48fe6b9cd6565e81382f27b0`.

Evidence: `result.json` SHA256
`28dd47c0a0357aa96abc9a5ff1e640e72813c22e351e2528721d1adec55cdc70`;
`activity-trace.json.gz` SHA256
`4e33a53592759c5bd71bdf1cc6e7e46d8822398895ac7aaf4dfa7ac79c01e7c8`.

Recovering the observed cut with zero implementation cost would model only
about **0.68% overall score gain**. Setting this weighted path's full-attention
time to zero models **1.341058x** coverage potential. These are conditional
score calculations, not measured score gains or bounds on other algorithms;
the diagnostic excludes layout-construction cost. No production change follows.

App `ap-HPcxoqismBdiogLGQ1HejU` was independently recorded stopped/tasks 0,
all apps 0, at 07:05:15 UTC. The already-returned receipt records provisional
usage delta $0.04288947 and paid spending $0. No extra billing query was made.
Selected source `3da22a16…` remains unchanged. This branch is terminal DROP;
no additional experiment, candidate or ranked entry is proposed. Review used
local reads only, with no service, GPU, authentication or Git operation.
