# Evidence-based stopping review after the prologue series

Finish the accepted A2-08 and retain/reject by its declared goal. Do not start
another request merely to use the two remaining authorizations. One is retained
for meaningful final verification/repair; an identical-source duplicate adds no
measurement and is not a useful final check. This is a bounded engineering pause,
not proof that higher performance is impossible or a reset of user budgets.

## Attention: remaining value is mostly in the attention core

Saved A2-07 receipt, jobba9e02c1d6584cc4:

| Full path | Adjusted call us | Profiled FA3 forward us | Preprocessing activity us |
|---|---:|---:|---:|
| Decode16 |201.746|189.7|7.6|
| Verify16 |206.031|192.3|8.6|
| Verify32 |391.681|378.3|9.2|

FA3 forward is92–95% of reported activity. This is timing evidence, not proof of
memory bandwidth/occupancy/register bottlenecks. Profile rows include PDL overlap
and cannot be treated as removable wall-time bounds. The only straightforward
extension is wider full-prologue fusion, preserving FA3. Hypothetically saving5us
on EACH of these three paths gives+0.829% aggregate;7us gives+1.167%. These are
frozen-parent six-load models, NOT measured savings, hard bounds or promises.
A single remaining exploratory check is unlikely to resolve such a small total.

Full decode32 remains E2:366.185us vsSGLang386.768us in this same receipt. Do not
replace that faster core with FA3 merely to reuse the new prologue. E1 separate
merge lost overall (six-weighted-load paired ratio0.91640); E3 pipelining gained
only0.25%, inconclusive. Historical E0/E1 and A07 E2 full-verify register counts
are different specializations from current FA3 and E2decode32, so they cannot
justify a current register/occupancy/traffic claim or a blind tile/split sweep.

A next core optimization needs a specific exact-path implementation gap and
compiler/traffic/stall evidence, then a single-factor test. No such defensible
high-value mechanism is established by the saved evidence. No new profiler
permission, credential, Modal launch or source execution is implied here.

## Projection: preserve P039 and five checks

P040 is correct but its6.51% target cut missed15%; target-only aggregate effect
is about0.16%. P038 TMA regressed but its copy control was never measured, so TMA
alone is not a demonstrated cause. QKV single-pass loses55.10vs21.89us despite
WGMMA/zero reported spills. Other layout variants failed or were parity. Current
exact-target hardware evidence is insufficient for a new large-M mechanism;
M34 metrics belong to other kernels. See projection-pipeline-preflight branch's
experiments/PROJECTION_BOUNDED_REVIEW.md for full saved evidence.

No new projection request was sent in this resume. Preserve all sources/results,
source hashes and both ledgers. Independent read-only reviews support this pause.
