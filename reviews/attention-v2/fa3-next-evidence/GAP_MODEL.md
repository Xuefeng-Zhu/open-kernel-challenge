# Attention gap: the next target must span more than large full attention

The public leader's rounded score is **2.9097**, versus our validated working
candidate's **1.233633059** nonranking check. Their ratio is **2.358643×**:
matching that point estimate requires an equivalent uniform **57.602741%**
reduction in complete weighted load time. This is the scale of the remaining
gap, not evidence that every individual path must improve by that percentage.

The public snapshot is from **2026-10-25 22:57:03 UTC**, leader team
**pjmullin3**,746 ranked entries, `you:null`. No rank is assigned to our
off-board check. Competitor private results, per-shape times, kernels and
confidence intervals were neither requested nor inferred. The public number
is rounded and is not a statistically exact target.

## Fixed-reference model

Use our own official twelve calibrated path times and their same-run reference:

`score = geomean_6_loads((46×R_sliding+16×R_full)/(46×C_sliding+16×C_full))`.

References and untargeted times stay fixed. This exactly reconstructs our
reported1.2336330588627955 score, CI[1.215128563,1.252137555]. Every scenario
below changes explicit candidate times only. No public per-shape reference or
competitor timing is invented.

| Selected paths, all others frozen | Score if selected times halve | Uniform cut to match2.9097 | Conditional zero-selected-time limit |
|---|---:|---:|---:|
| Four large full: D16,D32,V16,V32 |1.748478|88.736782%|3.823976|
| Three large FA3 full: D16,V16,V32 |1.596670|Unattainable in this model|2.816760|
| All six full paths |1.889069|81.081901%|4.571865|
| Two full verify: V16,V32 |1.460356|Unattainable in this model|2.098173|

The limits are mathematical thought experiments with the selected times tending
to zero and every other path frozen. They are **not hardware ceilings**. The
three large FA3 full paths alone cannot close the gap under those fixed-control
assumptions, even with impossible zero compute time. Improving only full verify
has an even lower model limit.

The apparently sufficient88.74%/81.08% full-path cuts would put all four large
full paths below their published TASK floors. They are useful counterfactual
requirements, not credible implementation targets.

## What the TASK floors do and do not support

Setting all six full paths to their TASK floors while leaving sliding unchanged
models a score of only **2.358505**, still below2.9097. After that optimistic
full-path assumption, sliding must additionally lose **32.839200%** of its
complete-call time uniformly to match the rounded leader point. All resulting
sliding times remain above their corresponding TASK floors.

TASK floors are published bandwidth/compute estimates for the nominal shapes.
They are not measured achievable complete-call latency, an architectural proof
of a lower bound, or a model of launch, schedule, normalization and merge costs.
No claim of actual attainability or a true global maximum follows. The
arithmetic nevertheless shows why a full-mainloop-only plan has insufficient
upside under these assumptions: sliding and/or B1 paths must also contribute.

## A broad engineering target that can close the arithmetic gap

One internally consistent scenario is:

- Halve all eight B16/B32 paths, so each of the four large loads improves2×.
- Cut all four B1 paths by70%, so each of the two B1 loads improves3.333×.

The six-load score becomes **2.925267**, from the exact factor
`(2^4 × (1/0.3)^2)^(1/6)`. All twelve modeled path times remain above TASK
floors. With the eight large paths halved, the exact B1 reduction needed to
match2.9097 is **69.515902%**, using
`cut_B1 = 1 − 4/(leader/current)^3`.

These are **engineering targets, not a forecast or a proposed implemented
algorithm**. The70% scenario barely clears a rounded historical point and
provides no robust winning margin. Staying above coarse TASK estimates does
not establish feasibility. An evidence-led next architecture plan must explain
how it can affect multiple large paths and the B1 load/launch overhead, and
what tests would falsify that mechanism before spending GPU or official quota.
Another small isolated path gain does not address the quantified gap.

## Reproduction and provenance

Run `python3 reviews/attention-v2/fa3-next-evidence/gap_model.py` followed by
`python3 reviews/attention-v2/fa3-next-evidence/test_gap_model.py` from this
worktree. An alternate saved public receipt can be passed with
`--public-receipt`; the script performs no network or service call. Five CPU
tests verify measured-score reconstruction, homogeneous scaling, solved roots
and unattainable groups, the independent broad-scenario/B1 closed forms,
TASK-floor comparisons and sanitized public metadata. Separate independent
local arithmetic reproduced the reported values.

`gap_model.json` retains every own path/reference/floor input, group and
scenario result. Its public summary contains only observation time, leader
team/rounded score, ranked count and our absent rank; no competitor job IDs or
leaderboard rows are copied. Provenance hashes are:

| Input provenance | SHA256 |
|---|---|
| Recorded public CLI response bytes | `eb728b106f0c1034609b5c02b5751b780a34ab39295fa118af2dbedb6be8404b` |
| Recorded CLI binary | `01a6b091bc827a54403bdb32db721f275489f07c2d3490199f5457e4da27b4e6` |
| Local receipt containing that public response | `55239187a95c2a3415f312a771c4a29700fa47cfe754a16e57ba5587d85828d8` |
| Published own official result | `a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb` |
| Local official TASK specification | `fdc57e97579dd8579bb6cc284e454578962d30a65b1b7a526719ce220b0bba76` |

Only local model/script/test/report files were added. No candidate source,
credential, billing setting, service, GPU job, official check or formal entry
was changed or started by this audit.
