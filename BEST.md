# Best completed, correctness-passing observed candidate

Current continuation: see MODAL_STAGE2_RESULTS.md. Unpooler019 and independent
QKV020 both completed/passed and stopped; all native correctness checks pass.
Latest cumulative Modal billing$0.21credits/net$0, conservative bound$0.3286below
$2. No new official check or push. Report before expanding experiments.

## Current cycle2 point-score best

candidates/projection_pooler_layout.py, job eeec05d897ef4946, official1.1290207051x, CI[1.1120853945,1.1459560157], all39checks passed. CI overlaps R10: no statistically resolved whole-workload improvement claimed. QKV m1SIMT has a passing Modal H200 diagnostic only, not an official score; see MODAL_RESULTS.md. R10 source and first-cycle evidence below remain unchanged.

## Preserved first-cycle evidence

Candidate: `candidates/router_pooler_hybrid.py`; source commit `5f491a34d52449f5d0c5c7985b4a0d4ecd17217d`.
SHA256: `891b2063edde9ef9c508c2983e6cf10804bf5066fc036d63912c47ddeb8fc9ee`. Job `02e099c3724e4a05`, mode check.

Official speedup **1.120732× SGLang**, 95% interval
**[1.103921, 1.137543]**.
The interval is above 1; this off-board trial measured an improvement over SGLang.
9/9 prerequisites and 30/30 scored-shape correctness checks pass. This is the highest
observed passing score so far, not a claim that every difference between candidates
is statistically resolved or that it is a final leaderboard placement.

Reference-normalized ratio over our original starter: **1.234822×**;
that is a cross-run comparison. Exact timings/environment/risks are in
`results/remote-R10-final-composition.json` and its CSV/experiment record.

The original `submission.py` remains unchanged and separately preserved. Job states
and the authorized remaining budget are in `REMOTE_RESULTS.md`; no scored entry or
paid resource is authorized.

## What this source changes

Router m<=8 uses the measured split32 native path; larger-M router uses the
contiguous wide hi+lo GEMM; pooler m==1 uses the measured SIMT GEMV. Other BF16
paths remain the original starter. All component sources and failed experiments
are preserved. No persistent input-derived state or reduced BF16 weight precision.

## Confidence and regression limits

The interval overlaps R07's [1.0827090, 1.1156850]; this highest point score is not
a statistically resolved optimum. R10 m8 router 7.4584µs is slower than isolated
R08's 6.8717µs, including the within-run SGLang-normalized comparison, despite
identical native kernel bodies. Decode32's paired load ratio is 1.0705 versus
R07's 1.0833. Preserve full shape/load evidence instead of attributing every change
to the targeted code. Controls remain available for future repeated validation.

No held-out/nearby-shape GPU rerun or final leaderboard placement is claimed. The
Mac's 57-test suite and 39-case CPU models do not establish those properties. All
ten new off-board checks are terminal; user budget is exhausted, zero uncertain,
$0 and zero scored entries. No further check is authorized by leftover service quota.
