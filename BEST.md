# Best completed, correctness-passing observed candidate

Candidate: `candidates/router_fused_wide.py`; source commit `2d7f55d150ea4ca266bbc95fa70c3555c1d75cd9`.
SHA256: `016dbc6828bbf85dd6479a119b0b6f4b06cdb13fdc486fa1df16a09139537ccd`. Job `89fe00c5034a4198`, mode check.

Official speedup **1.047695× SGLang**, 95% interval
**[1.031979, 1.063410]**.
The interval is above 1; this off-board trial measured an improvement over SGLang.
9/9 prerequisites and 30/30 scored-shape correctness checks pass. This is the highest
observed passing score so far, not a claim that every difference between candidates
is statistically resolved or that it is a final leaderboard placement.

Reference-normalized ratio over our original starter: **1.154349×**;
that is a cross-run comparison. Exact timings/environment/risks are in
`results/remote-R02-router-wide.json` and its CSV/experiment record.

The original `submission.py` remains unchanged and separately preserved. Pending jobs
and the authorized remaining budget are in `REMOTE_RESULTS.md`; no scored entry or
paid resource is authorized.
