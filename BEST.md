# Best completed, correctness-passing observed candidate

Candidate: `candidates/router_fused_view.py`; source commit `b5f6eddf6f7e6ec59692aaefb2f3600d8f4738a4`.
SHA256: `a7ba4be49f33c4eae5eba30a4b16950fcb867091cb662a9ef1ad57448a47c53c`. Job `1631810d19af4af4`, mode check.

Official speedup **1.050951× SGLang**, 95% interval
**[1.035187, 1.066715]**.
The interval is above 1; this off-board trial measured an improvement over SGLang.
9/9 prerequisites and 30/30 scored-shape correctness checks pass. This is the highest
observed passing score so far, not a claim that every difference between candidates
is statistically resolved or that it is a final leaderboard placement.

Reference-normalized ratio over our original starter: **1.157937×**;
that is a cross-run comparison. Exact timings/environment/risks are in
`results/remote-R05-wide-layout.json` and its CSV/experiment record.

The original `submission.py` remains unchanged and separately preserved. Job states
and the authorized remaining budget are in `REMOTE_RESULTS.md`; no scored entry or
paid resource is authorized.

## Candidate comparison and remaining risk

This is the highest observed aggregate point score, not a statistically established
winner over R02 or R04: their intervals overlap. R05 router m=128/256 is 11.66/12.63 µs
versus R02's 10.06/11.30 µs, so the new layout regresses those targeted cases despite
its slightly higher whole-workload score. Untouched projection ratios vary across runs.
R02's contiguous larger-M path and R04's low-M native path are the better-supported
components for a future controlled hybrid; that native/wide combination has not been
GPU-tested. The prepared library-only hybrid, pooler and split32 variants also lack
their own GPU results.

5 new checks were admitted; 5 are terminal. 5 allowances remain, but further uploads are paused because automatic approval review denied R06 before its command ran. No uncertain request or budget use resulted; see `DENIED_R06.md`.
