# Best completed, correctness-passing observed candidate

Candidate: `candidates/router_hybrid_native.py`; source commit `90032a97994ad51e7779793000946545a76ba86d`.
SHA256: `d8e29be3b47aa9ad248f7e068edfe24c3432a2c138b6b464cdfe7cddbf58cace`. Job `26ea99ee0869482a`, mode check.

Official speedup **1.099197× SGLang**, 95% interval
**[1.082709, 1.115685]**.
The interval is above 1; this off-board trial measured an improvement over SGLang.
9/9 prerequisites and 30/30 scored-shape correctness checks pass. This is the highest
observed passing score so far, not a claim that every difference between candidates
is statistically resolved or that it is a final leaderboard placement.

Reference-normalized ratio over our original starter: **1.211094×**;
that is a cross-run comparison. Exact timings/environment/risks are in
`results/remote-R07-router-native-wide.json` and its CSV/experiment record.

The original `submission.py` remains unchanged and separately preserved. Job states
and the authorized remaining budget are in `REMOTE_RESULTS.md`; no scored entry or
paid resource is authorized.
