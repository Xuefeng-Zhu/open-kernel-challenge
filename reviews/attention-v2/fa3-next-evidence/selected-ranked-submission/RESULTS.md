# One authorized Attention ranked submission: PASS

The unchanged selected candidate was submitted exactly once to
`muse_attention_decode` through `okc submit`, not an off-board check.
Job **`2404e0d81bff4441`**, server mode **`run`**, finished **done/PASS** at
2026-10-27 04:24:00 UTC. `duplicate_of` is null. The admission receipt records
one explicit request, no source modification and no paid resource request.

Source: `reviews/attention-v2/fusion-lookahead-probe/candidate.py` (40,538 bytes),
SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
It is the same source previously validated by check `5090b83f1d63407b`, not the
rejected M64N8 experiment. The opaque service content hash also matches that
prior check; it is not the raw-file SHA256.

| Result | Value |
|---|---|
| Official speedup | **1.242924596353655×** |
| Reported 95% interval | **[1.2242807274083503, 1.2615684652989598]** |
| Reported score time | 3030.1918644535035 µs |
| Correctness/timing/profile records | 16 tests + 12 benchmarks all pass; 12 profiles |
| Return code / flags / checks | 0 / [] / ok, no flagged or rejected checks |

A separate public leaderboard read at 04:24:24 UTC confirms team
`Xuefeng-Zhu`, **rank 707 of 840**, speedup **1.2429×**, **one attempt**, and the
exact job ID above. The current leader is 3.0316×; the API reports `to_first`
2.4391. This is a current provisional leaderboard snapshot, not a final contest
position. Ranked quota is now **9 left / 1 used**; checks remain **19 left**.
The one authorized Attention ranked request has been completed; no repeat is
needed or implied.

The previous check measured 1.2336330588627955×. This new point is another
measurement of unchanged code, not a new optimization result. The runs use
fresh inputs/nodes, their reported intervals overlap, and the driver changed
from 580.159.03 to 580.178.04. Do not attribute the difference to a code gain.
The selected implementation and A2-11 rollback remain unchanged.

Receipts: `ranked-admission.json`, `terminal-status.json`,
`ranked-leaderboard.json`, `terminal-problems.json`; copied profile artifacts
are indexed separately by the root task. No additional service, GPU, credential
or Git action was performed by this result reviewer.
