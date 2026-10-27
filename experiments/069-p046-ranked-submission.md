# P069 — one ranked submission of selected P046

Status: done/pass; ranked leaderboard independently verified. This is a publication/verification of the
unchanged selected candidate, not an optimization experiment.

- Track: `muse_bf16_proj`.
- Ranked job: `497a0b5130e548c1`; accepted 2026-10-27 04:14:19 UTC.
- Source: `candidates/projection_pooler_column_batch2_hybrid.py`.
- Source commit: `fa56f2b3958b0c8d358c906a0b6cbb8e741a150e`.
- SHA256: `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
- Prior successful off-board job: `41ba778763a5433b`, 1.1841399408925375×;
  9 prerequisites and 30 scored shapes passed.
- P048 independently confirmed the selected three-path pooler improvement.
- Exact source bytes, syntax, track marker and successful receipt binding checked;
  independent source review found no credentials or unrelated data.
- Current rules API version `2026-10-22.1` matches the previously used version.
  No new agreement, credential, payment, Modal job or candidate edit occurred.
- Before admission: 0/10 ranked runs and 0/20 checks used today. All 22 existing
  Projection jobs were terminal off-board checks; no ranked Projection row existed.
- Exactly one request is allowed and journaled before dispatch. An uncertain outcome
  is reconciled from existing jobs, never blindly resubmitted.

Command (already sent; do not rerun):

```sh
okc submit muse_bf16_proj candidates/projection_pooler_column_batch2_hybrid.py --no-wait --json
```

Read-only monitoring:

```sh
okc status 497a0b5130e548c1 --json
okc leaderboard muse_bf16_proj --json
```

Ledger: [projection-ranked-p046.json](projection-ranked-p046.json).
Original successful receipt: [P046 check](../results/remote-P2-09-pooler-column-batch2-hybrid.json).

## Terminal result

Job `497a0b5130e548c1` started 04:24:44 UTC and finished 04:25:41 UTC on
2026-10-27. It passed all **9 prerequisites + 30 scored shapes**, with no flags,
rejections or numerical errors. Exactly one ranked request was sent.

**1.1897129768845947× SGLang**, 95% interval **[1.1718672822313256,
1.2075586715378634]**. The unchanged earlier off-board P046 score was
1.1841399408925375×; this is +0.47064% point movement, inside both intervals.
It is a repeat measurement of the same source, not a new optimization gain.
No shape moved by more than 5% after normalization against its same-run reference.
Evaluator file hashes and judge image digest match the prior P046 check.

The independent board read shows **rank 678 of 1,114**, 677 entrants ahead,
rounded speedup 1.1897× and `to_first=1.3757`. This is a timestamped provisional
ranking, not a prize or final-review guarantee. The exact observation timestamp
is retained in the JSON receipt. Ranked quota is now **1/10 used, 9 left**;
off-board checks remain **0/20 used, 20 left**. The one-request user allowance
for this action is exhausted regardless of the remaining platform quota.

No Projection job remains active or uncertain. No Modal operation, new credential,
paid resource, source edit or experimental variant was used. This report does not
make any statement about the separately owned Attention job's result or lifecycle.

## Six-load score calculation

Each of the five shapes within a load has weight 62. Sum `62 × shape_time` for
candidate and SGLang, divide reference by candidate per load, then take the
geometric mean across the six loads. Recomputed score matches the official score.

| Load | SGLang weighted µs | P046 weighted µs | Load speedup |
|---|---:|---:|---:|
| decode 1 | 3701.151975 | 2876.304015 | 1.28677357× |
| decode 16 | 4673.312131 | 4176.079652 | 1.11906681× |
| decode 32 | 4769.535882 | 4218.518692 | 1.13061864× |
| verify 1 | 4679.264032 | 3231.821869 | 1.44787189× |
| verify 16 | 5306.208071 | 4773.123205 | 1.11168471× |
| verify 32 | 6298.208003 | 5820.360208 | 1.08209935× |

All 30 shape timings, original timing samples, profile records, source binding and
comparison arithmetic are preserved in the [sanitized receipt](../results/ranked-P046-497a0b5130e548c1.json)
and [30-row CSV](../results/ranked-P046-497a0b5130e548c1.csv).
The exact raw service response remains ignored locally; infrastructure identifiers,
local paths and raw logs are not published. No benchmark samples were discarded.

Validation: source bytes match the previously reviewed source commit and hash;
independent source review passed; 9/30 counts and every pass were checked; six-load
arithmetic independently recomputed; image/evaluator hashes compared; quota and
leaderboard read separately. Existing P048/P068 native coverage and concurrency
limitations remain unchanged. No new dedicated concurrency stress was performed.
