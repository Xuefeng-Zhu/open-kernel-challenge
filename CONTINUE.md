# A2-14 local research complete; external test pending approval

Read `CURRENT_AUTHORIZATION.md` before any external action. This stage permits
local research, tests and commits only. A2-14 has not been uploaded, checked,
pushed or run on a GPU. No new job is active or reserved by this stage. Do not
repeat any historical job.

## Next candidate and evidence

- Candidate: `reviews/attention-v2/mixed_fa3_deferred_denom.py`.
- Local source commit: `2092957b1c8312b1fc90b43f64c3ed5eed8f4658`.
- SHA256: `bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d`.
- Parent: A2-13 commit `21fefb946524a25d8a06562dd28b631d49cd5175`,
  SHA256 `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
- Only full decode B32 defers historical softmax denominator reduction. All
  other paths, precision, partition, cache writes and merge remain the parent.
- Seven source/CPU-model tests and eleven scoring-analysis tests passed.
  No native compilation, CUDA graph or performance claim follows from them.
- Report, Chinese experiment card, exact commands and pending one-check payload:
  `reviews/attention-v2/strategy/`. See `STRATEGY_REPORT.md`, `preflight.json`,
  `A2-14-CARD.zh.md` and `approval_request.json`.
- Target goal: 20% normalized full B32 latency cut; modeled total 1.24891× if
  everything else stays fixed. This is an arithmetic scenario, not a prediction.
- Known risks: longer-lived FP32 tile, changed summation order, no native
  register/spill metadata; B32 page64, non-unit scales, nearby lengths and exact
  candidate replay/concurrency/cache-unchanged-row coverage remain unverified.

## Preserved best versions and completed jobs

Selected attention A2-11:
`reviews/attention-v2/mixed_fa3_decode_parallel_merge.py`,
source commit `7e302d2366b194c32c433f6d6f9cc7b0b59b3928`,
SHA256 `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`,
job `cfa8a8670e394f80`, 1.1876787561×. A13 passed 42 bounded ABA/cache
checks on this exact parent; cross-graph overlap was observed on full D32 only.

Promising A2-13:
`reviews/attention-v2/mixed_fa3_fp8_qk.py`, source commit above,
job `c2fb7b23433f4888`, DONE, 16+12 pass, 1.2109908035×,
95% interval [1.1928259414, 1.2291556655]. The +1.9628% aggregate change
overlaps A2-11's interval. Four-target geometric cut 6.096% missed its 10%
goal. It remains provisional; no automatic best-version replacement.
Output errors passed the adaptive bounds but consumed more tolerance than
the BF16 fallback. A13 replay evidence does not transfer to A2-13 or A2-14.

Selected projection P039 is unchanged in its separate worktree/branch:
`candidates/projection_router_m1_fp32_gemv.py`,
source commit `cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`,
job `d5636ac8d1404abd`, 1.1732389990×, 9+30 pass. Last verified remote tip
`bc70cf261e86ca4c4bf8d0ba63d198dad6024b39`. Never overwrite projection's baseline.

## Historical quota, push and resource snapshots

Last actual platform quota observation: 2026-10-25 06:59:09 UTC, attention
15/20 used, projection 3/20 used, ranked 0/10 each; reset October 26 00:00 UTC.
These are historical observations, not a fresh capacity check. A2-14 used zero
new requests. There is no user-imposed free-check count ceiling, but current
code-transfer permission remains paused pending the exact approval.

A2-13's one-time authorized private push finished at 07:01:28 UTC:
`attention-fa3-sanitized` remote SHA
`e2aa2e51a7ffa69b5f108117414179abcd221a6a`. Current research commits remain
local. Never push the original attention history; earlier sanitized export
mapping and private receipts are preserved for a future authorized push.

All prior Modal apps were verified stopped. Historical conservative cost
$1.787819802 is unchanged. No new launch, credential action, paid resource or
formal competition entry occurred here. No new external action was denied in
this research stage: it was not attempted under the local-only instruction.
