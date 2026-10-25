# A2-13 complete: FP8 QK candidate retained provisionally

No active or uncertain official GPU request. Job `c2fb7b23433f4888` is DONE:
16 prerequisite + 12 timed shapes pass, 1.2109908035x SGLang,95% interval
[1.1928259414,1.2291556655]. Source `reviews/attention-v2/mixed_fa3_fp8_qk.py`,
commit `21fefb946524a25d8a06562dd28b631d49cd5175`, SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Do not resubmit. Read the exact source/card/result/12-shape CSV under that folder.

Four fallback QK paths use E4M3 queries/keys, BF16PV and FP32 sums. Full decodeB32
normalized latency fell12.671%; sliding decodeB1/B16/B32 fell5.584/3.533/2.243%.
Aggregate+1.9628%has overlappingCIs. Four-target geometric cut6.096%misses10%goal;
no path regressed>5%. Outputerrors3.935–9.903%passadaptivebounds but consume
more tolerance than BF16fallback. No PTX/nativeFP8 instruction evidence. New-source
ABA/concurrency/untouched-row stress remains unverified.

Keep A2-11 as the selected version pending a separate repeat/selection decision:
`reviews/attention-v2/mixed_fa3_decode_parallel_merge.py`, source7e302d2366b194c32c433f6d6f9cc7b0b59b3928,
SHA256c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0,
jobcfa8a8670e394f80,1.1876787561x. A13 passed42boundedABA/cachechecks on this
parent only, cross-graphoverlap observedfullD32only; do not transfer this to FP8.

Projection P039 unchanged in its separate worktree/branch:
`candidates/projection_router_m1_fp32_gemv.py`, sourcecdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da,
jobd5636ac8d1404abd,1.1732389990x,9+30pass. Remoteprojectiontip
bc70cf261e86ca4c4bf8d0ba63d198dad6024b39. Never overwrite the preserved baseline.

Actualplatformsnapshot2026-10-25T06:59:09UTC: attention15/20used5left,
projection3/20used17left,ranked0/10both. Reset2026-10-26T00:00UTC.
The exact FP8 source-transfer approval was consumed by ONE accepted check.
No future upload is authorized by this renewed payload approval. Prior denied
delegations executed nothing; this named action succeeded after explicit approval.
See CURRENT_AUTHORIZATION.md for latestscope. No paidcompute,Modal,newcredentials,
rankedentry or additionalcheck in this turn. FifteenCPUtests and officialreceipt
are separate evidence. SavedModalappsare stopped; no newModaloperationhere.
ConservativehistoricalModalcostremains$1.787819802; no change this turn.

Source/test/results privatepush is authorized for this payload. Never push the
original attention-fa3-local history. Use analysis/safe_attention_export.py,
audit_push.py attention-fa3-sanitized origin/attention-fa3-sanitized, then
push_private_verified.py attention-fa3-sanitized. Inspect outgoing history and
verify remoteSHA. The export maps source/results commits without changing bytes.
Final push receipts are kept only in ignored local-notes/auto-push-receipts.local.json;
remote verification is reported in the task, avoiding recursive status commits.
