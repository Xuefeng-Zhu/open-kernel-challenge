# Completed ownership/merge cycle; preserve A2-11

No active or uncertain GPU request. All A2-09 through A2-12 jobs are terminal.
See reviews/attention-v2/OWNERSHIP_MERGE_CYCLE.md for exact commands, evidence,
source identities and a reasoned stopping point. No job needs resubmission.

Selected attention source: reviews/attention-v2/mixed_fa3_decode_parallel_merge.py
Original source commit: 7e302d2366b194c32c433f6d6f9cc7b0b59b3928
SHA256: c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0
Job: cfa8a8670e394f80, DONE, 16+12 passed, 1.1876787561×.
95% interval: [1.1698635747, 1.2109342502]. The +2.761% point gain over
historical A2-08 has slight interval overlap; not statistically resolved.
No device-global completion counter remains. A13 passed 42 bounded-context ABA/cache replays; actual cross-graph overlap was
observed for full decode B32 only. Other paths remain concurrency-unverified.

Rejected A2-12: b24bbd05419c499a, DONE, 16+12 passed, 1.1734299856×.
Query preparation saved producer work but increased target whole-call normalized
latency 8.676%. Source/result retained independently; no speculative tuning sweep.

Projection selected P039 remains unchanged:
Worktree /tmp/okc-projection-pipeline-preflight, source
candidates/projection_router_m1_fp32_gemv.py, commit
cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da, job d5636ac8d1404abd,
1.1732389990×, 9+30 passed. Policy-only changes passed 12 admission/143 full
CPU tests. Verified private branch: bc70cf261e86ca4c4bf8d0ba63d198dad6024b39.

Current authorization: free off-board attention/projection user count caps removed.
Keep historical ledgers; actual daily limits still apply. Latest observed quota
05:55:57 UTC: attention 14/20 used, 6 left; projection 3/20 used, 17 left;
scored 0/10 each. Reset 2026-10-26 00:00 UTC. Verify again before any dispatch.
No paid compute, Modal launches, new credentials, ranked entry or serving launch.
Current authorization is in CURRENT_AUTHORIZATION.md.

A13 exact-source Modal diagnostic is complete: ap-f1GBalg3lZ0sfNmXyjJ7bu,
stopped/tasks0 verified. All 42 checked ABA/cache replays passed across seven
reduced-context cases. Actual cross-graph kernel overlap was observed for full
decode B32 only; other six profiled cases did not overlap. Default full-context
30-replay plan remains unrun. No general concurrent-safety or new score claim.
See reviews/attention-v2/A13-RESULT.md and A13-coverage.csv.
Worker35.706s/app43s; observed usage about$0.05, conservative incremental
$0.078901888 and cumulative$1.787819802, paid$0. One approved launch consumed;
no further Modal run/credential renewal authorized. Earlier A09 large-result401
remains historical; A13 small-result retrieval succeeded. Cleanup stop returned1
while task count drained; subsequent stopped/tasks0 read resolved shutdown.
No new OKC checks, algorithm changes or serving launch. Both best sources remain.
Current-source compiler resources/counters remain absent; no speculative rewrite.

Private autopush: never push attention-fa3-local original history. Use
analysis/safe_attention_export.py, then audit_push.py attention-fa3-sanitized
origin/attention-fa3-sanitized, then push_private_verified.py attention-fa3-sanitized.
Sanitized export maps original per-experiment commits without changing source
bytes. Keep credentials/local-notes untracked. Parent controls model configuration;
current tools do not expose that identity.
