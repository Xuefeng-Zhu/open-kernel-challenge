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
No device-global completion counter remains. Bespoke native stress is unrun.

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

Next priority: native_ownership_probe.py on the exact selected source with a
permitted H200 environment and official dependencies. Five CPU tests pass;
zero native probe replays executed. Planned 30 replays cover L1 independent
captures/changed contents and lengths/cache; add L8 coverage separately.
The Mac has no CUDA; no new Modal/credential grant exists. Current-source
compiler resources/counters are also absent. Do not claim a measured bottleneck
or native concurrency pass from old sources/CPU stand-ins. Serving audit is
complete; no integration patch or end-to-end gain has been tested.

Private autopush: never push attention-fa3-local original history. Use
analysis/safe_attention_export.py, then audit_push.py attention-fa3-sanitized
origin/attention-fa3-sanitized, then push_private_verified.py attention-fa3-sanitized.
Sanitized export maps original per-experiment commits without changing source
bytes. Keep credentials/local-notes untracked. Parent controls model configuration;
current tools do not expose that identity.
