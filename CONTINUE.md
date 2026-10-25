# A2-12 active; preserve A2-11 best

Current authorization: free off-board attention/projection user count ceilings
removed; actual platform quotas, serialized GPU and all other restrictions remain.
No paid/Modal/newcredentials/ranked entry. See CURRENT_AUTHORIZATION.md.

Selected attention complete source: reviews/attention-v2/mixed_fa3_decode_parallel_merge.py
original source commit7e302d2366b194c32c433f6d6f9cc7b0b59b3928
SHA256c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0
jobcfa8a8670e394f80 DONE16+12pass,1.1876787561x,
CI[1.1698635747,1.2109342502]. +2.761%versusA2-08withslightCIoverlap;
alltargetgoalsmetvsA2-10andnocontrol>5%regression. No device-global completion
counter; dedicated native concurrent graph/ABA stress remains unrun.

A2-09 localtickets passed1.14666xwithinitializationcost. A2-10 serialmerge
passed1.14055xbutregressed. A2-11 parallelmerge measured improvement is retained.
All source/results/failures/12timings/sixweightedloads are preserved.

A2-12 ACTIVE jobb24bbd05419c499a accepted2026-10-25T05:49:30UTC.
Candidate mixed_fa3_decode_qhoist.py, sourcee2dc35578f1d4aa8aa151fea27a89ed5bd82bbd8,
SHA25645003f03d471d484f5451fa35b534ced9819a886a7be8dc6c3bed339d5b45597.
Only B1/L1/sliding hoists exact repeated querynorm/RoPE/factor/finalBF16 into
an invocation-local timedbuffer. Target>=15%normalizedwholecallcut, including
extra launch. Other11computations preserved. Seven CPU/static tests pass;
native result PENDING. Poll SAME ID with check_cycle.py poll A2-12 --cycle2;
do not resubmit. Shared slot in ignored local-notes/gpu-slot.local.json.

Projection selected P0391.173238999x: candidates/projection_router_m1_fp32_gemv.py,
sourcecdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da,jobd5636ac8d1404abd.
Worktree /tmp/okc-projection-pipeline-preflight. Candidate unchanged; policy
runner/tests updated to explicit uncappedfreeauthorization.12admissiontests,
143fullCPUtests passed. Remote branch verifiedbc70cf261e86ca4c4bf8d0ba63d198dad6024b39.

Native ownership harness prepared, five CPU tests only, ZERO native stress
replays. NATIVE_OWNERSHIP_PROBE.md gives exact futurecommand and limitations.
Needs separately permitted H200execution; no newModal launch allowed.
Serving interface audit complete, no servingpatch/deployment/submission.

Private autopush: NEVER push attention-fa3-local originalhistory. Use
analysis/safe_attention_export.py; audit_push.py attention-fa3-sanitized
origin/attention-fa3-sanitized; push_private_verified.py attention-fa3-sanitized.
Source/hash mapping lives in exportmanifest. Preservecredentials/local-notes.
Model configuration is controlled byparent; tools do not expose currentidentity.
