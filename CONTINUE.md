# Active correctness audit and isolated repair

Read CURRENT_AUTHORIZATION.md: user count caps for free attention/projection checks were removed; actual platform daily limits remain. Historical ledgers/results are preserved. No paid/Modal/new credentials/ranked entry.

Selected measured attention best: reviews/attention-v2/mixed_fa3_fused_full_small.py, source f01799cea791006f109f09bd064f953eff6b584b, SHA256 34b0c6bc67357aac9b3306e32010f5254bd4df2ae14b07b827b8686902fa60e9, official cf979895a3f24685 = 1.1557671624x. Existing graph results are sequential; four scored fallback paths retain a device-global counter concurrency limitation.

A2-09 isolated per-call counter repair passed all16+12, job3fb683ba982d448c,
score1.1466598821x, with measured7.40/7.68%slidingB1/B16initialization penalty.
Its source is ef5f09a, mixed_fa3_local_tickets.py. It is the invocation-local
counter baseline; dedicated native multistream/ABA remains unrun.

A2-10 job08235841c4044048 is terminal,16+12pass,1.1405486256x; rejected for
sliding regressions. Profiles locate material serial-merge activity6.6/4.8/3.2us
on slidingB1/B16/B32 and8usfullB32 (not calibrated walltime/counters).

A2-11 is ACTIVE, jobcfa8a8670e394f80, accepted05:34:57UTC; latest05:36running.
Source7e302d2366b194c32c433f6d6f9cc7b0b59b3928,
mixed_fa3_decode_parallel_merge.py, SHA256
c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0.
Only the L1 separate merge is parallelized across splits/per-query-head;
producer/partition/FA3/L8unchanged. FP32sumorder changes explicitly.
Eight CPU/ASTtests pass; GPU result PENDING. Poll this SAME job via
check_cycle.py poll A2-11 --cycle2, never resubmit. Shared GPU slot in
ignored local-notes/gpu-slot.local.json. Free user count cap removed; dailyquota applies.

Native ownership stress harness is prepared, FIVE CPU orchestration tests pass,
ZERO native stress replays run. Read NATIVE_OWNERSHIP_PROBE.md before using;
needs separately permitted H200 environment and official dependencies. No Modal.

Projection selected P039: source cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da, job d5636ac8d1404abd = 1.173238999x. Worktree /tmp/okc-projection-pipeline-preflight. P040 higher point not promoted because declared target was missed; branch af7f149 preserved. No new projection check this audit.

Private export: NEVER push original attention-fa3-local history. Run analysis/safe_attention_export.py, audit_push.py attention-fa3-sanitized origin/attention-fa3-sanitized, push_private_verified.py attention-fa3-sanitized. Preserve attribution and no credentials/local-notes in export. Model configuration remains controlled by parent; no self-claimed Astra identity.
