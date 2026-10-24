# Attention E0–E4 cycle in progress

Current branch: attention-v2-preflight. Read reviews/attention-v2/COVERAGE_AUDIT.md,
A07_REPLAY.md and attention-cycle-budget.json first. No new projection work.
Projection branch projection-bf16-local remains at 0eabf0e with best 1.14864995×.

New authorization: at most six additional attention-only off-board checks, failures
count. Preserve one final verification check. No ranked entry, push, new credentials,
payment/spend-cap changes, or paid compute. Projection's eight remaining checks are
separate and untouched. Never duplicate an uncertain request; inspect existing jobs.

E0 job 5b0baa286659421a passed all 16 prerequisite tests and 12 scored shapes:
0.7224690513× SGLang, CI [0.7116320155, 0.7373518839]. Source submission.py is
preserved. E2 fixed_chunks.py changes only full B16/32 to fixed 8192-token history
chunks. Job 7edded086c8a47f2 passed: 0.7958818006× SGLang, CI [0.7839435736, 0.8078200276].
Three of six new checks admitted, three remain, including final reserve. E3 pipeline.py
job d4e3fc9a2c234d72 accepted; poll exact job through check_cycle.py poll E3. E1 separate
merge was a historical negative result; no new official check spent on it.

A07 passed explicit same-address A→B→A with changed actual length vectors, cache
oracle/untouched-row checks, empty chunks and page64/scale1/64/L8. 12 checked H200
replays, no speed claim. App ap-c7hDHYlZH7Ja3VAJNM4Iur stopped/tasks0. Shared global
E0 tickets still limit simultaneous invocations; sequential replay is what passed.

E2 selected as parent from E0/E2. E3 changes ONLY full B16/32
history loop to tl.range(..., num_stages=3), launch stages stay3. Local transformation
and two AST invariants prepared. Compare emitted code/resources before a new check.
Then explicit complete candidate E4 and final official validation if justified.

Latest authoritative Modal ledger reviews/attention-v2/modal-budget.json: cumulative
conservative guard $1.708917914 of $2, remaining $0.291082086; net paid $0. Never lower
this guard from delayed billing. Short compiler/replay probe must reserve $0.25.
Existing 12h owner-only auth expires 2026-10-25 06:26:47 UTC; no renewal authorized.
A09 app ap-R6eHydPQwAqE8eg0p5lSCi stopped/tasks0; result transfer failed at
Modal BlobCreate401. No report received, so no E3 Modal GPU pass/compiler data
claim. No retry/credential or permission change. All Modal apps stopped. Source bundles must be committed and explicitly allowlisted;
no auth/config/repository contents. Known failed operation: A09 large return BlobCreate401; no GPU outcome recovered.
Do not retry that transfer or launch a replacement without resolving authorization.
Independent official E3 remains approved and accepted; no duplicate requests.
