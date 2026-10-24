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
chunks. Job 7edded086c8a47f2 was accepted; poll exact ID with check_cycle.py poll E2.
Two of six new checks admitted, four remain, including final reserve. E1 separate
merge was a historical negative result; no new official check spent on it.

A07 passed explicit same-address A→B→A with changed actual length vectors, cache
oracle/untouched-row checks, empty chunks and page64/scale1/64/L8. 12 checked H200
replays, no speed claim. App ap-c7hDHYlZH7Ja3VAJNM4Iur stopped/tasks0. Shared global
E0 tickets still limit simultaneous invocations; sequential replay is what passed.

Next: select the best correct parent from E0/E2. E3 will change ONLY full B16/32
history loop to tl.range(..., num_stages=3), launch stages stay3. Local transformation
and two AST invariants prepared. Compare emitted code/resources before a new check.
Then explicit complete candidate E4 and final official validation if justified.

Latest authoritative Modal ledger reviews/attention-v2/modal-budget.json: cumulative
conservative guard $1.632886938 of $2, remaining $0.367113062; net paid $0. Never lower
this guard from delayed billing. Short compiler/replay probe must reserve $0.25.
Existing 12h owner-only auth expires 2026-10-25 06:26:47 UTC; no renewal authorized.
No active Modal app. Source bundles must be committed and explicitly allowlisted;
no auth/config/repository contents. No denied or uncertain current operation.
