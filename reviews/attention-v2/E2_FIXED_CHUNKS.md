# E2 — fixed 8192-token historical chunks

Parent: corrected v2 E0, not the rejected separate-merge E1. Source
`fixed_chunks.py`, SHA256
`7efbb370d2828effabdc7ed717d2e7d5785f9ee0da55380fad6e591adf7c8782`,
source commit `2613417dcd806d64e3744be9e3d8773396692765`.

Only full attention at batch16/32 changes: CHUNK_SIZE=8192;
SPLITS=ceil(shape-only page-table capacity/8192). Grid and invocation-local scratch
use that static bound. Actual sequence lengths stay on device. Chunks cover all
valid historical keys exactly once. Empty chunks keep neutral m=-inf, l=0, acc=0,
publish scratch and tickets, and the last split still owns the new tokens.

Other paths retain the original partition, block64, widths, warps4, launch stages3,
BF16/FP8/FP32 math, QK norm/RoPE, gating and completion protocol. Two CPU/AST tests
pass: boundary coverage with empty chunks, and unchanged entire attention,
new-token and completion body. Command:
`.venv/bin/python reviews/attention-v2/test_fixed_chunks.py`.

Additional actual H200 evidence: A07 explicit same-address A→B→A with changed
lengths passed for parent and E2. It includes neutral non-owner chunks and a
page64/scale1/64/L8 combination. See A07_REPLAY.md for exact scope and limitations.
This probe did not measure speed. Shared original ticket counters are still unsafe
for simultaneous invocations; sequential graph replay passed.

Official check `7edded086c8a47f2` was accepted from the new six-check attention
budget. Use `check_cycle.py poll E2` and the durable ledger for current status.
Performance decisions wait for the actual terminal receipt. Preserve E0 unchanged.

## Official outcome

Passed all 16 prerequisite tests and 12 scored shapes: 0.7958818006× SGLang,
95% CI [0.7839435736, 0.8078200276]. E0 was 0.7224690513×; score ratio
1.101613694×. Separate official runs, not directly paired candidates. Target
reference-normalized shape improvements versus E0: full decode B16 +13.04%,
B32 +38.59%, full verify B16 +11.88%, B32 +15.52%. All twelve timings and
raw samples remain in E2-official-result.json. No >5% control regression;
sliding decode B1 changed -3.36%, so preserve cross-run noise caveat.
E2 is selected as E3 parent. It is still slower overall than SGLang.
