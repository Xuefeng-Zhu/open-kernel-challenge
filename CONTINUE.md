# Active iteration — resume the same accepted jobs

R06 `b70d87fc51864276` and R07 `26ea99ee0869482a` are admitted. Never resend. Five earlier experiments
are terminal/passing. Budget7/10, three remain, zero uncertain, $0/no scored entry.
The specific R06 denial is resolved by new human approval and its sole accepted retry;
history retained in DENIED_R06.md and the ledger. Do not repeat that retry.

Poll exact accepted jobs through `python3 remote_check.py poll <experiment>` or official
`okc status <job> --wait --json`; read pending IDs from experiments/remote-budget.json.
At most two active jobs; follow every admitted job to terminal. Preserve source hashes,
record actual results separately, sanitize then push to the existing private repo.
No paid compute, scored/final entry, new credentials or material agreement.

Local experiment011 prepares native R04 small-M + R02 larger-M composition; CPU39+4
pass, GPU unverified. R07 is now admitted; monitor its exact accepted ID. Other remaining
hypotheses: measured R04 partial-kernel split32 alternative; isolate improvements then
compose only actual winning paths. Stop early if there is no justified candidate.
See STATUS.md, BEST.md and REMOTE_RESULTS.md; queue state is not completion.
