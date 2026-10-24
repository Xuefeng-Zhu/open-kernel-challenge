# Resume this same project and exact accepted jobs

Budget 10/10, 0 remain, 0 uncertain; $0, check mode only.
Read `STATUS.md`, `BEST.md`, `REMOTE_RESULTS.md` and `experiments/remote-budget.json`.
Never resend admitted experiments. At most two active jobs; inspect terminal results before choosing a meaningful replacement. No paid compute, scored/public/final entry, new credentials or material agreement. Uncertain requests stay reserved until reconciled read-only.

**The ten-new-check budget is exhausted. Send no further check requests.** Finish monitoring/recording accepted jobs, tests and private push only. Any new GPU experiment requires additional explicit user authorization before changing the ceiling.

## Already accepted — monitor to terminal

```sh
python3 remote_check.py poll R10-final-composition  # 02e099c3724e4a05
```

Official CLI `status <exact-ID> --wait --json` monitors may also be running; these are read-only, not replacement jobs. Do not end iteration merely because jobs queue.

For terminal results: `python3 summarize_check.py <experiment>`, inspect all correctness and timing/profile records, record failures too, then independently commit sanitized evidence. `python3 update_progress.py` refreshes derived status offline. Push only sanitized files to the existing private repository; preserve original baseline and immutable experiment sources.

Historical R06 denial and its sole approved retry: `DENIED_R06.md`. Never retry that upload again. Any new denial must be honored without alternate transfer routes.

Candidate-specific hypotheses and CPU/GPU limitations are in `experiments/`. R06 pooler split8 regressed and must not be composed into the best. QKV split8 stays local-only after that evidence. R08 validated the router split count; R09 validated the one-row SIMT pooler component; R10 is their separately committed composition with the verified wide router. Future hypotheses require remaining allowance or a new authorization; do not infer approval from this plan.

The unrelated attention attachment review remains ignored/local-only under `local-notes/attention-review/REVIEW.md`; never upload/commit it as part of this project.
