# Resume this same project and exact accepted jobs

Budget 9/10, 1 remain, 0 uncertain; $0, check mode only.
Read `STATUS.md`, `BEST.md`, `REMOTE_RESULTS.md` and `experiments/remote-budget.json`.
Never resend admitted experiments. At most two active jobs; inspect terminal results before choosing a meaningful replacement. No paid compute, scored/public/final entry, new credentials or material agreement. Uncertain requests stay reserved until reconciled read-only.

## Already accepted — monitor to terminal

```sh
python3 remote_check.py poll R08-router-split32  # ab74f25caa2e46a8
python3 remote_check.py poll R09-pooler-simt  # 97aa5690ff134369
```

Official CLI `status <exact-ID> --wait --json` monitors may also be running; these are read-only, not replacement jobs. Do not end iteration merely because jobs queue.

For terminal results: `python3 summarize_check.py <experiment>`, inspect all correctness and timing/profile records, record failures too, then independently commit sanitized evidence. `python3 update_progress.py` refreshes derived status offline. Push only sanitized files to the existing private repository; preserve original baseline and immutable experiment sources.

Historical R06 denial and its sole approved retry: `DENIED_R06.md`. Never retry that upload again. Any new denial must be honored without alternate transfer routes.

Candidate-specific hypotheses and CPU/GPU limitations are in `experiments/`. R06 pooler split8 regressed and must not be composed into the best. QKV split8 stays local-only after that evidence. Current remaining hypotheses: router split-count isolation; single-row SIMT pooler; combine only measured winning regions in a separately committed/checked final candidate if justified.

The unrelated attention attachment review remains ignored/local-only under `local-notes/attention-review/REVIEW.md`; never upload/commit it as part of this project.
