# Experiment policy

Each new experiment gets a separate Git commit and a record under `experiments/`.
Preserve the baseline; do not squash experiments or manufacture historical commits.

Every record must state:

1. The hypothesis and exact code/configuration change (or explicitly no change).
2. Source commit and SHA256 of `submission.py`.
3. Exact reproducible commands and environment, distinguishing CPU and real GPU work.
4. Actual results, including failures, limitations and evidence filenames.
5. GPU run ID, mode, quota delta and timing when a remote run occurred.

The first commit is a snapshot of the already-existing baseline and prior local
verification. It is not a reconstruction of separate historical experiment commits.
The next experiment is the **one** user-approved off-board H200 check of that baseline.
No extra GPU check, scored submission, paid compute or kernel tuning run is authorized.

See [the evidence-based backlog](experiments/BACKLOG.md) for conditional hypotheses,
correctness gates and the source/configuration/job/result-commit evidence chain.

## Current remote authorization

The user now authorizes up to **10 NEW** meaningful off-board dense-projection checks,
$0 spend, with the prior baseline excluded. This supersedes the older one-check-only
restriction. Each admission reserves one allowance before its POST, and an uncertain
outcome stays reserved until reconciled. At most two independent jobs are active.
Use `remote_check.py`; it never creates a scored entry or retries a POST automatically.
No new material terms or credential changes are authorized. Current statuses, exact
source hashes/commits and quota snapshots live in `experiments/remote-budget.json`.
