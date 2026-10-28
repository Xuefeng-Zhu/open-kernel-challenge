# Experiment index

Use source-bound results rather than a filename or a point score alone. **Check**, **ranked**, **native diagnostic**, **CPU model** and **compile-only** are separate evidence classes.

| Track / question | Entry point |
|---|---|
| Attention current control and prior rollback versions | [Attention guide](reviews/attention-v2/README.md) |
| Attention v298 official score and all 12 shapes | [2.618804× check](reviews/attention-v2/vjeux-v298-probe/official-check/RESULTS.md) |
| Attention CUDA Graph/scratch validation | [Serial replay stress](reviews/attention-v2/vjeux-v298-probe/graph-stress/RESULTS.md) |
| Attention baseline versus changed page scheduling | Paired native test (local evidence; `reviews/attention-v2/vjeux-v298-probe/page-bank-probe/RESULTS.md`) · paired instruction-level profile (local evidence; `reviews/attention-v2/vjeux-v298-probe/profile-04-page-bank-paired/RESULTS.md`) |
| Attention earlier successful preparation-fusion composition | [Fusion + lookahead](reviews/attention-v2/fusion-lookahead-probe/RESULTS.md) |
| Projection selected P046 and source/score receipts | [BEST.md][p-best] · [ranked receipt][p-ranked] |
| Projection P046 same-H200 confirmation | [P048 paired correctness/timing][p-paired] |
| Projection recent diagnostic and rejected experiments | [STATUS.md][p-status] · [experiments directory][p-experiments] |
| Rejected ideas, compiler/setup failures and inconclusive tests | [Short failure index](FAILED_EXPERIMENTS.md) |

Existing full records and raw artifacts remain at their current paths. New failure summaries belong in the short index with a link to evidence; this navigation does not duplicate reports or authorize a new run. Unpublished local diagnostics are labelled explicitly and are not included in this documentation-only publication.

[p-best]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/d55496b422e996807614a2661346d589144bc1ba/BEST.md
[p-ranked]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/d55496b422e996807614a2661346d589144bc1ba/experiments/069-p046-ranked-submission.md
[p-paired]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/d55496b422e996807614a2661346d589144bc1ba/experiments/048-p046-paired-confirmation.md
[p-status]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/d55496b422e996807614a2661346d589144bc1ba/STATUS.md
[p-experiments]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/tree/d55496b422e996807614a2661346d589144bc1ba/experiments

<details>
<summary>Historical Projection experiment workflow and authorization snapshot</summary>

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

## Offline comparisons of actual remote evidence

```sh
.venv/bin/python compare_results.py R02-router-wide R05-wide-layout
.venv/bin/python compare_results.py R03-router-dispatch R04-router-triton16
```

This reads saved terminal records only. It reports actual shape latencies, within-run
SGLang-normalized ratios and six weighted-load speedups with cross-run qualifications.
It does not benchmark, predict a composite score, or establish statistical significance.
Generated examples: COMPARISON_R02_R05.md and COMPARISON_R03_R04.md.

</details>
