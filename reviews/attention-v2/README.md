# Attention v2 experiments

Current cycle: E0–E4, up to six newly authorized **off-board attention checks**.
Projection's remaining eight checks are separate. No ranked entry or repository
push is authorized. See `attention-cycle-budget.json` for exact accepted jobs,
source hashes and current states; never resubmit a pending or uncertain request.

- E0 compile-fixed v2: official correctness passed, 0.7224690513× SGLang;
  [official result](E0_OFFICIAL.md), [receipt](E0-official-result.json).
- E1 historical separate merge: Modal correctness passed, overall regression;
  [report](REPORT.md). Not used as E2's parent; no new check spent on it.
- E2 fixed8192 history chunks for full B16/32 only: [scope](E2_FIXED_CHUNKS.md).
  Official job 7edded086c8a47f2 passed: 0.7958818006× SGLang, about 10.2% over E0.
- [Coverage audit](COVERAGE_AUDIT.md) distinguishes the older tests from the new
  [explicit same-address A→B→A test](A07_REPLAY.md). New length vectors, page64,
  non-unit scales, cache writes and untouched rows were checked on H200.

Only sequential graph replay is validated. The original per-device shared ticket
map still has a simultaneous-invocation limitation; no concurrency pass is claimed.
CPU/AST checks are separate from actual GPU compilation and correctness.

```sh
.venv/bin/python reviews/attention-v2/check_cycle.py poll E0
.venv/bin/python reviews/attention-v2/check_cycle.py poll E2
.venv/bin/python reviews/attention-v2/summarize_official.py
.venv/bin/python reviews/attention-v2/test_fixed_chunks.py
.venv/bin/python reviews/attention-v2/test_pipeline.py
.venv/bin/python reviews/attention-v2/test_modal_worker.py
```

Official summaries use six loads, each 46×sliding +16×full, then their geometric
mean. Complete receipts retain all twelve raw/adjusted timing rows and samples.
Modal public-image measurements are diagnostic, not official scores.

The cumulative Modal ceiling is $2 free credits, net paid $0, across both projects.
The authoritative ledger is `modal-budget.json`; do not reset it from older branches
or lower it when billing lags. All new bundles use exact committed allowlists and
existing authorized authentication; secrets/config and repository data are excluded.

Historical initial review, original paste, formatting normalization and failed
check bf4b49bd8bb34dae are preserved in REVIEW.md, REMOTE_CHECK.md, original files,
and Git history. Their old pending/one-check restrictions describe that earlier
stage. The subsequent minimal compiler repair is commit 05922b7, unchanged in E0.
Root projection source/results and its best branch remain preserved.
