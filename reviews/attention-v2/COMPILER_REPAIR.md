# Experiment A01 — minimal compiler repair

Only source change: LOG2E = tl.constexpr(1.4426950408889634).
Original formatting-only source b1a87afb retained in git history and original
manifest; actual official checkbf4b49bd8bb34dae failed all16prerequisites at
non-constexpr LOG2E, no numerical results/timings. No prior fix was executed.
Applied exact proposed one-line patch, no math/partition/ownership/precision change.

Command: .venv/bin/python -m py_compile reviews/attention-v2/submission.py
Syntax pass only; actual Triton/H200 correctness still unverified at this commit.
All projection work preserved on projection-bf16-local (0eabf0e), best1.14865x.
No OKCcheck, push, credentials or GPU operation in this source-only experiment.
New bounded attention Modal testing authorized separately, cumulative existing
stage1.146411930/2USD retained, net paidcap0. See subsequent harness/result report.
