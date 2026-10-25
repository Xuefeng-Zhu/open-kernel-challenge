# Current authorization

The latest user instruction removes the self-imposed count ceiling for **free,
off-board attention and projection OKC checks**. Conversation reference supplied
by the coordinator: `Sentinel_26cefe33699481919e869a7bce3c3476`.
This supersedes earlier personal ten-check cycles and remaining-check allowances;
it does not erase their accepted jobs, results, failures or historical accounting.

Actual service quotas and rate limits still apply. Obtain the current platform
quota before any future admission; a UTC reset does not create a new personal
budget because there is no current personal count ceiling. The projection ledger's
last quota snapshot is dated evidence, not a live balance. Serving has separate
platform caps and is currently closed according to the public API read in this
session; lifting kernel-check ceilings does not authorize serving submissions.

The authorized scope remains evidence-led, meaningful experiments, with immutable
reviewed sources, complete failure records and no nonce repeats or arbitrary
parameter sweeps. One primary coordinator owns the shared GPU slot across both
problems. Resolve active/uncertain requests before admitting another measurement;
never repeat an uncertain POST automatically. A remaining platform quota is not
itself a reason to run an experiment. The projection engineering pause after
P040 remains appropriate until a concrete new mechanism or integration task exists.

No extra paid compute, Modal runs, new/renewed credentials, account/payment/cap or
binding-terms changes, ranked submissions, or public visibility changes are
included. Reviewed **private auto-pushes to the existing private repository** are
allowed by the standing user instruction; the coordinator handles them. Review
sources and sanitized evidence first, retain attribution, and exclude credentials,
raw ignored/private logs and unrelated files. This documentation task performs
no push, admission, check or GPU action.

## Compatibility with the existing runner

`experiments/projection-cycle2-budget.json` retains numeric
`maximum_new_experiments=10` and `remaining_authorized=5` for historical accounting
and compatibility. They are **not current user authorization ceilings**. The new
`current_authorization` annotation and this document record the superseding scope.
Earlier authorization strings, pauses and dated count statements are historical;
accepted request records and measured results remain authoritative evidence.

`cycle2_check.py` still enforces the legacy `maximum_new_experiments` hard cap and
recomputes `remaining_authorized` from it. Its enforcement must be reviewed and
updated to understand the removed personal ceiling **before it reaches the old
hard cap**. Do not replace the cap with an arbitrary large number, set a value
that breaks existing arithmetic, or bypass daily quota/shared-slot checks.
The runner is deliberately unchanged in this documentation-only update. No check
is launched today by this update, and no new per-source admission is recorded.
