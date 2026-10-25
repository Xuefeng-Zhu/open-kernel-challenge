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
raw ignored/private logs and unrelated files. This local policy update performs
no push, admission, check or GPU action.

## Compatibility with the existing runner

`experiments/projection-cycle2-budget.json` retains numeric
`maximum_new_experiments=10` and `remaining_authorized=5` for historical accounting
and compatibility. They are **not current user authorization ceilings**. The new
`current_authorization` annotation and this document record the superseding scope.
Earlier authorization strings, pauses and dated count statements are historical;
accepted request records and measured results remain authoritative evidence.

`cycle2_check.py` now bypasses only the legacy personal count cap when
`current_authorization.personal_check_count_ceiling_removed` is explicitly the
JSON boolean `true`. Missing, false or malformed policy keeps the old capped
behavior; truthy strings/numbers do not remove the cap. With explicit removal,
ledger saves preserve the historical numeric `maximum_new_experiments` and
`remaining_authorized` values rather than recomputing a negative allowance.
Old capped ledgers retain their existing count admission and remaining arithmetic.

Daily platform quota, rules-version agreement, committed source/CPU hash gates,
shared GPU ownership, active/uncertain reservations and no-automatic-retry behavior
remain unchanged. No arbitrary large replacement cap is introduced. Current
request records remain the record of actual admissions; historical remaining
counts are not a live personal allowance. The real ledger is not advanced by tests.

Focused local fake-CLI tests cover explicit removal above ten requests, legacy
caps and missing/malformed authorization, zero daily quota, active/uncertain jobs,
shared attention ownership, source/CPU hashes, rules version, pause/duplicate gates
and uncertain outcomes without retries. All subprocess/git/rules/CLI responses are
fakes inside disposable worktree fixtures: these tests make no outbound requests.
Run with the existing Python:

```sh
<primary-workspace>/.venv/bin/python -m unittest discover -s tests -p test_projection_admission_policy.py -v
```

No check is launched today by this update, and no new per-source admission is
recorded. The coordinator reviews, commits and pushes these local changes.
