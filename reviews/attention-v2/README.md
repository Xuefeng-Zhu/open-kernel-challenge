# Local-only attention v2 preflight

Read REVIEW.md first. The user's algorithm is unchanged; submission.py only restores
five corrupted Markdown links. Original paste, escaped transport, exact diffs and
SHA256 manifest are preserved. Root dense-projection submission/results are unchanged.

Python syntax passes. Eight CPU/analytical tests pass; eight equation-model cases
pass the official numerical checker. The actual Triton source is **unexecuted**:
interpreter import fails because Triton is absent, and the Mac has no CUDA.

The concrete robustness defect is device-global tickets shared by simultaneous
calls/graphs; the state-model counterexample includes a published-style16split case.
Sequential replay and concurrent replay are classified separately in REVIEW.md.
No numerical failure in published single-call shapes has been reproduced.

```sh
.venv/bin/python -m py_compile reviews/attention-v2/submission.py
.venv/bin/python reviews/attention-v2/preflight.py
```

**Do not push, publish or upload this draft.** Its commit is local-only on
`attention-v2-preflight`, without an upstream. The prior ten projection checks are
exhausted. Additional attention remote-check approval is pending; this folder grants
no execution/upload permission. No new credentials, installs or paid resources.
