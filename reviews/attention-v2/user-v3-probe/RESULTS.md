# Two requested original-source checks: compilation failure

Archive SHA256 verified:
`2702d954222431d7808ffe43d35c0cffe759ad3840c0af7fbc025088de1645eb`.
Both sources were submitted byte-for-byte unchanged. They differ only in
`PREFETCH=1` versus0. Exactly two distinct off-board checks were requested and run.

| Source | Job | Outcome | Score / D16 / D32 / V16 / V32 |
|---|---|---|---|
| submission.py |02242a404e144f2a|Triton compile failure|Not measured|
| submission_v3b.py |d704c8d1be974b62|Triton compile failure|Not measured|

Each job is terminal `done`, mode `check`, returncode112, passedFalse, with
16 failed test records from compilation and **zero benchmarks/profiles**.
This is not a numerical-error or performance result. Sliding changes are also
unmeasured. The selected champion remains1.233633059x; no tuning follows.

Both report the same source location and error (verbatim excerpt):

```text
triton.compiler.errors.CompilationError: at 36:31:
    qk_scale = sm_scale * ks * LOG2E
                               ^
NameError("Cannot access global variable LOG2E from within @jit'ed function.
```

Exact returned official log tails, without editing, are `v3-error.txt` and
`v3b-error.txt`. The published result JSON only makes local output/profile
paths relative; raw local receipt hashes and both source hashes are in
`summary.json`. An ordinary constexpr instance such as
`LOG2E = tl.constexpr(1.4426950408889634)` addresses this particular error;
**no repair was applied**, per the user's request to return errors for them to fix.

Source review found no candidate network/process/harness mutation or precision
violation. Persistent shared ticket buffers remain an untested overlapping-graph
hazard. The bundled local tester was not executed: it changes the cache tolerance
from the official0.08 to0.25. No reference or tolerance was changed here.
No Modal job, ranked submission, credential action or paid spending occurred.
