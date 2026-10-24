# One additional approved attention v2 check

The local preflight is complete. No proven official single-call blocker was found;
concurrency robustness and intermediate-precision qualifications remain in REVIEW.md.
The user then explicitly approved one additional $0 off-board H200 check, source only,
with no algorithm changes beyond five formatting repairs. No ranked entry or GitHub
publication is authorized. The projection10-check budget remains exhausted/unchanged.

Candidate `reviews/attention-v2/submission.py`, source commit
`7eea86ba1f63cc18a229d276eb1e917b5b20dc74`, SHA256
`b1a87afb4c428502d5066688ba784f577236b6bad4fea8879c8cd98fefb9c3dc`.

```sh
python3 reviews/attention-v2/remote_once.py send
python3 reviews/attention-v2/remote_once.py poll
bin/okc status bf4b49bd8bb34dae --wait --json
```

The sole send was accepted at2026-10-24T17:37:11Z as job **bf4b49bd8bb34dae**, mode
**check**, problem muse_attention_decode. Before admission attention quota0/10checks,
0/10scored; rules version2026-10-22.1 unchanged, existing supported CLI auth verified.
No token extraction, new credentials, agreement, paid resource or scored entry.

**Additional allowance1/1 used, no retry/new check allowed.** Any uncertain result
must be reconciled with own jobs instead of resubmitting. Follow this accepted ID to
terminal; queue state is not a completed experiment. The one-admission guard saves
its reservation before running official CLI and refuses a second send.

Current state and exact read timestamps are in remote-policy.json. Pending result:
no correctness or performance is inferred from the preceding CPU model tests.
All commits remain local on attention-v2-preflight; do not push this source.

The original preflight manifest/README/review preserve their creation-time pending
status. This later approval/admission record supersedes that historical status only;
it does not alter their numerical or concurrency findings. The syntax-only source
hash remains identical. `okc whoami --json` returns plain login text in this CLI;
that harmless read succeeded and was handled without exposing private account data.

## Terminal result — confirmed in-contract compilation blocker

Job **bf4b49bd8bb34dae** is terminal **done**, but `passed=false`, returncode112.
All **16/16 prerequisite records fail compilation** at the same source line124:
plain module float `LOG2E` is read inside the JIT function. The target compiler
requires `tl.constexpr(...)` instantiation, not merely a type annotation. Integrity
checks themselves pass with no flags/rejections. No attention kernel numerical
execution or benchmark timing succeeded; **speedup and score are null**, not zero.

This is now a definite blocker in the real official H200 environment, superseding
the preflight's earlier statement that no confirmed single-call blocker had been
reproduced. The earlier CPU equation tests did not compile Triton and missed this
restriction; their numerical-model results remain separate evidence.

Minimal proposed fix at source63:

```python
LOG2E = tl.constexpr(1.4426950408889634)
```

`compile-fix.proposed.diff` records exactly that one-line proposal. It is **not
applied** to submission.py, uploaded or GPU-validated. Its Python syntax passes;
that cannot establish subsequent target compilation. Inlining the same literal in
line124 is another source-level fix; do not set a harness environment override.
Other previously reviewed concurrency/precision issues remain untested on GPU,
because compilation stopped first. No claim is made that this is the only remaining
compiler issue, nor that the candidate would pass after this single fix.

Sanitized full16-case errors and evaluator/image/device provenance: remote-result.json.
Raw reply/log remains ignored locally. The single new check is fully consumed:
**1/1 used,0remaining,0uncertain,$0,0scored**. Attention service quota1/10checks used,
9left; those leftover service slots grant no new user authorization. The projection
10-check budget remains unchanged. No second check, credentials or publication.
