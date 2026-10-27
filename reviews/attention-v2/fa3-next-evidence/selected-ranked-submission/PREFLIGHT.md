# Selected Attention ranked-submission preflight

Read-only preflight only; no candidate was uploaded and no submit/check/login/token
command was run by this reviewer. The root agent exclusively owns the single
authorized ranked request.

Selected source SHA256: `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
It exactly matches successful off-board check `5090b83f1d63407b`, PASS with no
flags and speedup 1.2336330588627955. The source begins with the correct
`# okc: problem=muse_attention_decode` header.

At the saved read time, ranked quota was 10 left / 0 used, checks 19 left / 1 used.
The CLI returned 47 own recent jobs: all terminal; the 25 Attention jobs are all
mode=check. Public leaderboard `you` is null. Thus no current ranked row or
ranked request appears in the returned evidence. `jobs` is documented as recent,
so this is not an exhaustive identity search over withdrawn historical jobs.

Current official registration reports the kernel track open. TASK.md, AGENTS.md,
baseline.py, reference.py, checks.py and template.py exactly match saved local
files. Read-only calls succeeded without any terms-acceptance or login challenge.
This does not authorize accepting new explicit terms if the actual submit presents
one; stop for any such new material condition.

The supported ranked syntax is `okc submit [problem] <file> [--no-wait] [--json]`.
Use the explicit `muse_attention_decode` problem and unchanged selected path,
exactly once. Save the returned receipt immediately; reconcile any uncertain
outcome before another action. Do not mutate source to bypass deduplication.

`PREFLIGHT_SUMMARY.json` contains the exact command and source/evidence links.
The raw public leaderboard is saved but no competitor private job was accessed.
