# A07 — explicit replay coverage (H200)

Source parent: `submission.py` (E0); candidate: `fixed_chunks.py` (E2).
Command: `.venv/bin/python reviews/attention-v2/modal_control.py run --mode replay --experiment A07-replay --bundle local-notes/attention-a07-bundle`.
Frozen allowlist, manifest and environment are in `A07-replay-result.json`.

Passed 12 checked sequential replays: two cases × two kernels × A1/B/A2.
Each kernel captured once per case and reused identical tensor addresses. Both
actual sequence-length vectors are saved. A2 output and new K/V are bitwise equal
to A1. Official adaptive output and 8% cache tolerances passed; all untouched
cache rows and other inputs remained bytewise unchanged. Cases: full B16 L8,
80K mixed lengths/coherent/page1 (many empty non-owner chunks); full B32 L8,
80K agentic/normal/page64/scale1/64. This is additional coverage, not a retroactive
claim about old A03/A05. Simultaneous calls/streams remain untested; shared E0
counters remain a concurrency limitation.

Parent/candidate compile metadata: B16 135 registers, no driver-reported spills,
24832 B shared memory; B32 134 registers, no spills, 24832 B shared. E2 splits
40/11 versus E0 9/5 on these two cases; grids depend on static capacity only.
No timings or speed claims from this probe. Worker 26.086 seconds. App
`ap-c7hDHYlZH7Ja3VAJNM4Iur` stopped; exact termination and conservative cost are
recorded in the result and cumulative Modal ledger. Net billed amount remains $0.

Local gates: 2 fixed-chunk CPU/AST tests and 5 harness isolation tests passed;
syntax and git whitespace checks passed. These are separate from GPU evidence.
