# E0 — official corrected v2 baseline

Job `5b0baa286659421a`, off-board check, passed 16 prerequisite tests and all
12 scored shapes. Source `submission.py`, commit
`05922b7bf0353a3f4a54aaa27f9eef0692a4ee32`, SHA256
`3a86e83511725c581f288c4b5b07a8484a50e88cd396466d00a97f330cd2d353`.
Only the previously documented LOG2E constexpr compiler repair was applied.
Command: `.venv/bin/python reviews/attention-v2/check_cycle.py send E0`;
monitor: `.venv/bin/python reviews/attention-v2/check_cycle.py poll E0`.

Official H200 score: **0.7224690513× SGLang**, 95% interval
[0.7116320155, 0.7373518839]. It is slower overall. Every timing row has its own
same-run SGLang comparison. Most shapes have 60 repetitions, final full B32
verify has 56; all raw samples and overhead calibrations remain in the receipt.
The official six weighted loads reproduce the reported score exactly. Raw
CUDA graph times are distinct from official paired/overhead-adjusted times.

See `E0-official-result.json` for immutable evaluator/image hashes, machine data,
raw timings and test errors; `official-cycle-summary.json` is the compact offline
summary. Run `.venv/bin/python reviews/attention-v2/summarize_official.py` to
regenerate it. Public Modal image timings are separate diagnostic evidence.

E0 is useful as our corrected baseline, not a leaderboard entry. No scored run,
publication or repository push occurred. The global ticket map remains shared
between simultaneous invocations on the same device; only sequential graph
replay has been validated. A07 adds explicit same-address A/B/A with changing
lengths, including page64/non-unit-scale L8, and is recorded separately.
