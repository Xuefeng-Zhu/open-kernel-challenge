# Final narrative review: PASS; candidate rejected for selection

Reviewer: `/root/compact_review`. Saved artifacts only were read. Only this
report was written; no source, service, GPU, authentication or Git action occurred.

| Bound artifact | SHA256 |
|---|---|
| `RESULTS.md` | `485e1219efbcf9c533697fab5182e87115c03a9efcc16066c41129dbb8fda914` |
| `terminal-receipt.json` | `990e0286342ee2d11d2bcb8c526d861d10b179fbc778fc69d46d3bf4e6a2645e` |
| `ACTUAL_RESULT_REVIEW.md` | `c83eeb3332bb7bcd67f846c259da909ae7dc55fe883c31eb884e7205bda702c0` |
| `analysis.json` | `e6f57d79a35f627f0e081fe45c7e471b40f409b166960ebe9a5e4f295fffcf71` |
| `official-result.json` | `e61d99b67ebbf917cfc4e5a5ad29848e37b92be60a08e9825728e0bd428b81c0` |

The narrative accurately reports job `3e0a7ef51d4849c6` as a terminal,
nonranking original-oracle PASS:16 prerequisites and12 benchmark calls, with
returncode0 and clean checks. That is correctly separated from performance
rejection: score1.0995474217354162 versus selected1.2336330588627955.
The four normalized target regressions are21.0917%,20.7343%,29.5344% and
79.5972%; maximum absolute drift across eight unchanged controls is1.9018%.
Both declared target performance gates fail. The official point-score change
−10.8692% is correctly distinguished from the target-only fixed-reference
model's −10.7816%, rather than presented as the same measurement.

The saved independent arithmetic review confirms all twelve raw/profile
benchmark records agree for both runs. The report correctly describes the
historical selected comparison as reference-normalized, not a paired
selected-versus-new experiment. Evaluator/image/file hashes and reported
stable GPU fields match, while nodes, fresh inputs/seeds and transient
conditions differ. It does not infer statistical certainty from point scores
or marginal intervals.

Instrumented producer/merge durations are explicitly separate from calibrated
whole-call timing. They associate the regression with producer activity but
do not establish a causal instruction, achieved residency or hardware-counter
bottleneck. S9/S5 now have official runtime/oracle evidence; exact executed
ISA/resources/binary identity and targeted replay/concurrency coverage remain
unknown. The result rejects this integration without rejecting every WGMMA
or transposed architecture.

The earlier static STOP remains intact. The later explicit direct-runner
decision and reassessment of common baseline RoPE stack/local behavior are
transparent; no source/precision/tolerance/configuration rescue is claimed.
The frozen96e candidate and selected3da source remain distinct, with the
selected candidate and A2-11 rollback preserved.

Saved closure is consistent: one check used,19 remaining, ranked usage0;
`active-jobs-final.json` at02:37:35UTC records this job done and an empty active
list. The terminal receipt records the finished wait client, no Modal use,
credential action or formal entry, and paid spending$0. These are saved
observations, not fresh live checks by this reviewer. No further unchanged
candidate check or post-result sweep is scheduled. No reporting blocker found.
