# Independent official-check review

The complete official check passes. Job `2f5f1fb54b48486d` is terminal `done`,
mode `check`, board `muse_attention_decode`, with no duplicate, error, flags or
rejections. All sixteen prerequisite tests and twelve scored benchmark shapes
pass. Twelve additional profile events are diagnostic records, not extra
correctness tests.

The reported aggregate speedup is **1.2255554363764571×**, with reported 95%
interval **[1.2071721048308102, 1.250062192674554]**. The normalized score time
is 3073.1371982124647µs, equal to 3766.3µs divided by that speedup. Independent
reconstruction of all six weighted load ratios reproduces the speedup exactly:
each load uses `46 × sliding + 16 × full`, followed by a geometric mean over
six reference/candidate load ratios.

## Source and receipt binding

The accepted request records the frozen native-tested source commit
`0280972637e600e5f82a45f35778aed3f72886b1`, the exact standalone candidate
SHA256 `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`,
and the absolute candidate path passed to `okc check --no-wait --json`.
Those local source bytes remain identical to the native-passing candidate.
The returned official job ID agrees with the accepted request.

The reviewed raw terminal status receipt SHA256 is
`a71bde08d0164c2d7f2c5163460466757318cd9a96bfbd2ac34902bd1cd2743e`.
The service's content identifier is
`90ae245f9a419b645830505d98afb32f809a4fe6c33b07ac1d3eab62ef37d52e`.
It is a distinct identifier, as in historical receipts; it is not presented as
an independently echoed raw source SHA256. Any sanitized exported receipt may
have a different file hash and must preserve its provenance mapping.

## All twelve scored shapes

Times below are official adjusted values from each benchmark record. Candidate
`median_ns` is normalized from the same-run reference and paired ratio; it is
not simply the median of raw CUDA-event samples. The paired comparison within
this check is candidate versus SGLang, not candidate versus A2-13.

| Load / layer | Same-run reference µs | Candidate µs | Paired speedup |
|---|---:|---:|---:|
| Decode 1 / sliding | 21.216 | 11.621 | 1.825634× |
| Decode 1 / full | 31.504 | 27.638 | 1.139893× |
| Decode 16 / sliding | 26.816 | 14.463 | 1.854137× |
| Decode 16 / full | 200.800 | 194.361 | 1.033131× |
| Decode 32 / sliding | 30.976 | 19.915 | 1.555406× |
| Decode 32 / full | 384.896 | 317.205 | 1.213398× |
| Verify 1 / sliding | 22.928 | 17.017 | 1.347355× |
| Verify 1 / full | 36.416 | 30.126 | 1.208797× |
| Verify 16 / sliding | 29.296 | 22.893 | 1.279709× |
| Verify 16 / full | 204.864 | 197.827 | 1.035571× |
| Verify 32 / sliding | 34.048 | 27.460 | 1.239927× |
| Verify 32 / full | 389.744 | 382.480 | 1.018993× |

## Historical comparisons and uncertainty

The current point score is **1.202704% above historical A2-13's 1.210990803×**
and **3.189135% above historical A2-11's 1.187678756×**. These are cross-run
point comparisons. The A2-11 comparison also includes A2-13's pre-existing
full-decode B32 QK arithmetic change and cannot be attributed to this fusion.

For the three dispatch targets, normalize both runs to the same reference
latency by comparing their recorded paired speedups. The resulting conditional
latency cut is `1 − old_paired_ratio / current_paired_ratio`:

| Changed full-attention path | Historical A2-13 candidate µs | Current candidate µs | Control-normalized cross-run cut |
|---|---:|---:|---:|
| Decode B16 | 200.209 | 194.361 | 3.338865% |
| Verify B16 | 204.454 | 197.827 | 3.551224% |
| Verify B32 | 389.167 | 382.480 | 1.908098% |

All three signs agree with the separate native control-versus-candidate paired
session, whose reductions were 3.7837%, 4.1058% and 2.3596%. The remaining nine
unchanged paths have control-normalized cross-run changes from approximately
−2.162% to +1.901%, with no unexplained >5% regression. This is useful supporting
consistency, not a matched-input causal experiment.

Current and historical receipts have identical evaluator code/file hashes and
judge image. They still use different seeds, nodes and sessions; A2-11 also
has a different driver version. Reference normalization cannot remove every
runtime or input difference. The reported intervals are for each candidate's
speedup versus its own same-run SGLang control. Their overlap is not a test of
the difference between candidates, and no confidence interval for the 1.203%
cross-run improvement was measured here.

The evidence supports a small preparation improvement and complete official
correctness coverage. It does not establish an award-level advance, a formal
leaderboard result, or automatic promotion over the selected source. A2-11
remains preserved. This check consumed one check quota and created no formal
entry. Parent reported the measurement slot released immediately on terminal
status; no additional official or GPU job was started by this review.

This review used local receipts and source files only. No CLI, network,
browser, credential or job-control action was taken.
