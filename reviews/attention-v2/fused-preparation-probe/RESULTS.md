# Preparation fusion: native and official checks passed

Extending the existing fusion to full B16 decode and B16/B32 verify produced a
small, consistent complete-call gain. The one-line dispatch candidate passed
the bounded native diagnostic and one complete nonranking official check.
A2-11 remains selected. This is incremental evidence, not an award-level
architectural result or a proven cross-run score difference.

Official job **`2f5f1fb54b48486d`**, mode **check**, completed successfully:
16 prerequisite tests, 12 benchmark shapes, no correctness rejection or flags.
The six-load speedup reconstructs exactly to **1.225555436×**, with the official
95% interval **[1.207172105, 1.250062193]**. Its score time is available in the
retained receipt and `official-check/analysis.json`. The measurement slot was
released immediately upon terminal observation at 20:26:20UTC. No ranked entry
or additional Modal job was started.

| Changed full path | Official candidate µs | Same-run reference µs | Reference-normalized cross-run latency reduction versus A2-13 |
|---|---:|---:|---:|
| B16 decode | 194.361 | 200.800 | 3.3389% |
| B16 verify | 197.827 | 204.864 | 3.5512% |
| B32 verify | 382.480 | 389.744 | 1.9081% |

The official aggregate point estimate is 1.2027% above historical A2-13 and
3.1891% above historical A2-11. The latter also includes the pre-existing
A2-13 full-B32-decode QK change. These are different jobs with different fresh
inputs and calibration. Reference normalization helps assess controls but does
not create a paired new-versus-parent experiment. The nine unchanged paths'
normalized drift ranged from−2.162% to +1.901%; none regressed by 5%. Overlap of
marginal intervals is not a test of the difference. Do not present the 1.2027%
point delta as a separately established significance result.

Before the official check, one bounded H200 session passed 12 strict preparation
comparisons (36 exact Q/K/V byte comparisons), 36 A→B→A oracle calls, 6 independent
disjoint-cache concurrent-submission oracle calls and 96 checked timing replays.
Paired whole-call reductions were 3.7837%, 4.1058%, 2.3596%, respectively, with all
18 reverse-order pairs positive. These are single-session local measurements,
separate from the official aggregate. Decode fixtures were bimodal by actual
lengths/schedule, so paired ratios are primary. `runtime/RESULT.md` retains the
protocol, per-version medians and limitations.

The frozen source commit is `0280972637e600e5f82a45f35778aed3f72886b1`, based on
the authoritative private baseline `ebb80b91c58032641bf5341927261eb64929f14e` in
the isolated `attention-fused-preparation` worktree/branch. Source SHA256s:

| Source | SHA256 |
|---|---|
| Frozen A2-13 control | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| Tested candidate | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Preserved selected A2-11 | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |

The native worker is `b4b47a4729fb06b85995c115cb91492bc6d552f5a809653359e38a67cd35f08f`;
its raw result is `576d8e09e41b344d6c262c5bbdeb830e5800c6e3a1e19ed6f1aa8cb266e06868`.
The exact committed candidate bytes were checked immediately before the sole
accepted official CLI upload. The service's content ID is separately recorded
and is not asserted equal to raw source SHA256. Independent source, runtime and
official reviews are included. No helper, FA3 argument, matrix precision,
softmax, tolerance, cache format or scheduler/split setting was changed.

Modal app `ap-I6PNHOGgL6uRL1AjjG2yoV` was independently confirmed stopped/tasks0
at 20:18:08.870706UTC, with all listed apps stopped/tasks0. Rounded observed
workspace credit usage increased $0.09 (detailed $0.08614060), actual paid $0.
The conservative closed allowance $0.267981667 is within the reviewed $0.58
free-credit reservation. Billing is delayed and workspace-level, not an exact
per-app invoice. No later charges were attributed here. The official check
used one free quota; **3 Attention checks remain**, resetting at 00:00UTC per
the parent's verified schedule. No payment, limit, credential or installation
change occurred.

Reproduce the retained arithmetic with:

```sh
python3 -B reviews/attention-v2/fused-preparation-probe/runtime/analyze_result.py
python3 -B reviews/attention-v2/fused-preparation-probe/official-check/analyze.py
```

The preparation `manifest.json` describes the immutable local-preparation
snapshot at 0280972; it is preserved as historical provenance. The separate
`result-manifest.json` binds the subsequent sanitized receipts and analyses.
Original local request/diagnostic files stay ignored; published launch and
terminal observations remain separate instead of rewriting a stopping state.

The next independent question is the conditional sliding-B16/B32 split-combine
elimination in `OFFICIAL_QUEUE.md`, after checking the installed wrapper and
obtaining a separate serialized test slot. It has not been implemented or
admitted here. The uniformly positive three-path result gives no reason for
quota-consuming dispatch ablations. No repeat of BF16 custom verify, dynamic
reordering, NCU timeouts or blind knob sweeps is proposed.
