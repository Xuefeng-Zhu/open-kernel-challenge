# P055 — compiler eligibility correction and a quantitative leader-gap audit

Local audit on 2026-10-26; no compilation, GPU run or official check. P046 stays
selected at 1.1841399408925375×. This report supersedes the universal AOT-only
assumption in P047/P054, while preserving their historical observations.

## Rules finding and provenance

The [official global rules](https://openkernelchallenge.dev/rules) explicitly say:

> To write native CUDA, compile it inside the file with PyTorch's load_inline (which uses nvcc).

The same page permits building on the image's supplied frameworks, listing
CuTe DSL, CUTLASS and TileLang among others. Compilation consumes the run timeout,
not the performance score. It does not permit arbitrary subprocess execution,
network installation, evaluator inspection or moving input-dependent work out
of timed calls. BF16 precision, fresh inputs and graph compatibility still apply.

Read through the user's Mac Chrome at approximately 00:04 UTC. The public
[rules metadata endpoint](https://openkernelchallenge.dev/api/v1/rules) returned
version `2026-10-22.1`; this read did not accept any agreement. A normal HTTP
reader returned 403 and the web reader returned no page text; the ordinary
browser displayed the content. No access restriction was bypassed.

The current [Projection task API](https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj)
was also read in the browser. Its generic process/network prohibition is followed
by a link to the full rules; it does not separately ban native CUDA. The explicit
compiler exception qualifies that generic prohibition. No current byte-level
snapshot equality is asserted: the live JSON was inspected, not archived anew.

Our unchanged snapshot is independently hash-verified against
`official/manifest.json`, retrieved 2026-10-24T06:41:11.761566+00:00:

- TASK SHA256 `deb9118fa75cbc03c2ebd16b283e72a0314168d40299fa7501373eb1960a2bc4`.
- AGENTS SHA256 `d6f3fb0cedb7cfac9aaef929c5ec2999eba5e1103cc3fa240f49de4e732e21fc`.
- Task JSON SHA256 `ee02afa1e848c15b16d677fc1e405617a51182fbda7ed6b21c9e5f6a7c5ef0e9`.

Local TASK lines 101/103 contain the generic restriction/full-rules link;
line 116 allows ordinary diagnostics, and line 120 includes compilation in the
ten-minute limit. Neither official TASK nor AGENTS contains our import allowlist.

`static_check.py` was authored in our baseline `fde8a04`, most recently changed
before this audit in `60a8b4b`; its own first line says it is not the official
validator. Pre-audit SHA256 was
`4789eee9a2b1088df493199801b65adbcd92a7c6fb79a39cd660d862cf572aad`.
Its import whitelist and all-phase `print` ban were stricter than the rules.
The narrow correction admits the exact `load_inline` import and a direct build
at initialization/prepare, plus ordinary diagnostics outside `kernel_fn`.
It retains the remaining guards; no general subprocess/import permission was
added. AST checks cannot prove transitive graph safety, native precision or
legality. A passing local preflight is not official acceptance.

Compiler/header/library availability in the actual runner remains unverified.
An existing precompiled binding is convenient, but no longer a universal legal
prerequisite. Unknown FlashInfer loader behavior remains a reason to inspect
that implementation, not evidence that all native compilation is forbidden.

## Measured gap and conditional requirements

Public CLI observation at **2026-10-26T00:06:43.166237+00:00**: 1,010 ranked
participants; `colinlikescode` leads at **1.6238×**. Our `you` field is null because
the preserved work is off-board. [Public board](https://openkernelchallenge.dev/leaderboard?problem=muse_bf16_proj).
No competitor's private job or code was accessed. Rounded public and our official
scores come from different runs and are provisional, not a paired comparison.

The immutable P046 CSV reconstructs the six-load official score exactly. Each
load is the sum of five family latencies with weight 62; the six reference/test
ratios are geometrically combined. This is not Attention's 46/16 formula.

| Conditional model | Required latency cut to reach 1.6238× |
|---|---:|
| All paths improve uniformly | 27.0760% |
| Pooler + QKV + unpooler, all M; other paths fixed | 35.7169% |
| Those three families, only M>1; other paths fixed | 41.4128% |

Mean share across six load totals: pooler 31.0467%, QKV 23.4581%, unpooler
21.2960%, O 11.1032%, router 13.0960%. Uniform cuts of 10%, 20%, 30% across the
three main families model approximately +8.20%, +17.87%, +29.44% in whole score.
These are requirements/sensitivities, not attainable predictions or hardware
limits. TASK's idealized floors are not strict physical bounds.

Reproduce without CUDA or candidate imports:

```bash
python -B projection_reverse_inference.py --output results/router-p055-score-model.json
python -B -m unittest tests.test_projection_gap_model tests.test_native_compilation_policy
python -B static_check.py --submission candidates/projection_pooler_column_batch2_hybrid.py
```

All ten model/policy tests and the unchanged P046 static preflight passed on the
local Python 3.12.13 runtime. Adding the five existing CPU contract tests yielded
15/15 passes; Torch emitted an unavailable-NumPy warning, with no test failure
and no package installation. The first all-path calculation hit the solver's
zero-latency boundary; it was corrected to the analytic `1 - selected/leader`
formula and all three requirements are checked by reconstructing their score.
This was a local analysis failure, not a GPU failure. The output records both input
hashes and all scenario assumptions.

## What might explain a large gap — and what the evidence does not say

The leaderboard alone does not reveal the leader's implementation. Public code
shows native kernels and mixed library dispatch are used by some participants;
it does not connect those sources to this leader or prove a large speedup.

- [Pinned lilaclilac09 source](https://github.com/lilaclilac09/open-kernel-challenge/blob/d9bdcc9f01dff1018f2f256a0a1a63911a16557e/muse_bf16_proj/submission.py)
  uses native single-token GEMV and library fallbacks. Its separate tensor-core
  experiment records a previous correctness failure; no strong verified gain
  was established by this audit.
- [Pinned ozgureyilmaz source/results](https://github.com/ozgureyilmaz/okc-muse-kernels/blob/3c6e97000c67d33982cfdf6d6038c2719a100be9/state/results.jsonl)
  self-report 1.0139× scored, with uncertainty crossing parity. This is not an
  explanation of the current leader's margin.
- Static weight packing is allowed; prepare has no X. Score aggregation does
  not create a shared activation lifetime between the five independent calls.
  Approximate low-rank/sparsity or FP8 BF16-weight substitution cannot be
  justified by numerical tolerance alone. Cold-L2 graph timing also rules out
  assuming a warm-cache-only gain will survive scoring.

Ranked next mechanisms:

1. **Explicit vendor execution-plan control with exact BF16 representation.**
   P046 proved static packing/library routing can yield ~10% paired target cuts;
   P049 proved one wrapper change helps QKV but hurts unpooler. Neither proves
   the vendor library's ceiling. A small owned cuBLASLt bridge can expose legal
   algorithms and precision/workspace metadata unavailable in ordinary matmul.
   Test one QKV M128 and unpooler M16 first, against both a native default-plan
   control and actual P046. This is the highest-information next probe, not a
   predicted 35% gain. Two-family M>1 cuts of 10%/20% model only +4.07%/+8.54%;
   substantial broader coverage would still be needed.
2. **A distinct split-preserving producer/consumer mainloop and static layout.**
   Preserve precision and useful parallelism, demonstrate the scheduling and
   layout costs in generated instructions, then beat actual P046 under cold
   graphs. P053's partial-only M8 is already 14.60% slower than P046's whole
   call, so cheaper reduction alone cannot rescue that fixed implementation.
   A changed pipeline, not another orientation/tile tweak, needs its own causal
   control. A 20% cut over the three M>1 families models +14.76% score.
3. **Fused lossless decoding, only after a competitive raw backend exists.**
   P043 reduced stored bytes ~23.3% but regressed 158–384%; its matched raw
   backend was never GPU-tested. Exact reconstruction, full storage accounting
   and a matched competitive raw path are necessary before retrying this idea.
   Byte reduction is not measured memory bandwidth or equal latency reduction.

The first probe will use our own interface code; no competitor source has been
copied or executed. NVIDIA's [native algorithm example](https://github.com/NVIDIA/CUDALibrarySamples/blob/9b347f161cce5cd2a51a0406b140f2a2d81df024/cuBLASLt/Common/LtMatmulCustomFind.h)
demonstrates explicit algorithm initialization/check/selection, not performance
or contest-environment compatibility. Any future source reuse requires license
and attribution review.

## Separate external push outcome

P053's first push failed at automatic approval review before process creation;
it was not a static-checker rejection or native-compiler restriction. Current
local official AGENTS had no GitHub push grant. Earlier forwarded human push
authorization existed in conversation; the reviewer's internal view is unknown.

After newly supplied explicit user-authored authorization, the same helper was
retried once, verified PRIVATE visibility and fast-forward ancestry, and pushed
only the reviewed committed Projection history. At **00:11:27.470752 UTC** the
remote `projection-pipeline-preflight` SHA was verified as
`fd1c495e3d19a7e965daecd3b5a96045c18fc731`. It included 49 reviewed historical
blobs / 1,067,631 bytes and no detected private paths or sensitive patterns.
This report and the new interface preparation were not included in that push.

实验卡：我们此前误把官方允许的原生编译也排除了，搜索空间因此过窄。
下一步先用最小 cuBLASLt 接口验证可用算法、精度与 graph 行为，再对照 P046；
不把“能编译”当成“会更快”，也不推断领先者一定用了这个方案。
