# A2-15 passes focused execution checks; target latency falls 5.19%

2026-10-25. One bounded H200 run completed. App
`ap-DjbFQfnZXQjpr1Zgt3rtFK` was created at 16:17:04 UTC and stopped at
16:17:37 UTC. A separate 16:17:56 UTC read verified stopped, zero tasks and
zero active apps. No official OKC check, credential change or paid spend.

**Decision:** retain A2-15 as a modest promising experiment, but do not promote
it over selected A2-11. The target improved in every measured comparison, yet
missed the predeclared 10% latency-reduction goal. This run is neither a complete
12-shape validation nor an official score.

## Exact source and execution

Harness source commit: `5cb8db0` (full SHA in `paired-modal-run.json`).
The frozen candidate files are unchanged from NATIVE_RESULT.md:

- Parent A2-13 SHA256:
  `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
- A2-15 SHA256:
  `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c`.
- Executed worker SHA256:
  `8f7cfd52daf2af5f7b4ea23cb07b88b88668dc4e910ef63c47e07f4d5b7287fc`.
- Retrieved result SHA256:
  `92b5ba197689c3e17ec3c86cefdefdb1edd4bfe0222171b0bd63bd67f000e867`.

The allowlist contained exactly the two candidates, worker, existing harness
helpers, official reference/checks and two license/notice files. No repository
automount, credentials, private logs or unrelated files were uploaded. Existing
image `im-cPTbGpNiX3RarKResPgX7h` was reused.

Observed worker environment: NVIDIA H200, capability 9.0, Torch 2.14.1+cu130,
CUDA 13.0, Triton 3.8.0, Python 3.12.3. Driver version was not collected in this
paired run; do not substitute the separate native run's driver as a measurement.
The worker ran for 22.562 s, including 15.821 s imports, 3.877 s adversarial
preparation/checks, 2.740 s target preparation/checks and 0.122 s timing phase.
Remote wrapper wall was 23.756 s; client wall 32.630 s; app lifetime 33 s.
These wall times are separate from the kernel latency measurements below.

## Actual correctness coverage

| Case | Purpose | Official output errors for A/B | Allowed bounds A/B |
|---|---|---|---|
| Full B32/L1, TP8, 80K agentic/coherent, page1, scale1 | Changed scored shape family | 4.495% / 3.821% | 12.4% / 11.2% |
| Full B32/L1, TP8, page64, scale1/64, scattered boundary lengths | Page/tile/chunk tails and empty history | 5.634% / 5.116% | 15.0% / 14.6% |

Both implementations reported the same rounded errors. This is not a direct
cross-implementation byte-equality measurement. The official adaptive tolerance
is 2.5 times its FP8-query reference error plus 1.5%, clamped to 3–15%; its
worst-row and nonfinite-output checks also passed. New K/V entries passed the
official 8% bound; successful messages do not expose their separate error values.

Four graphs were captured: two cases times two sources. Each replayed A→B→A
at fixed tensor/output addresses, with different A/B contents and actual lengths.
Eight A/B replays invoked the official oracle; four returning-A replays exactly
matched their previously validated output/new-KV bytes. All twelve checked
immutable inputs and every untouched cache row exactly.

The page64 case explicitly used lengths 0/1/63/64/65/8,191/8,192/8,193, reversed
between versions, with consistent positions/cache locations/FA3 metadata. A
zero-length request wrote live cache row zero. Invalid page-table suffixes were
masked to -1. Zero historical length is extra robustness coverage beyond the
official generator. The short case had two splits; the scored-family case eleven.

Timing then performed eight warmup calls and 24 measured calls. Every one of
these 32 calls reproduced the same-source, same-input oracle-checked output/new-KV
bytes and preserved immutable/untouched rows. Total validated replays: **44**.
The three eager warmups per capture were not individually oracle-checked.

Not covered: other ten scored shapes, all sixteen official prerequisites,
unchanged-path controls, overlapping streams/calls, all nearby shapes, or an
independent session. No broad correctness or concurrency claim follows.

## Paired target timing

Inputs and caches were restored outside each timed region. A 256 MiB read
preceded each CUDA-event measurement of whole `kernel_fn` graph replay. Four
warmup pairs were predefined and preserved separately. All twelve measured
parent/candidate comparisons are retained, with alternating order; adjacent
opposite-order pairs share an input version. Only two input fixtures were used.

| Target latency | A2-13 | A2-15 |
|---|---:|---:|
| Raw median | 312.032 µs | 295.904 µs |
| Raw observed range | 310.080–313.984 µs | 293.920–297.728 µs |
| Separately calibrated median | 308.928 µs | 292.800 µs |

All twelve measured comparisons favored A2-15. The median of six reverse-order
pair ratios is **1.054731×**, or **5.189% latency reduction**. The six ratios range
from 1.050148× to 1.057979×. This supports a modest within-session improvement,
not a confidence interval or proof across machines/seeds/sessions.

Six empty-graph calibration samples are retained, including the first 24.928 µs
sample; their median is 3.104 µs. Nothing was deleted after inspection. Separate
subtraction gives a paired ratio of 1.055312×. Raw timing remains primary;
this calibration is not the official judge's calibration.

The predeclared target was 10% latency reduction. **It was not met.** The prior
native report showed both page-load scheduling and K/V buffer changes induced
by the source edit. These timings do not isolate either mechanism's causal share
or establish bandwidth, occupancy or stall behavior.

For scale only: historical A2-13 full decode B32 contributes 84.446% of that
load. Transferring the observed 5.189% relative cut to that official path while
holding everything else fixed gives
`(1 / (1 - 0.8444616473 × 0.0518910800))^(1/6) = 1.007496`.
That is a **modeled +0.750% aggregate gain**, not an achieved score or prediction.
It is smaller than the official report's minimum ±1.5% interval. The changed
runtime/calibration and unmeasured controls make direct score extrapolation unsafe.

## Cost, cleanup and reproducibility

The prelaunch reservation was $0.146368320, within $0.150503782 remaining.
Existing Mac browser/CLI checks verified $28.77 credits, paid cap $0, $0 billed
charges and no active apps. One ephemeral H200, CPU cap 2, RAM cap 8 GiB, max 1
container, no idle capacity, no retries. The 52-second internal deadline, 55-second worker kill and 60-second
function timeout and 75-second app watchdog were not reached. Stop returned exit 0;
the separate status read confirmed zero tasks before reservation release.

Time-based allowance: $0.064547328. The 16:18:59 UTC workspace billing snapshot
rose from $1.23 to $1.30 gross, with **$0 paid**; the ephemeral breakdown rose
$0.06787048. Because the rounded observed increase exceeds the time model,
accounting uses **$0.07000000**. Workspace snapshots may include delayed prior
usage; this is not a final per-app invoice. Prior cost was never reduced.
Cumulative conservative accounting is **$1.919496218**, leaving **$0.080503782**
under the $2 stage ceiling. No active reservation or uncertain job remains.

Commands actually used locally:

```sh
.venv/bin/python -B reviews/attention-v2/a215/test_paired_worker.py
.venv/bin/python -B reviews/attention-v2/a215/paired_modal.py self-test
.venv/bin/python -B reviews/attention-v2/a215/analyze_pair.py
```

Six protocol tests, four budget/payload guards and the saved-result audit passed.
Native launch entrypoint was `paired_modal.py launch`, with the existing local
Modal configuration passed through the environment. Its persistent run record
prevents duplicate dispatch. Exact payload hashes, source commit, command
settings, app state and raw outputs are in `paired-modal-run.json` and
`paired-result.json`; derived arithmetic is in `paired-analysis.json`.

## Next step

Keep this component and its evidence, but leave A2-11 selected. Full official
correctness plus unchanged-path timing would be necessary before adoption;
a single near-noise aggregate receipt would still be inconclusive. Do not spend
another run merely to turn this small target improvement into an aggregate number.
The remaining $0.08050 does not fund the unchanged conservative $0.14637 safety
envelope, so this bounded Modal cycle ends here. No further GPU or official
request was started. Future local research can use the native FP8-to-FP16
lowering finding to investigate a larger mechanism without claiming an unmeasured gain.

## 中文实验卡

**假设：** 提前处理页地址并双缓冲 K/V，能降低 full decode B32 延迟。
**结果：** 两种页大小/缩放配置、同地址 A→B→A 和缓存检查通过；目标中位延迟
312.032→295.904 µs，配对延迟下降5.189%，12次比较都更快。
**判定：** 有小幅收益，但未达到预设10%；推算总分影响仅约0.75%，且不是实测总分。
保留候选，不替换 A2-11。应用已停止、任务数0；按$0.07计账，实际付费$0。
