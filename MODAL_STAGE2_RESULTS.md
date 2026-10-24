# Modal continuation: unpooler and independent QKV repeat

Both requested experiments completed, passed and are verified stopped with zero
tasks. No further experiment started. No official OKC request or push. The prior
single-smoke guard was a staged safety limit; the user authorized Modal-first
experimentation. This continuation stays within the conservative $2 total-credit
ceiling and reports before expanding scope.

| Experiment | App | Setup / worker / client seconds | Outcome |
|---|---|---|---|
| 019 unpooler m1 | ap-dMJsl67NMlcBiZs5KJWUFv | 2.281 / 47.775 / 54.924 | stopped, passed |
| 020 QKV m1 repeat | ap-8GLNIGgfFD2naewj9oD2vD | 1.606 / 55.997 / 63.842 | stopped, passed |

Whole app lifetimes were56s and65s respectively. Worker/client wall times include
CPU work and waiting and are not GPU busy time. No timeout. Reused image
im-cPTbGpNiX3RarKResPgX7h; no new base-image pull/build. One H200/app at a time,
max 1/min 0/buffer 0, scaledown 2, timeout 600/startup 120, internal 540/wrapper 555,
retries 0, CPU 2 request/limit and 8 GiB hard memory limit. No secrets, volumes,
schedules or deployment; only 17 allowlisted sources/licenses and a hash manifest.

Each app ran candidate, frozen R10 and official SGLang against all 39 CUDA correctness
cases using fresh shifted seeds, then all 30 scored shapes with fresh inputs for
each graph repetition. Every output passed the original FP64/probe/row oracle;
weights and activations unchanged. Native environment: NVIDIA H200/SM90,
Python 3.12.3, Torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0, SGLang 0.0.0.dev0;
TF32 false, BF16 reduced reduction true. Public base, not the private OKC judge overlay.
Driver version, GPU clocks/utilization and GPU-busy duration were not sampled;
do not infer them from worker wall time.

| Target, gross graph microseconds | Candidate median [p10,p90], max | R10 median | SGLang median | Paired R10/candidate median |
|---|---|---:|---:|---:|
| unpooler m1, 12 repetitions | 8.368 [8.131,8.573],9.728 |11.552|11.744|1.38704 (6 pairs)|
| QKV m1, 30 repetitions |16.544 [16.346,16.963],43.904|18.608|18.816|1.13013 (15 pairs)|

Unpooler m1 median latency fell 27.56% relative to R10. All six reversed-order pairs
win (ratios 1.3204–1.4230); candidate max 9.728 us is below R10 min 11.424 us in this
12-sample window. The m8 original path has paired ratio 1.00057. Promising isolated
component, not a measured combined implementation or official whole-workload gain.

QKV repeat median latency fell 11.09% relative to R10; 13/15 pairs win. Candidate first
two samples were 43.904 us and 41.600 us; these produce the two losing pairs and are
fully retained. Initial smoke also had a first-sample 38.176 us spike. Median benefit
repeats under new seeds, while startup/tail behavior remains unresolved. All raw
samples, distributions, seeds, pair indices and correctness reports are preserved.

The harness warms native calls before capture but does not warm graph replays
before timing. NVIDIA documents extra first-launch graph-upload cost
([official technical article](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)).
That is a **possible measurement hypothesis**, not a proven explanation for these
two spikes (it does not automatically explain the second replay). No samples were discarded and no warmup/protocol change was made mid-run.

Each six-order block pairs reversed permutations (0,5),(1,3),(2,4), using every
sample exactly once; ratios are geometric means within each pair, then medians.
These are gross diagnostic comparisons, no guessed 4 us subtraction, official CI,
leaderboard score or prize/rank prediction. Unchanged-path paired ratios span
0.9662–1.0346 in 019 and 0.9666–1.0166 in 020. Preserve that variation; do not attribute
small unchanged-path differences to the candidate.

## Exact sources and evidence

- 019 source bundle commit 7b0efd963c39092003033c535fa0725f6035c9c7;
 candidate SHA256 65ae42bfc6214dc84239f9e6c0155980464f53c77c1f8c42d19f76ed82290ec0.
- 020 source bundle commit c22d5a9; candidate unchanged from 016/018, SHA256
 9cea9ceed1dab548368746e83540f25edff94f472e8701c05526ea9995759d55.
- Full records: results/remote-modal-M19-unpooler.json/.csv and
 results/remote-modal-M20-qkv-repeat.json/.csv. Commands, hypotheses, source/result
 commits: experiments/019-modal-unpooler.md and 020-modal-qkv-repeat.md.
- Budget/state: experiments/modal-diagnostics-budget.json. Initial-smoke ledger and
 all original baseline/R10/P2-01 artifacts remain preserved separately.

Latest supported workspace billing: **$0.21 metered, -$0.21 credits, $0 billed**;
unrounded app breakdown $0.20879646. Includes the initial smoke and both continuation
apps. Earlier snapshots $0.12/$0.07/$0.13 are preserved; counters settle asynchronously.
Final Mac UI independently shows $29.79 credits left, $0.21 used, $0 spend cap and
$0 total spend. Conservative whole-app-time stage bound $0.328587546, below $2. No new
credential, lifetime extension, cap/payment/plan change or paid-service resume.
Existing token expires 2026-10-25 06:26:47 UTC; keep it private/ignored.

Local 72 tests and syntax checks passed. Next decision: investigate graph-first-replay
spikes with a separately recorded protocol, then consider an isolated composition
of supported pooler layout, QKV m1 and unpooler m1 while preserving R10 controls.
No composition or extra run is implemented/launched in this report; report first
as requested. Official best point score remains P2-01 1.1290207051x, not these
single-shape Modal ratios. New official dispatch remains paused.
