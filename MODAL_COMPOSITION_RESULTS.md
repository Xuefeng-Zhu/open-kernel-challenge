# Composition and stability result — one stopped H200 app

Best observed composition: **candidates/projection_full_composition.py**, SHA256
`c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7`, source commit
`b9b0fd9`. It adds verified QKV/unpooler m1 SIMT paths to R10 and the separately
measured pooler BF16 transposed layout for m>1. All independent sources remain.
The layout-free native composition is retained as a close control/fallback.

App **ap-XxzsrPfRejsEpPVQCxIqzJ**, source bundle commit
`a50ae6c45e409e6e628c2d6cea61919c3d5785b5`, stopped at 2026-10-24 19:14:31 UTC,
tasks 0. No retries or replacement. Setup 2.157 s, worker wall 61.223 s, client 68.667 s,
whole app 70 s. These include CPU/waiting, not GPU-busy measurements. All four scoped
Modal apps are stopped. No new OKC check, ranked entry, push or paid-service change.

## Correctness and measurement contract

Five runners: full composition, layout-free native composition, P2-01 pooler-layout
control, frozen R10, official SGLang. **39/39 CUDA correctness cases each** with
fresh shifted seeds and unchanged official FP64/probe/per-row oracle; weights and
activations unchanged. All 30 scored shapes passed 900 predeclared-warmup output
checks and 4500 steady replay output checks. Local 92 tests and syntax/static checks
pass; each new source separately passed 39 CPU + 9 focused cases before GPU execution.

Protocol023 was committed before launch: exactly 6 fresh-input graph warmups per
runner/shape, all timed and saved; then 30 steady samples per runner/shape. Ten
position-balanced cyclic/reverse orders repeated three times, paired with their
reverses. Every steady sample used once; no post-hoc deletion. Cold L2 before
each event interval, no fixed-overhead subtraction. See experiment023 for commands.

Hardware: NVIDIA H200, 132 SMs, CUDA usable memory 150101688320bytes. Driver 610.57.04;
before/after snapshots both P0, SM 1980MHz, memory 3201MHz, power limit 700 W; no clock
settings changed/locked and no continuous utilization measurement. Python 3.12.3,
Torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0, public SGLang base/SGLang metadata 0.0.0.dev0.
TF32 false, BF16 reduced reduction true. Private OKC overlay remains distinct.

## Aggregate evidence, gross diagnostic only

Official shape multiplicities form each load's gross time; reversed-order paired
load ratios are reduced by median, then equally weighted across six loads by
geometric mean. These include graph overhead: **not an official score or CI**.

| Comparison, control/full | Gross weighted aggregate ratio | Overall pair directions |
|---|---:|---|
| R10 / full |1.022014|15/15 above 1|
| P2-01 pooler layout / full |1.010889|14/15 above 1|
| Layout-free native / full |1.007073|13/15 above 1|
| SGLang / full |1.119534|15/15 above 1|

Full is the best point estimate. Its 0.7%margin over native-only composition is
small; no formal statistical superiority claimed. Per-load R10/full ratios:
decode1 1.08099, decode16 1.00866, decode32 1.01270, verify1 1.00948,
verify16 1.02265, verify32 0.99969. Large verify blocks remain nearly unchanged.

## Isolated pooler layout decision

Old official cross-run results were suggestive only. The new direct P2-01/R10
paired comparison and full/native comparison agree on four useful m>1 cases:

| Pooler m | Full median us | R10 median us | P2-01/R10 isolated paired speedup | Winning pairs |
|---|---:|---:|---:|---:|
|8|24.592|25.488|1.04151|15/15|
|16|24.960|25.936|1.05350|15/15|
|32|24.960|26.160|1.04491|15/15|
|128|28.000|30.064|1.07562|15/15|
|256|34.000|34.352|0.99693|7/15|

Keep the full composition as best observed; retain native-only fallback. At m=256,
paired variation is wide (isolated0.9568–1.2365) and median is about 0.3% below 1;
this is unresolved, not a claimed gain or a basis for a new shape-specific cutoff.
No known regressing split8 pooler or lower-precision weights were composed.

## QKV startup spikes and steady tails

Full QKV m1 warmup samples: **40.000,16.256,17.408,16.736,16.736,16.512 us**.
The native control's second warmup is 32.512 us. All remain in raw records. After
the predeclared six warmups, full QKV steady median 16.768 us, p90 17.680 us,
max 17.952 us (range16.160–17.952), versus R10 median19.008 us. No 40us-level spike
occurred in these 30 steady samples. This supports the fixed warmup protocol, but
does not prove the underlying graph-upload/driver/cold-start cause.

Full unpooler m1 steady median 8.384 us, p90 8.714 us, max 8.864 us; R10 median 11.552 us.
The paired R10/full medians are 1.13495 for QKV and 1.39165 for unpooler.
All retained steady samples of these two targets improve in this window; unrelated
o/router paired ratios vary 0.9863–1.0159, demonstrating residual measurement noise.

## Remaining bottlenecks and costs

Mean share of each load's gross weighted time: pooler 29.00%, QKV 21.73%,
unpooler 20.83%, router 15.72%, o 12.72%. Medium/large-M projections, especially
pooler m256 and QKV/unpooler beyond m1, dominate the remaining work. These shares
include replay overhead and are not pure hardware utilization or attainable gains.
Next hypotheses should target these cases and retain held-out/nearby-shape checks.

Latest cumulative billing includes all four Modal apps: **$0.29 metered,
-$0.29 credits, $0 paid**, detailed apps $0.28971725. Conservative stage bound
$0.446246746 versus$2 ceiling; no cap change/new credential/upgrade. Counters can
settle asynchronously. Existing 12-hour credential remains ignored/private and
expires 2026-10-25 06:26:47 UTC. Verify actual cap/balance before any future stage.

Full raw warmup/steady samples, seeds/orders, hashes, per-shape distributions,
load aggregates, hardware/software, oracle reports and termination/billing snapshots:
**results/remote-modal-M23-composition.json** and its CSV. Durable ledger:
experiments/modal-diagnostics-budget.json. Original official best point score stays
P2-01 1.1290207051x; the new composition has no official check yet. Official cycle2
1/10 used, 9 remaining, dispatch paused. Report here before further expansion.
