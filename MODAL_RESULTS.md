# First Modal H200 smoke — completed, stopped, no official entry

Current continuation: see MODAL_STAGE2_RESULTS.md. Unpooler019 and independent
QKV020 both completed/passed and stopped; all native correctness checks pass.
Latest cumulative Modal billing$0.21credits/net$0, conservative bound$0.3286below
$2. No new official check or push. Report before expanding experiments.

App `ap-ukvOyTeF1ztdOLggl5xJee`, created2026-10-24 18:31:07UTC,
stopped18:40:10UTC. Supported `modal app list --json` confirms stopped, tasks0.
Exactly one application/function invocation; no retries or persistent deployment.
The initial smoke limit was a staged safety guard; continuation019/020 is recorded separately. New official
dispatch remains paused: cycle2 1/10used,9remaining, all accepted jobs terminal.

## Actual native result

Candidate `candidates/projection_qkv_m1_simt.py`, source544f4a8, SHA256
`9cea9ceed1dab548368746e83540f25edff94f472e8701c05526ea9995759d55`.
Frozen bundle source commit `8f0148036e19cd49ec07a529c6be373bcaea22cc`.
Only QKV m=1 gains the unchanged R10 pooler SIMT body. No new pooler layout or
unpooler optimization composed into it; original BF16 weights preserved.

- Candidate/R10/SGLang: each **39/39 native CUDA correctness cases passed**,
  original weights and activations unchanged, official FP64/probe/row oracle.
- All30 scored shapes,6fresh-input graph replays per runner/shape, six balanced
  execution orders, outputs checked each replay: passed.
- NVIDIA H200, capability9.0, Python3.12.3, Torch2.14.1+cu130, CUDA13.0,
  Triton3.8.0, SGLang metadata0.0.0.dev0. TF32false; BF16 reduced reductiontrue.
  Public SGLang base digest is in the bundle manifest; private judge overlay differs.
- Initial app/image setup486.849s; GPU worker48.875s; client total542.120s.
  Individual correctness stages5.346/4.384/18.792s; graph phase17.226s.
  These are stage wall times including imports/compile; not kernel latency.

| QKV m1 gross graph time | Candidate | R10 | SGLang |
|---|---:|---:|---:|
| Median of6samples, microseconds |16.960|19.152|19.728|

Observed candidate latency11.45%below R10 (gross ratio1.12925),14.03%below SGLang
(gross ratio1.16321) **on this shape in this diagnostic**. No official score or CI.
First candidate sample38.176us retained; other samples17.216,16.160,18.144,16.512,
16.704us. Do not remove the outlier. Six samples are a smoke, not strong statistical
evidence. Unchanged paths vary by several percent between candidate/R10, illustrating
the noise. Do not infer whole-workload superiority or leaderboard placement.

Gross times include graph overhead. No guessed4us subtraction or judge-statistic
shortcut. Full per-shape times, every raw repetition/seed/order, source hashes,
environment and all three correctness reports: `results/remote-modal-smoke.json`;
compact timings: `results/remote-modal-smoke.csv`.

## Credits and account state

Latest supported `modal billing summary --json` snapshot reports gross metered
**$0.12**, credits **-$0.12**, billed **$0**; unrounded ephemeral-app breakdown
$0.12054281. Before run all costs zero and no apps existed. UI counters update
asynchronously: earlier app total$0.07/H200$0.06, later account breakdown$0.12 while
header still$29.93credits. Preserve the snapshot timestamps; do not claim a final
invoice or a perfectly synchronized balance. $0 net Spend limit reverified after
the run. No paid service resumption, payment change or upgrade.

Stripe's unexplained “Your service is paused” did **not** prevent this approved
credit-covered H200 execution. No inference about the underlying Stripe field or
future entitlement. The separate12-hour token is task-local ignored, mode0600,
expires2026-10-25 06:26:47UTC. No secret is in the evidence or bundle.

## Reproducibility and next measurement

Exact host command and resource bounds: `experiments/018-modal-initial-smoke.md`;
ledger: `experiments/modal-smoke-budget.json`. Do not rerun its launch command: the
one-run guard refuses all repeats. Container internal540s deadline, wrapper555s,
function600s/startup120s, retries0, max1/min0/buffer0, scaledown2, no detach.

Local checks after client installation:69unittests pass, syntax pass, `pip check`
reports no broken requirements. Frozen bundle has17source/license files plus a
generated hash manifest; no git/config/auth/cache/attention or unrelated files.

Before an additional run, obtain its budget authorization. Ranked next measurements:
1. Repeat QKV m1 under several seeds and more balanced orders, retaining all samples,
   to test the apparent improvement and first-replay outlier.
2. Test independent unpooler m1 candidate (experiment017), with full39CUDA gates;
   currently CPU/static-only. Smaller CTA count may hurt bandwidth utilization.
3. Separately compose only supported pooler-layout and QKV components after adequate
   isolated evidence; keep R10 and P2-01 controls. Revalidate every shape and nearby
   probes. Any Modal measurement remains distinct from a future approved OKC check.
