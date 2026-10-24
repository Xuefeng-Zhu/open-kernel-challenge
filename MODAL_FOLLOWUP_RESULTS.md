Latest focused follow-up: [QKV_M128_SINGLEPASS_RESULTS.md](QKV_M128_SINGLEPASS_RESULTS.md).
H2tested at BM64/BN32, emitsWGMMA, correctnesspass but55.10us vs21.89us,
0/15pairedwins. Rejected/noofficialcheck; broadH1pipeline remains unimplemented.
All11apps stopped,112localtests pass,Modalguard$1.146412/$2,netpaid$0.
Earlier dated diagnostic/hypothesis context follows unchanged.

Latest follow-on resource/timeline diagnosis: [HOPPER_DIAGNOSTICS.md](HOPPER_DIAGNOSTICS.md).
M34completed/all10apps stopped,108localtests pass; best and officialbudget unchanged.
M33CLI failure preserved/resolved. Conservative stageguard$1.034495/$2,netpaid$0.
Two future hypotheses assessed; no new algorithm benchmark or officialcheck.

# Modal follow-up projection screens — 2026-10-24T20:02:58.403543+00:00

Best unchanged: candidates/projection_full_composition.py, source
b9b0fd9fb0cf326b5836167ffc29d1a0d3386fa7, SHA
c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7.
Official off-board48b1357016134e52:1.1486499523x SGLang,
95%CI[1.1314202031,1.1658797016],9+30pass/flags[]. Original starter0.9076064x;
independent reference-normalized ratio1.26558, not paired. Prior control CIs overlap.
No new candidate promoted, no new official admission, ranked entry or push.

| Screen | Isolated hypothesis | CUDA result | Paired gross best/candidate | Decision |
|---|---|---|---|---|
| M26 | QKV m>1 contiguous BF16 W.T |39cases each/control pass|aggregate0.998410; m2560.945845 (0/15wins)|reject regression|
| M28 | unpooler m>8 contiguous BF16 W.T |39cases each+30nearby replays pass|aggregate0.999841; m1281.026538 (15/15wins)|reject whole replacement/parity|
| M30 | pooler m>=128 BM32/BN128/BK64/GM8 full-K GEMM |39cases each+30nearby replays pass|aggregate0.900344; m1280.411849,m2560.462193|reject severe regression|
| M32 | change only BN128to32 (raise CTAcount) |39cases each+30nearby replays pass|aggregate0.863060; m1280.431404,m2560.277096|reject, stop basic-tile route|

The ratios are all-sample reverse-order paired Modal diagnostics, not official
scores/CI. Within each app: candidate, immutable official-best control and SGLang,
full39CUDA gates each;6predefined graph warmups retained/excluded,30steady fresh-
input replays per30shapes, six balanced orders repeated5times, reverse orders
paired,256MiB cold-L2 read outside CUDA-event interval. No sample deletion/fixed
cost subtraction. All warmup/steady outputs checked. Nearby graph cases test
m8/9/16/129/257 unpooler or m127/128/129/255/257 pooler and noncanonical N tails;
inputs/weights unchanged. CPU models do not execute Triton.104local tests pass.

M28's single m128 gain is encouraging but narrow. Applying its measured2.65%ratio
only to that shape while freezing every official P2-02 timing gives a *hypothetical*
1.001115 aggregate multiplier (~0.11%), not a measured score. Other unchanged
paths fluctuate; this is insufficient evidence to spend official quota now.

M30 low grid parallelism was a plausible hypothesis (128CTAs on132SM at m128).
M32 increases it to512CTAs but does not recover performance. That rules out this
simple proposed remedy; timings alone do not prove bandwidth, spills, registers,
shared-memory stalls or MMA instruction choice as the cause. Do not label an
unmeasured resource bottleneck as confirmed. No post-hoc score reconstruction.

Hardware/runtime: actual NVIDIA H200,132SM,driver610.57.04,P0/1980MHz SM/3201MHz
memory/700W snapshot; no clock changes. Torch2.14.1+cu130,Triton3.8.0,Python3.12.3,
SGLang0.0.0.dev0 from pinned public base image/reusedimageim-cPTbGpNiX3RarKResPgX7h.
Public image differs from private judge overlay. Setup/wall duration differs from
GPU busy time. Same-app medians/tails/allseeds/full source hashes in
results/remote-modal-M26-qkv-layout.json/.csv,
results/remote-modal-M28-unpooler-layout.json/.csv,
results/remote-modal-M30-pooler-grouped.json/.csv,
results/remote-modal-M32-pooler-narrow.json/.csv.

## Stop, accounting and next priorities

All8Modal apps, including initial smoke, verified stopped/tasks0. No active or
uncertain official job. Last CLIbilling snapshot$0.87000000gross, creditedfully,$0net;
observed gross highwater$0.87077815. Accounting counters
settle asynchronously and have revised earlier values. Conservative entire-app
stage guard$0.964253594<$2, not billed cost. Initial smoke included. Existing paid cap
$0 verified in Mac browser; no credential/cap/payment/plan/terms change. Existing
12hcredential expires2026-10-25 06:26:47UTC; never renewed or included in bundle.

Official second-cycle2/10used,8remaining. Fresh platformsnapshot2026-10-24T19:59:17.493091+00:00:
checks13/20used,7left today; scored0/10. Reset00:00UTC, next2026-10-25T00:00:00Z.
No quota blocker today. Stop early because these candidates do not warrant admission,
not because approval is missing. Exact scoped wrapper stays held pending a justified
Modal-qualified plan; user authorization for remaining meaningful checks persists.

Ranked next research, no unmeasured implementation claimed:
1. Keep current best. Obtain compiler/resource and GPU memory/occupancy evidence
   for QKV split-K reduction and pooler m128/256 rather than blindly sweep tiles.
   Actual official shares pooler32.14%,QKV22.42%,unpooler21.01% prioritize this.
2. If profile supports it, prototype a Hopper-specific async/TMA/WGMMA pipeline
   with legal BF16 operands, full-K correctness and nearby-shape checks. Basic
   tl.dot loop variants here do not establish any benefit for that architecture.
3. Revisit independent unpooler m128 dispatch only alongside stronger improvements
   and another stability screen; retain m256/default paths to avoid regressions.

No new GPU app is needed to complete this bounded stage. None of the authorized
operations in this stage was denied or left pending/uncertain. Further architectural
work is unimplemented/unmeasured, a technical limitation rather than an access failure.

## Exact commands

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python check.py --submission candidates/projection_pooler_narrow.py --device cpu --suite all --output results/router-cycle2-pooler-narrow-cpu.json
.venv/bin/python static_check.py --submission candidates/projection_pooler_narrow.py
.venv/bin/python analyze_gap.py P2-02-full-composition
```

Exact source/Modal launch/finalize commands and hypotheses are in experiment025–032
records. Launch commands are historical reproducibility records, not instructions
to duplicate accepted apps. Host controller rejects an already journaled ID; prepare
an independently justified new experiment before any future admitted request.
