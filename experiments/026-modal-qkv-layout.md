# 026 — Modal-first screening of QKV m>1 layout

Candidate025 (SHA50a76d4e...), control official best full composition c34c497b...,
plus supplied SGLang. Only the lossless BF16 QKV weight layout/dispatch differs.
Hypothesis and39CPU+3focused/static gates: experiments/025-qkv-layout-local.md.

Predefined screen-v1: full39CUDA cases for all3runners,6graph warmups retained
and excluded by protocol,30steady fresh-input replays per30shapes. All six runner
orders repeated5times, opposite orders paired, no sample removal or overhead
subtraction. Report target shapes, tails, unchanged paths and weighted gross load
ratios versus the same-app best; no official score or CI inferred. Seed1300000.

Reuse existing H200 image; one ephemeral app, max1/min0/buffer0,CPU2limit/RAM8GiB,
internal240s/wrapper255s,function300s,startup90s,retries0,scaledown2. Shorter bounded
run fits this library-layout screen. Reserve$0.60 with conservative stage total
before startup. Cumulative credit ceiling$2 includes all prior apps; net paid$0.
No new credentials/cap/payment/terms/push. Check live credit/cap/expiry and stop
state/accounting after each. Full best is explicitly allowlisted (18files+manifest).

User authorized autonomous remaining-cycle Modal-first iteration; official2/10used,
8remaining but only7platform checks left today. Do not spend official quota unless
this screen warrants a meaningful attempt. Preserve every failure/regression.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python modal_preflight.py --candidate candidates/projection_qkv_layout.py --build local-notes/modal-m26-bundle
.venv/bin/python modal_diagnostics.py --experiment M26-qkv-layout --bundle local-notes/modal-m26-bundle
.venv/bin/python finalize_modal_diagnostic.py M26-qkv-layout
```

Local95unit tests/syntax pass. Actual app ap-KFijr1ps085dubW3fuiteg completed and
verified stopped/tasks0. All3runners39CUDA checks and30shapes x6warmup+30steady
replays pass. Worker55.754s, app65s, no timeout. Same H200/public pinned image.

**Rejected for performance, no official admission.** Gross aggregate best/candidate
0.9984097. QKV m256 paired best/candidate0.9458453 (0/15winning pairs), about5.4%
slower, candidate median25.376us vs best24.064us. Other m8/16/32/128 ratios are
1.00168/1.01435/1.00682/1.01593, only8/10/10/12of15win: no robust whole-workload
benefit. m1 is unchanged; ratio0.99704 is noise/control variation. Preserve all
samples/source. Do not compose this layout or spend official quota on it.

Full evidence results/remote-modal-M26-qkv-layout.json/.csv; raw medians/tails,
source hash, all seeds and every pair retained. Billing asynchronously updates;
conservative cumulative guard$0.55673<$2, net paid$0. Next independent direction
is large-M unpooler layout, without any QKV-layout changes.
