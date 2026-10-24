# 030 — large-M pooler grouped GEMM screen

Independent source029 SHA573f2f9e..., full-composition best control c34c497b and
supplied SGLang. No QKV or unpooler layout experiment composed. Native GPU risk:
new Triton BM32/BN128/BK64/GM8 full-K loop may be memory bound or regress versus
library tactics. CPU models cannot answer compilation/correctness/performance.

Predefined screen-v1: full39CUDA checks each, nearby pooler graph correctness
m127/128/129/255/257 including N4093, two fresh replays each. Six retained graph
warmups excluded by protocol;30steady repeats, all six orders x5, reverse-order
pairs, all samples retained. Seed1900000,256MiB coldL2 outside timing events,
no overhead subtraction. Promote only meaningful whole-workload evidence;
otherwise retain regression and spend no official check.

Same existing image/token,18source/license allowlist+manifest. One ephemeral H200,
CPU2/RAM8GiB request+limit, max1/min0/buffer0, scaledown2,retries0, function300s,
startup90s,internal240s/wrapper255s. Reserve$0.60within cumulative$2free credits,
net paid cap$0. No credentials/cap/payment/terms/push changes.

```sh
.venv/bin/python modal_preflight.py --candidate candidates/projection_pooler_grouped.py --build local-notes/modal-m30-bundle
.venv/bin/python modal_diagnostics.py --experiment M30-pooler-grouped --bundle local-notes/modal-m30-bundle
.venv/bin/python finalize_modal_diagnostic.py M30-pooler-grouped
```

Preparation:39CPU+3focused checks, static/syntax pass. Modal pending. No official
admission for this source at preparation commit.
