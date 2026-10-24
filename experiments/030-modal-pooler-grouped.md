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

Terminal: app ap-cjt62CgauSmkFmLcwhlgPT stopped/tasks0, worker69.756s/app80s.
Full117official CUDA checks +30nearby fresh graph checks +all30timed shapes pass.
Rejected performance: aggregate best/candidate0.9003443; pooler m1280.4118493
(67.264us vs27.712),m2560.4621926(74.496vs34.448),0/15pairedwins each.
No official request. All samples preserved results/remote-modal-M30-pooler-grouped.
Billing snapshot$0.73gross/$0net; stageconservative$0.817885. No new credential.
One plausible bottleneck is low grid parallelism: BM32/BN128 gives128CTAs at
m128, below132SM. This does not prove the cause. Next controlled BN32 yields512
CTAs at sameM and keeps all other source/loop/precision choices unchanged.
