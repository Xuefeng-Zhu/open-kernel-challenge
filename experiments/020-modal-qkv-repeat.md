# 020 — independent QKV m1 repeat, unchanged candidate

Hypothesis: the experiment018 m1QKV improvement persists under fresh inputs and
more balanced graph repetitions, rather than reflecting one noisy six-sample
window. No algorithm or candidate-byte change. SHA256
9cea9ceed1dab548368746e83540f25edff94f472e8701c05526ea9995759d55.
Control frozen R10 plus official SGLang. All samples retained, including first.

Same measured worker as experiment019; seed offset700000 and30repetitions per
shape/runner (five blocks of all six orders). Full39CUDA correctness cases per
runner use shifted fresh seeds too. Fifteen opposite-order pairs per shape;
record gross distribution and paired ratios, no official score/CI or subtraction.

Authorized continuation remains limited to these two diagnostics before reporting.
Stage credit ceiling$2total including the initial smoke, net paid cap$0, no new
credentials/cap/payment change, no OKC checks/push. Reuse same built image;
one H200/app at a time, bounded600s/540s, retries0, max1/min0,scaledown2, CPU2/RAM8GiB
limits. Experiment019 is stopped and costs reconciled before this launch.

```sh
.venv/bin/python modal_preflight.py --candidate candidates/projection_qkv_m1_simt.py --build local-notes/modal-m20-bundle
.venv/bin/python modal_diagnostics.py --experiment M20-qkv-repeat --bundle local-notes/modal-m20-bundle
.venv/bin/python finalize_modal_diagnostic.py M20-qkv-repeat
```

Local72unit tests/syntax passed for this unchanged driver. Original source and
experiment018 evidence preserved. App ap-8GLNIGgfFD2naewj9oD2vD completed/passed,
verified stopped/tasks0 at18:57:30UTC. Setup1.606s, worker55.997s, client63.842s,
whole app65s, no timeout. Candidate/R10/SGLang39/39CUDA checks each and all30graph
shapes x30fresh repetitions pass. Same H200/Torch2.14.1+cu130/Triton3.8.0 environment.

QKV m1 gross median16.544us vs R10 18.608us vs SGLang18.816us. Fifteen reversed-
order paired ratios vs R10 have median1.13013,13/15win. Candidate first samples
43.904us and41.600us retained; p9016.963us, MAD0.112us. Two pairs lose because of
these samples. Repeated median benefit, unresolved startup/tail spikes; no outlier
removal, official score/CI or overhead subtraction. Unchanged-path paired ratios
range0.9666–1.0166. See MODAL_STAGE2_RESULTS.md for qualifications.

Latest cumulative billing$0.21metered, credits-$0.21,billed$0; unrounded apps
$0.20879646. Conservative stage bound$0.328587546 below$2 including initial smoke.
All apps stopped; no extra run, OKC check, credential, cap change or push.
Raw samples/paired indices/seed/environment/oracle results:
results/remote-modal-M20-qkv-repeat.json/.csv.
