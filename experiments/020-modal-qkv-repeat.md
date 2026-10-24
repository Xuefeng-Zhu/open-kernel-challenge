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
experiment018 evidence preserved. Native repeat outcome pending at preparation
commit. Record exact app and terminal results; never replace an uncertain request.
