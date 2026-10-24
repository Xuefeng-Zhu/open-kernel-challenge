# R09 — one-row SIMT BF16 pooler

Source `candidates/pooler_simt_row.py`, commit
`e4a262b2f727c9f26971c4cab6e331a4dac5d46c`, SHA256
`f78af55dfd533ec18d776752f93e9754005ec0361c4bd43f437aa072f9a04e30`.
Hypothesis/configuration/CPU gates: local experiment013. Only pooler m==1 changes
atop R01 control: FP32 SIMT products/reduction from original BF16 operands, one CTA
per output row, no separate partial-output reduction. Other shapes remain R01.

```sh
python3 remote_check.py send R09-pooler-simt candidates/pooler_simt_row.py
python3 remote_check.py poll R09-pooler-simt
python3 summarize_check.py R09-pooler-simt
```

Sole accepted off-board job `97aa5690ff134369`,2026-10-24T16:15:17Z. Admission9/10,
$0/no scored entry, currently pending. Never resubmit or infer native correctness
from CPU. Compare pooler m1 to R01/R06, inspect unchanged shapes and all6 loads.

## Terminal measured result

Job `97aa5690ff134369` done/pass,9/9 prerequisites and30/30 scored-shape
checks, no flags/rejections. Official1.0076197260× SGLang,95%interval
[0.9925054301,1.0227340219]; overall parity, not a demonstrated whole-workload win.
Reference-normalized ratio1.1101946316× original starter across separate trials.

Targeted pooler m1 **18.3032µs** versus R01's21.6569µs; within-run SGLang-normalized
shape ratio is **1.19824× versus R01**. R09 m1 speedup over its own SGLang is above1;
other pooler shapes stay on the R01 path, with normalized ratios0.9941–1.0134 versus
R01, consistent with modest cross-run variation. Warm diagnostic _gemv18.4µs is not
substituted for official cold timing. No occupancy or HBM bandwidth inference.

This supports the single-row component despite the unchanged whole-workload paths
dominating the overall score. Full samples/profiles/environment in
results/remote-R09-pooler-simt.json and CSV; COMPARISON_R01_R09.md reproduces the
cross-run control comparison. Preserve source unchanged. The remaining tenth check
may test a transparent composition with verified router paths; do not predict its
score by adding component gains. No paid resource or scored entry.
