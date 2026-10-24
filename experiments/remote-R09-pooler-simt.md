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
