# R03 — earlier hi+lo dispatch on H200

Hypothesis: change only the R01 threshold from 8 to 1, using legal hi+lo at m=8,
while retaining m=1 FP32 and all other operations/layouts. Source:
`candidates/router_split_after1.py`, commit `25cfb05d081ad25f691f96a3dd2d90da331ee808`,
SHA256 `5945aa3b214cece1363641a86b7cae2827cdd9d7af956fa42eab4af91994fdb8`.

```sh
python3 remote_check.py send R03-router-dispatch candidates/router_split_after1.py
python3 remote_check.py poll R03-router-dispatch
python3 summarize_check.py R03-router-dispatch
```

Sole accepted job **718e0730e1bd4c30**, mode check, terminal **done**.
**9/9 prerequisite and 30/30 scored-shape checks pass**; no flags/rejections.
Official speedup **1.0267803842× SGLang**, 95% interval
**[1.0113786784,1.0421820900]**.
This validates the dispatch hypothesis in the official trial. It is independent of
R02's larger-M wide product; neither result proves the untested combined dispatch.
Reference-normalized ratio over the original starter: **1.1313058299**,
a cross-run comparison, not paired same-input timing against the starter.

Exact H200/driver/image/evaluator metadata, all 30 adjusted/raw timing records,
profiles, confidence interval and weighted-load recomputation are in
`results/remote-R03-router-dispatch.json` and companion CSV. New allowance admission
3/10; global quota snapshots include other concurrent admissions. No paid resource,
scored/final entry, new agreement/credential or duplicate POST. Result commit is the
commit containing this completed record. R04's custom low-batch kernel remains pending.
