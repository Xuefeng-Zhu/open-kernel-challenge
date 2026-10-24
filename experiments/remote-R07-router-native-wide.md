# R07 — native low-M plus wide large-M router composition

Source: `candidates/router_hybrid_native.py`, commit `90032a97994ad51e7779793000946545a76ba86d`,
SHA256 `d8e29be3b47aa9ad248f7e068edfe24c3432a2c138b6b464cdfe7cddbf58cace`. Exact component configuration and CPU gates: experiment011.
Hypothesis: retain R04's measured low-M gains and R02's larger-M gains in one
shape-dispatched file, without changing their arithmetic/tile/packing parameters.

```sh
python3 remote_check.py send R07-router-native-wide candidates/router_hybrid_native.py
python3 remote_check.py poll R07-router-native-wide
python3 summarize_check.py R07-router-native-wide
```

Accepted sole off-board check job `26ea99ee0869482a` at 2026-10-24T16:05:02Z.
Admission7/10, no paid/scored entry. Result pending; never resubmit this job/source.
Compare aggregate confidence interval and every router shape to both controls,
including unaffected BF16 projections, rather than assuming additive speedup.

## Terminal measured result

Sole job `26ea99ee0869482a` is **done/pass**,9/9 prerequisites and30/30 scored-shape
correctness checks, no flags/rejections. Official **1.0991970009× SGLang**,95%interval
**[1.0827090459,1.1156849559]**. Reference-normalized ratio to our original starter
is **1.2110944020×**, across independent runs rather than a paired starter trial.
This is the new highest passing observed score; its reported interval is above the
previous R05 interval. That is stronger evidence than the earlier overlapping
R02/R04/R05 differences, but it is still one full-workload check, not final placement
or held-out/nearby-shape validation. No repeated-run confidence is invented.

Router m1/8:9.2852/9.1417µs; m16/32/128/256:9.2756/9.6759/10.3692/10.9153µs.
Versus R02, normalized router ratios are1.1004/2.8450/1.0164/1.0232/0.9796/1.0187
in m1/8/16/32/128/256 order. The m128 difference is slightly worse across runs;
large-M arithmetic/layout is unchanged from R02, so do not attribute noise to a new
algorithm. Versus R04, the large-M route now uses the measured wide GEMM. All six
weighted loads exceed SGLang in this trial, with verify1 at1.2982×.

Actual samples/profiles/evaluator+image/device provenance are in
`results/remote-R07-router-native-wide.json` and CSV. Reproducible control comparisons
are COMPARISON_R02_R07.md and COMPARISON_R04_R07.md. Source remains immutable at the
recorded source commit. Baseline `submission.py` remains unchanged; this is an
isolated off-board candidate, no scored/public/final entry or paid resource.
