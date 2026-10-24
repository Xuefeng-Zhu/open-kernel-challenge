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
