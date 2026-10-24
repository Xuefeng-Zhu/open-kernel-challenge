# R06 — isolated low-batch BF16 pooler split-K

Source: `candidates/pooler_triton_split8.py`, commit
`971ade4100cc5621c2baa4d3b87be346f1eeaae2`, SHA256
`76f202f5fe9efb23ce0e10c6739871fab74512464a0e9f1229cda53c0c625406`.
Hypothesis/configuration and actual CPU gates are in local experiment007.
Only pooler m<=8 changes atop R01 control; BF16 weights remain BF16.

```sh
python3 remote_check.py send R06-pooler-triton8 candidates/pooler_triton_split8.py
python3 remote_check.py poll R06-pooler-triton8
python3 summarize_check.py R06-pooler-triton8
```

The previously denied command was verified not executed, then retried exactly once
after new specific human approval. Accepted at 2026-10-24T16:02:53Z as check job
`b70d87fc51864276`, confirmed queued. No second job or uncertain reservation.
Admission6/10, $0/no scored entry. Follow this ID to terminal; no result inferred yet.
Historical automatic-review denial and its resolution are preserved in `DENIED_R06.md`.
The CLI initially returned minimal `job_id/status=received`; the read-only wrapper
now preserves that pending acknowledgement without inventing terminal result fields.

## Terminal measured result

Job `b70d87fc51864276` is **done**, passed9/9 prerequisites and30/30 scored-shape
checks, no flags/rejections. Official speedup **0.9930291495× SGLang**, 95% interval
**[0.9781337122,1.0079245867]**, reference-normalized ratio1.0941187457× over our
original starter across runs. This is not an improvement over the validated R01
control or a demonstrated win over SGLang.

The targeted pooler m1/8 is22.7957/23.0604µs versus R01's21.6569/21.1521µs.
Within-run SGLang-normalized shape comparisons also regress (ratios0.9550/0.9270
versus R01). Warm diagnostic profiles place94–95% in the partial kernel,20.7–21.8µs,
with1.2–1.3µs reduction. Warm profiles are not substituted for official cold timings.
Do not compose this failed performance hypothesis into the current best. No specific
occupancy/HBM bottleneck is claimed from timings alone.

Full sanitized records, all timing samples, profiles and environment hashes:
`results/remote-R06-pooler-triton8.json` and CSV; reproducible cross-run comparison
`COMPARISON_R01_R06.md`. The related QKV split8 source stays CPU-only: a different
footprint alone is insufficient reason to spend another check on this now-regressing
strategy. Next distinct hypotheses are router split-count isolation and, if justified,
a single-row SIMT BF16 GEMV avoiding the padded tensor-core path and extra reduction.
