# R08 — router split count16→32 only

Source `candidates/router_triton_split32.py`, commit
`7a826459b20a71076dc465681621c8928ffea813`, SHA256
`3f269bf18100c8dbe111dfe6f4380ab323f684a23855e9a0fee5379bc451adf9`.
Exact hypothesis, configuration and CPU results: local experiment010. R04 control
has79% diagnostic time in the partial kernel; change only split count to increase
parallel work while increasing temporary/reduction traffic. No occupancy claim.

```sh
python3 remote_check.py send R08-router-split32 candidates/router_triton_split32.py
python3 remote_check.py poll R08-router-split32
python3 summarize_check.py R08-router-split32
```

Sole accepted check job `ab74f25caa2e46a8`,2026-10-24T16:12:21Z. Admission8/10,
$0/no scored entry, result pending; no duplicate request. Compare low-M router
shape times/ratios to R04; larger-M control remains R01, not the native/wide hybrid.

## Terminal measured result

Job `ab74f25caa2e46a8` is done/pass,9/9 prerequisites and30/30 scored-shape
checks, no flags/rejections. Official1.0617021045× SGLang,95%interval
[1.0457765730,1.0776276361], reference-normalized1.1697825543× original starter.
The whole-workload interval overlaps R04; do not claim a statistically resolved
aggregate difference. This source retains larger-M R01 control, not R07's wide path.

Targeted router m1/8: **7.3016/6.8717µs**, versus R04's9.1967/9.2721µs.
SGLang-normalized shape speedup ratios versus R04 are **1.2773/1.3435**; m16/32/128/256
ratios remain0.9945/0.9972/1.0079/0.9948 on the unchanged control path. These are
independent trials, not paired between candidates. The split count is the only source
change from the R04 candidate. Warm diagnostic partial/reduction3.2/1.3µs, total4.5µs;
these diagnostics are not substituted for actual cold-L2 timings or proof of occupancy.

Full samples/profiles/environment provenance in results/remote-R08-router-split32.json
and CSV; offline comparison COMPARISON_R04_R08.md. This supports composing S32 with
R07's wide large-M path in a separate final check, preserving both controls. No
unverified composite score is predicted. Budget9/10 currently, R09 still pending.
