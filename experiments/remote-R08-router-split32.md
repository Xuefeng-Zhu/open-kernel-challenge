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
