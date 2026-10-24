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
