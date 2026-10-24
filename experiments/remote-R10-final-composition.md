# R10 — final controlled composition, tenth and last allowance

Source `candidates/router_pooler_hybrid.py`, commit
`5f491a34d52449f5d0c5c7985b4a0d4ecd17217d`, SHA256
`891b2063edde9ef9c508c2983e6cf10804bf5066fc036d63912c47ddeb8fc9ee`.
Configuration/hypothesis/provenance and local gates: experiment014. Relative to R07,
compose only independently measured split32 low-M router and m1 SIMT pooler; wide
large-M router stays R02. All native kernel bodies match their checked controls.
No new tuning, lower precision or regressing poolerSplit8/QKV path is included.

```sh
python3 remote_check.py send R10-final-composition candidates/router_pooler_hybrid.py
python3 remote_check.py poll R10-final-composition
python3 summarize_check.py R10-final-composition
```

Sole accepted check job `02e099c3724e4a05` at2026-10-24T16:28:45Z. Admission10/10;
**no remaining new checks**. Existing nine admitted jobs terminal/passing. No uncertain
request, paid resource, scored/public/final entry. Monitor this exact ID to terminal;
never launch a replacement. Full composition GPU correctness/performance pending.
Local39/39 full-size CPU cases +57/57 combined tests pass, which is not native GPU proof.
