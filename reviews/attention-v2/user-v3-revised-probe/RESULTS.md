# Revised v3/v3b: correct, both performance-rejected

Two original-source, non-ranked official checks completed. No source edits were
made. The revised archive SHA256 is
`fa516f52a835b2bce45ae6b75740a54d3c2a9950cf00d858bcaf38eccaa1f2c1`.
Its only entrypoint change from the previous archive is the LOG2E constexpr fix.

| Variant | Job | Score | 95% interval |
|---|---|---:|---|
| v3, PREFETCH1 |2079ea1e49854f4c|0.515922845x|[0.508184003,0.523661688]|
| v3b, PREFETCH0 |2e1d37a7732449d7|0.567039508x|[0.558533915,0.575545100]|
| Selected champion |5090b83f1d63407b|1.233633059x|[1.215128563,1.252137555]|

Each new check passes **16 prerequisites and 12 timed shapes**, has12 profiles,
no flags, and the same evaluator identity as the saved champion. Both six-load
weighted scores were independently reconstructed exactly. The LOG2E repair works.

Full-path official calibrated times, microseconds:

| Full path | Saved champion | v3 | v3b |
|---|---:|---:|---:|
| Decode B16 |194.272|323.243|237.752|
| Decode B32 |304.032|614.308|492.654|
| Verify B16 |197.602|669.473|664.407|
| Verify B32 |380.381|1335.920|1334.222|

Sliding changes are separate; these routes also changed and are **not controls**:

| Sliding path | Saved champion | v3 | v3b |
|---|---:|---:|---:|
| Decode B1 |11.477|26.189|25.653|
| Decode B16 |13.921|28.447|26.426|
| Decode B32 |20.104|33.693|24.180|
| Verify B1 |16.988|39.373|37.996|
| Verify B16 |21.740|59.757|55.653|
| Verify B32 |27.468|73.975|73.026|

Each candidate/reference comparison is the official same-run paired fresh-input,
cold-L2 CUDA-graph benchmark. Champion comparisons above use historical receipts,
not a same-session champion A/B. After reference normalization all12 paths still
regress. V3b's four full regressions are +21.77%,+62.12%,+238.08%,+250.15%; its
six sliding regressions range from+20.78% to+166.48%. Complete rows are in
`summary.json`. No synthetic or profiler time is used as a scored latency.

**Keep the selected champion. No CTA-per-SM sweep:** the user's condition that
both variants pass and beat the champion is false. V3b's higher score than v3
does not establish a useful improvement over the champion. PREFETCH also selects
num_stages1 versus3, and these are separate jobs, so the comparison does not
isolate prefetch causality. No native spill, occupancy or overlap claim follows.
The source's shared-ticket simultaneous-graph limitation remains untested.

The bundled tolerance-relaxing local tester was not run. The reviewed static
global scanner reports both entrypoints OK; native correctness used the unchanged
official checks. Both jobs are terminal, no Modal job or ranked submission was
created, no credentials changed, and paid spending is0. Original sources/hashes
and raw-result hashes are retained; published result copies only make local
output/profile paths relative. Selected SHA256 remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
