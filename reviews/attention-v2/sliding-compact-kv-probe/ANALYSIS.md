# Offline official-result analysis

`analyze_official.py` reads saved JSON only. It does not import candidate,
Torch or vendor modules, contact services, start processes or change selected
records. The candidate source is frozen to
`c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594`.
Selected job5090 raw/profile hashes are pinned to the reviewed saved evidence.

Example after the root has saved terminal files:

```
python -B reviews/attention-v2/sliding-compact-kv-probe/analyze_official.py \
  --job reviews/attention-v2/sliding-compact-kv-probe/official-check/official-result.json \
  --profile reviews/attention-v2/sliding-compact-kv-probe/official-check/artifacts/candidate.92fb22b141fc4d67.json \
  --request reviews/attention-v2/sliding-compact-kv-probe/official-check/request-receipt.json \
  --output reviews/attention-v2/sliding-compact-kv-probe/official-check/analysis.json
```

For an early failed run with no profile, omit `--profile`; failure reporting
does not require benchmark rows. Source/job request binding is reported even
for an early failure. A false binding is an invalid receipt. A failed original
prerequisite is labeled `original_prerequisite_failure`, retaining its case,
message and result error: a native exception is not automatically a numerical
oracle failure. Pending, duplicate, native/execution failure and incomplete
evidence remain separate states.

Performance analysis requires all16 unique prerequisite indices0..15 and all12
passing clean timing rows. Raw/profile job identity, shape sets, median/reference
times and paired ratios must agree; profile summary must also match the passing
raw result and its score. Six-load score reconstruction is checked explicitly.

Only sliding verify B1/B16/B32 are changed targets. Per shape,
`r=(Cnew/Rnew)/(Cselected/Rselected)`. For the prospective fixed-reference model,
set target times to `Cselected*r` and freeze the other nine candidate times and
all selected references. Apply the card's ≥3% modeled score gain, ≤3% target
normalized regression and ≤3% absolute unchanged-control drift gates.
Control changes are never counted as candidate benefit.

Compare the entire evaluator dictionary, including code, image and file hashes.
Retain all reported machine differences. GPU name, driver, SM/memory clocks and
power limit before/after are stable comparability fields; missing or differing
stable fields make comparison inconclusive. Node, temperature and instantaneous
draw are reported as context rather than interpreted as binary identity or
measured utilization. Changed evaluator or excessive control drift also makes
performance attribution inconclusive.

The bounded `OKC_COMPACT_FA3_METADATA` log marker may expose Torch/CUDA versions
and an operator registration schema. Identical repeated markers are counted;
inconsistent or oversize markers remain UNKNOWN. No input schema is interpreted
as resolved PackGQA/splits, internal QK/PV dtype, executed tile or a binary hash.

Passing the descriptive screen is `descriptive_gain_requires_confirmation`;
automatic promotion is always false. The official run pairs candidate with its
own SGLang reference, not selected3da. Historical normalized comparisons and
point scores are not a paired candidate-versus-selected significance test.

Thirteen CPU tests pass, including saved-score/zero-gain reproduction, the
card's published25% target-cut model, insufficient gain, single-target regression,
control drift, changed evaluator/driver, machine context, failed prerequisites
without timings, native failure/pending/duplicate, profile contradiction,
duplicate prerequisite indices, source binding and bounded metadata. A real CLI
same-baseline smoke reconstructed1.2336330588627955 and zero modeled gain,
classifying it as performance rejection. Neither tests nor smoke are new GPU or
official measurements.
