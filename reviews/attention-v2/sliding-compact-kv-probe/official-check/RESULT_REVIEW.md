# Independent official-result review: PASS, engineering NO-GO

Reviewer `/root/compact_result_analysis` read the saved raw job, complete CLI
profile JSON/text/log, request and quota receipts, and reproduced the analysis
without any service call. This review verifies the evidence and stop decision;
it is not a performance pass or a new benchmark.

## Terminal execution and source binding

Job `92fb22b141fc4d67` is terminal `done`, mode `check`, board
`muse_attention_decode`, with `duplicate_of=null`. Its recorded service times
are created22:29:45.262568, started22:29:52.436116 and finished22:35:01.175328
UTC on2026-10-26. It passed, returncode0, errornull, static checks clean and
no benchmark flags. All16 prerequisite indices0..15 are unique and passing;
all12 scored shapes are present, positive, unflagged and passing.

The request records the reviewed source commit
`d7790c5483edc15d3e5c021692d38af11e5a9345` and candidate SHA256
`c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594`.
The current candidate bytes independently rehash to that exact SHA and the
request/accept/raw/profile job IDs agree. Git-object and remote verification
belong to the root's separate export checks; this task performed no Git action.

The complete official pass establishes native acceptance and original-oracle
correctness for these tested calls, including the changed contiguous-FP8 route.
It does not establish arbitrary serving robustness, native slot0 coverage,
same-address mutation or simultaneous graph isolation. Local source/address
models are distinct from those remaining native tests. No precision tolerance
was changed by this review or experiment.

## Independent arithmetic and prospective gates

Using a separate calculation directly from raw benchmark rows, the six-load
weighted geometric mean is **1.207015236205559**, matching the official score.
Its reported95% interval is [1.1889100076624757,1.2316803472514632]. Selected
job5090 remains1.2336330588627955; the new aggregate point is2.1576775% lower.
Those are separate runs, not a paired candidate-versus-selected comparison,
and marginal intervals do not establish a difference test.

For each path, this review separately computes
`r=(Cnew/Rnew)/(Cselected/Rselected)`. All values below are calibrated whole-call
times or descriptive cross-run normalization; they are not profiler durations.

| Changed target | Selected C, µs | New C, µs | New same-run R, µs | Normalized latency change |
|---|---:|---:|---:|---:|
| Sliding verify B1 |16.988207|19.060847|23.136001|+11.579706%|
| Sliding verify B16 |21.740250|25.987806|29.647999|+15.667128%|
| Sliding verify B32 |27.467619|34.227761|34.720000|+23.233128%|

Changing only those three selected times to `Cselected*r`, while freezing all
nine controls and all selected references, gives modeled score
**1.20402355591918**, or **−2.400187213766014%** versus selected. Independent
arithmetic agrees with saved `analysis.json`. The worst absolute normalized
drift among the nine unchanged paths is1.7898770604940384%, below the declared3%
control flag. All three changed paths have unfavorable descriptive ratios;
both the ≥3% modeled gain and ≤3% target-regression numeric screens fail.

The entire evaluator dictionary is identical, including code SHA, all file
hashes and judge-image digest. Reported GPU is H200 with matching name, SM
clocks1980, memory clock3201 and power limit700. However, the stable reported
driver changed **580.159.03 →580.178.04** in both before/after records.
Node and instantaneous draw also differ and are retained as context.

Therefore the analyzer correctly returns **`inconclusive_comparison`**, not a
precisely attributable performance regression: the declared stable-environment
gate is false. Reference normalization cannot prove that driver effects cancel.
Nevertheless this result supplies no positive evidence for the new route and
fails the prospective descriptive gain screen. **Engineering NO-GO** is the
appropriate stop decision; it does not claim that compaction caused exactly
the displayed11.58/15.67/23.23% changes. No further check, sweep or promotion is
justified by this result. Selected3da/1.233633 and A2-11 rollback remain preserved.

The saved analyzer was independently rerun into `/tmp`; its complete output
object exactly matches saved `analysis.json`, in addition to the separate
arithmetic above. Its13 CPU tests and prior independent analyzer review remain
local evidence, separate from this new official execution.

## Actual API metadata and activity limits

The actual official log contains one complete bounded metadata marker:
Torch2.14.1+cu130, CUDA13.0 and registered `sgl_kernel::fwd` schema. Direct schema
parsing confirms37 arguments; positions31/32/33 are `Tensor? scheduler_metadata`,
`int num_splits`, `bool? pack_gqa`. Returns are four unnamed tensors:
`(Tensor(a!), Tensor, Tensor, Tensor)`.

This is new observed registration evidence in the official runtime. It is
not a resolved packing flag, per-request split count or named auxiliary-output
layout. Executed tile, vendor binary hash and internal QK/PV instruction dtype
remain UNKNOWN. The profile symbols are truncated; no M192/N160 or TMA-path
claim follows from this result. BF16 output does not prove BF16 PV.

The three instrumented profile reports show one added gather activity of
1.4/3.7/7.9µs for B1/B16/B32, and FA3 forward activity7.6/10.6/14.4µs. Schedule
preparation and combine remain visible. These are rounded diagnostic activities,
some explicitly including predecessor wait/overlap. They cannot be subtracted
from adjusted whole-call medians or interpreted as measured HBM traffic,
bandwidth, achieved occupancy or causal slowdown attribution. They establish
that the added gather executed, not why the whole-call ratios changed.

## Resource and quota evidence

Exactly one new nonranking official check was admitted: quota changed from
0/20 to1/20 used,19 left; ranked usage remains0/10. The request records paid
cost0 and no formal entry. There is no Modal app for this experiment. This
review started/stopped no service and makes no fresh global Modal-app-state
claim. It confirms the official job's terminal state from its raw response;
the parent separately handles serialization and export closure.

## Bound artifact hashes

Paths below are relative to this `official-check/` directory unless stated.
The raw profile hashes and byte sizes match `profile-artifacts.json`; bytes
were read without reformatting them. These hashes cover the reviewed result,
not future narrative/report edits.

| Artifact | SHA256 |
|---|---|
| `official-result.json` | `9d6ad7180bc40478df2deed54f58a086d5db30544a614d6409651bfe3f9dcc0c` |
| `analysis.json` | `3bcc2fd6ec3f0792237694f400f5bb004093840322652ba99bb6fd296d266460` |
| `artifacts/candidate.92fb22b141fc4d67.json` | `705de818f748610d80e8695792740e7ddb9e0943f67662e9e88a231c267c6584` |
| `artifacts/candidate.92fb22b141fc4d67.txt` | `8370fee76c93090d601deb9e4d8842fa68f52791df528f7b84d56326ec608b48` |
| `artifacts/candidate.92fb22b141fc4d67.log` | `f39cda8c0b8a605fa8f51bfb037b83fb412703a109cde04b2ec72b2e6607786f` |
| `profile-artifacts.json` | `cfb2dabd58956c5ea2de529eda7c8fbd9f19390c2548549d75970433fb3d2f40` |
| `request-receipt.json` | `c727752b43d84b87b445ef55da5a7e7a5f1a4e6c9a35cb698ddddf67c4d48975` |
| `accept.json` | `61bdb7fe957c034efa4696e435644d4c278c35885df897ec348f1773873b30df` |
| `quota-before.json` | `ce85f492a0dbd4c2a4844e3fa865e68f4f3707bd3d2bdcd42a3e58104ea00730` |
| `quota-after-admission.json` | `e1b57a33842881d138e75a230ec189c585ad533bac5170acce76ea7b4a04568d` |
| Selected5090 raw result | `a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb` |
| Selected5090 profile JSON | `3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300` |
| `../candidate.py` | `c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594` |
| `../baseline.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| `../EXPERIMENT_CARD.md` | `3dfe8928c3449ff4717f8d88ee21d07a6a96467c4c504f62dfab2a29ef72fc28` |
| `../SOURCE_MANIFEST.json` | `34e5e6fd9f293e429d17855edcbd9f4feaa2b665b66ccfb913efa6329073c759` |
| `../REVIEW.md` | `a0d8416b20f9f1e8fe1e74dd3aecc384e7747762294102d9d7a7545cb7990396` |
| `../analyze_official.py` | `4a2b33cdcd0973b5dc3345984d10f8c4b33c6b296e89537ca7061532abff3683` |
| `../test_analysis.py` | `8589c1c4bddc39e60b8fa44701f9482284bc50cfec0942bf8c09ea222518cd66` |

No raw trace, GPU counter result, source revision or binary identity was inferred
from missing evidence. No service, credentials, candidate, card, selected record
or Git object was changed by this result-review task.
