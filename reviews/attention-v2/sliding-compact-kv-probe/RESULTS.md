# Compact sliding KV: correctness passed, no adoption

The one official nonranking check passed all **16 prerequisites and 12 timed
shapes**, establishing that this FP8 contiguous-cache route executes under the
original oracle. It did not demonstrate the declared benefit. Its score was
**1.207015236×**, 95% interval [1.188910008, 1.231680347], versus selected
fusion-plus-lookahead's historical **1.233633059×**. The selected candidate
remains unchanged; this compact candidate is not adopted or submitted as a
ranked entry.

The formal historical performance comparison is **inconclusive** because the
reported driver changed from 580.159.03 to 580.178.04. Independently, every
target's descriptive reference-normalized direction was worse and the
predeclared +3% modeled-score screen failed. These observations justify
declining production adoption; they do not establish exact causal slowdowns
or statistical superiority of one source on a matched driver/session.

## Exact experiment and admission

| Binding | Value |
|---|---|
| Official job | `92fb22b141fc4d67`, `mode=check`, terminal `done`, passed |
| Frozen source commit | `d7790c5483edc15d3e5c021692d38af11e5a9345` |
| Candidate SHA256 | `c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594` |
| Selected baseline SHA256 | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Original reference SHA256 | `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b` |
| Original numerical checks SHA256 | `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055` |
| Raw official response SHA256 | `9d6ad7180bc40478df2deed54f58a086d5db30544a614d6409651bfe3f9dcc0c` |
| Raw CLI profile JSON SHA256 | `705de818f748610d80e8695792740e7ddb9e0943f67662e9e88a231c267c6584` |

Only L8 sliding2048/RoPE500000/page1/TP8/D64 calls change, for any batch.
The unchanged preparation first appends live KV to the original cache and
produces Q. One fixed 32-token gather copies the exact union of at most
2047 historical plus eight new tokens into invocation-local contiguous FP8
scratch, through UINT8 views. Live lengths are produced on GPU. The same FA3
call then uses the scratch and no page table, retaining causal masking,
window (2047,0), automatic split/packing requests, scales and output norm/gate.
All other routes retain the original function bodies and dispatch.

The source, precision, exact per-query window mapping, shuffled-page byte
addressing, padding, slot0 and fixed-address CPU models passed 17 local tests
and independent review before the sole upload. An independently reviewed
offline analyzer passed 13 further tests. The separate CPU identity worker
also passed 13 tests; its remote image phase was not run. These local checks
are not GPU replay/isolation or hardware evidence.

## Complete-call results and prospective gates

Each new candidate measurement is paired with its own SGLang reference by the
official service. The comparison below additionally normalizes each job by its
own reference: `r=(C_new/R_new)/(C_selected/R_selected)`. It is a descriptive
cross-run comparison, not a paired candidate-versus-selected experiment.

| Changed sliding-verify path | Selected C, µs | New C, µs | New same-run reference, µs | Historical normalized latency change |
|---|---:|---:|---:|---:|
| B1 |16.988207|19.060847|23.136001|+11.5797%|
| B16 |21.740250|25.987806|29.647999|+15.6671%|
| B32 |27.467619|34.227761|34.720000|+23.2331%|

Applying only those three normalized ratios to the selected run while fixing
its nine controls and references produces modeled score **1.204023556**,
or **−2.400187%**. The prospective requirement was at least +3%, with no
target regression above 3%; both conditions fail. The largest absolute
normalized drift among nine unchanged rows was **1.789877%**, within its 3%
flag threshold. The raw aggregate point score changed −2.157677%, which must
not be confused with the target-only model or a statistical treatment effect.

The complete evaluator code/image/files dictionaries match. Both report H200,
Torch2.14.1+cu130, CUDA13.0 and matching reported clock/power-limit fields;
the driver differs in both before/after snapshots. The analyzer therefore
retains `inconclusive_comparison`, even though the benefit gates failed. No
threshold was relaxed and no additional check or split/tile sweep followed.

The official 16+12 pass does not establish targeted native empty-history,
slot0, metadata-ABA or independent-graph overlap coverage for this new route.
Those remain separate from CPU models and standard official graph replay.
No serving-concurrency claim follows.

## Stage evidence and mechanism limits

The rounded instrumented activity reports show:

| Path | Added gather, µs | FA3 forward, µs | Combine, µs | Schedule, µs |
|---|---:|---:|---:|---:|
| B1 sliding verify |1.4|7.6|3.0|1.8|
| B16 sliding verify |3.7|10.6|3.8|1.8|
| B32 sliding verify |7.9|14.4|4.4|1.9|

The historical selected forward activities were 7.8/10.7/14.5µs. The new
reports provide no visible large forward reduction that offsets the added
gather, but they are separate instrumented runs on different drivers. Rows
may include predecessor wait and overlap. They cannot be subtracted from
adjusted whole-call medians, treated as HBM bandwidth or CTA residency, or
used to prove TMA selection or a specific bottleneck. The native symbols are
still truncated in these official profiles.

The experiment changes representation and cache capacity, so automatic
specialization and scheduling may also change. It does not isolate page-table
latency, and the outcome does not prove all compact layouts or attention
architectures fail. It rejects adoption of this exact source under the
predeclared screen; another minor fusion is not justified merely to rescue it.

## Actual registration metadata recovered

One bounded import-time print, permitted by the saved TASK's diagnostic-print
allowance, reported the official runner's registered `sgl_kernel::fwd` schema.
It has **37 inputs and four tensor returns**. The registered inputs include
`scheduler_metadata`, `num_splits` and optional `pack_gqa` at positions31/32/33.
The last three returns are unnamed in this schema; their shapes and meaning
were not observed. The full string, versions and receipt hash are preserved
in `../fa3-next-evidence/dispatch-identity-cpu/OFFICIAL_REGISTRATION_METADATA.json`.

This resolves a registration/API evidence gap without loading live inputs or
patching modules. It does **not** expose resolved packing/splits, P dtype,
QK/PV instruction formats, executed tile or vendor-binary hash. The older
three complete Modal symbols remain historical 64×128×64 evidence, not proof
of current official M192/N160 or any inferred 24×/3× runtime waste.

## Access, cost and resource closure

At 22:14 UTC, independent read-only checks verified official access with
20 free checks remaining, while Modal explicitly rejected the existing token
as expired. The denied credential-creation operation was not retried. No
credential, account permission, billing limit or visibility change occurred.

Exactly one free official check was admitted at 22:29:45 UTC and started at
22:29:52 UTC. It finished at **22:35:01 UTC**, about 308.739 seconds after
start; terminal closure was read before result polishing. Actual post-admission
quota was **19/20 checks remaining, 0 ranked entries used**. The final quota
receipt is retained separately. No further check was used by this hypothesis.

**No Modal app was created**, so there is no app to stop or paid GPU invoice
for this experiment. Actual paid spending is $0. Current global Modal app
state could not be read because authentication failed; historical closure
receipts are not relabeled as a fresh global-state check.

The selected1.233633 candidate and exact A2-11 rollback source are unchanged.
Reviewed preparation was pushed to the existing private branch and its remote
commit verified. Raw receipts, CLI artifacts, analysis and independent result
review are preserved beside this report; source-preparation manifests remain
unchanged as records of the pre-admission freeze.
