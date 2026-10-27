# Independent Attention baseline/methodology matrix review

**PASS for the source, numeric matrix, coverage and causal qualifications.**
No new algorithm or GPU launch is admitted. This review used local saved
sources/results and CPU-only reconstruction; it made no service, browser,
network, compiler, credential or Git request. Only this review was written.

## Exact selected baseline and twelve paths

Selected source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The reviewed source guard at585–602 and custom host at281–330 establish four
custom producer/merge paths and eight fused-preparation/FA3/gate paths:

| Load | Sliding: adjusted us / route | Full: adjusted us / route |
|---|---|---|
| D1 |11.477114 / custom S16,32 producer CTAs|27.031303 / fused FA3|
| D16 |13.921137 / custom S9,288 CTAs|194.271909 / fused FA3|
| D32 |20.103604 / custom S5,320 CTAs|304.031893 / custom S11,704 CTAs|
| V1 |16.988207 / fused FA3|29.568836 / fused FA3|
| V16 |21.740250 / fused FA3|197.602134 / fused FA3|
| V32 |27.467619 / fused FA3|380.380746 / fused FA3|

Times match official check5090 exactly before rounding. These are workload
families, not competitor architectures. All scored custom paths use M16/N64,
eight valid query rows and independent per-head merge CTAs16/256/512/512.
S11 is capacity-derived from agentic maximum86000 plus one new token and
8192-token chunks, not11 populated historical chunks for every request.
Sliding covers2047 history keys plus one new key. Last-capacity-split new-KV
ownership remains required even if its history is empty.

All eight FA3 routes already have fused normalization/optional RoPE/FP8
conversion/live cache append. Full decode legitimately receives the contract's
scheduler metadata; verify leaves it None and prepares per call. Every saved
FA3 profile includes combine; six verify profiles also include scheduler
preparation. The handoff clarification preserves supplied decode metadata as
a lawful input while keeping new/internal/live planning within the proper call.

## Independent numeric reproduction

Executed the actual matrix builder with its sole `Path.write_text` intercepted
in memory. Its complete generated JSON equals the saved matrix; no output file
was changed. Independently checked12 unique family/layer pairs, all six
weighted-load sums, shares summing to one,10%-path sensitivity factors and
the reconstructed score1.2336330588627955. Each load uses46 sliding/16 full
layers; the six load speedups are geometrically averaged. Own-load fractions
are not global score weights. Fixed-reference improvements/zero-time limits
remain conditional arithmetic, not hardware ceilings or predictions.

The source/profile hashes in the builder match actual bytes. The profile JSON
SHA256 is`3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300`.
Instrumented activity rows are correctly kept separate from calibrated
whole-call values, including the D16-full297.3us activity versus194.27us
benchmark discrepancy. No difference is treated as recoverable latency or
proof of a launch, memory, residency or stall bottleneck.

## Strongest exact-source native capability evidence

The matrix now correctly uses the **selected3da control**, not merely an old
donor, for compiled D32-full evidence:

- `../alpha-one-common-pv-probe/native/result.json` SHA256
  `de0efb07a8e46e6e261d16db249668619e7d7083c17c599478db5f2aff861e90`.
  Its `source_sha256.baseline.py` equals selected3da; producer options are
  M16/N64/S11/lookaheadTrue, four warps/three stages.
- Exact baseline producer PTX SHA256
  `692b99394b0f534bbdf1938f39460f7da37b9474ba17d019d4b8461e21967764`.
  It contains F16-QK/BF16-PV `mma.sync`, FP32 accumulation, and existing
  `cp.async` global/shared copies plus commit/wait groups; no WGMMA or native
  FP8 MMA is present. E4M3-encoded data is not a native-FP8-instruction claim.
- That producer reports96 registers/20,480 shared bytes/zero spills; its
  unchanged merge reports27/1,024/zero. These are source/geometry-bound
  compilation observations, not the official5090 binary, achieved residency
  or proof that another asynchronous schedule would help.

No exact-selected native resource/ISA receipt for the three custom sliding
specializations was identified in this bounded review. The older executed
FA3 full-B32-verify symbol establishes staged SM90 dynamic persistence for that
historical specialization, not all current eight FA3 binaries or runtime split
maps. Missing headers/counters do not prove a missing mechanism. The two
counter timeouts remain timeouts, not permission denials.

## Proposal reach and retained negative evidence

The parked B1 SIMT source changes only D1-sliding: M8 scalar/reduction mapping
and valid-row partial writes, with unchanged S16/N64/merge. That is a genuinely
different execution-family question; GQA reuse, window clipping, FP8 encoding,
preparation/cache fusion and gate fusion already exist. Its distributed TL
arrays, gather/reduction lowering and payload reuse remain uncompiled. The
15% single-path scenario gives about1.44% overall uplift and cannot explain
the public-leader gap. No supported replacement is manufactured for the other
eleven paths; matrix unknowns are information gaps, not implemented candidates.

The V-stage result is correctly a local mechanism pass and production NO-GO:
same nearby4096/padded-table complete calls show5.04%/3.89% prototype-pair gains
but the faster prototype remains about3.82x selected FA3. It does not establish
80K competitiveness or broad verify coverage. Alpha-one, invalid strict-serial
control and sliding-unsplit failures retain their exact tested scopes; neither
overgeneralized rejection nor incremental rescue is proposed.

Later exact3da DP validation already passed page64/S2 and page1/S3/BS4 cases
with15 original-oracle calls; they supersede the older selection record's
historical pending-gap text. Internal scratch pointers, actual simultaneous
hardware overlap and universal serving coverage remain unobserved.

Reviewed `B1_WINDOW_BUDGET.md` and `B1_SINGLE_FALSIFIER.md` as historical local
notes. Their costs are source/ideal-model quantities, not measured bandwidth
or predicted gains. The latter's feature16xhead8 and manually shared logits/P
sketch is **superseded** by the implemented source card's distributed TL arrays;
it must not be described as the candidate's physical layout. The current
methodology matrix makes that distinction. The notes do not launch a subgroup
sweep or revive rejected query hoisting/serial merge.

The handoff asks for methodology/error classification and ordered information
gates, rather than another unsupported kernel list. It accurately distinguishes
the official reference from the public leader, whose private code/per-shape
times are unknown. No external message was sent by this reviewer.

## Reviewed artifact hashes

| Artifact | SHA256 |
|---|---|
| `BASELINE_METHODS_AUDIT.md`, final closure note included | `dfd66b8e1027daef3e8978465c415a92942894bb5d25852f850971319dca91c4` |
| `METHODS_HANDOFF_ATTENTION.md`, scheduler clarification included | `0ba0bc9c594576338c4c58ba6ff8c40c0cffda24adba379064b908fb9e530e5c` |
| `baseline_matrix.json` | `71cef53426c36790a7cd8a109452eba33248e6fdb70ab218ca554f89e9741fa2` |
| `build_baseline_matrix.py` | `08784483ca28b7e4919b61a60f4a2dfb643bd17c64e818a369461ccd76795af2` |
| Historical `B1_WINDOW_BUDGET.md` | `b43e62de73051aad6e8b1c5b0bc005a7dc78c5fc779536eaeb889423de65063e` |
| Historical `B1_SINGLE_FALSIFIER.md` | `55dabdab2dc3c7c39d5e6f140839bb20941472de52b0bbb514d20d0c082d8ca1` |
| `METHODS_RESOURCE_CLOSURE.json` | `bc17742f37b537487e4630cf6aa9b4d9eeacafc7f2788db9ec8f0eeb1b623fad` |
| `../b1-simt-probe/STATUS.md` | `c40960561c8e85c373e2eaffbf0448e831b3ef7fd088c211cab64f414bfea8ca` |
| `../b1-simt-probe/PREPARATION_MANIFEST.json` | `15f079a5932122b41e6fa8256b1e759ce8fac87a97f16b028f332dc62b26c4a0` |

The final closure note matches the parent's sanitized01:23:50UTC receipt:
nine returned main apps are stopped/tasks0. This bounds the list response,
not every deleted historical app or a fresh invoice. That read-only service
action was performed by the parent, not this local reviewer. The parked B1
status/manifest retain the exact source/card/test hashes,27 reported local
checks and explicit absence of upload/admission/native results or uncertain
remote action. All listed preparation-file hashes were checked locally.
All new launches remain held; no selected source or admission status changed.
