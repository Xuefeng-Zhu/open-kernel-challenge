# Attention baseline and methodology audit

Prepared 2026-10-26 UTC from local sources and saved own results. New GPU
launches are held for the methodology handoff. **B1 SIMT has not been uploaded,
admitted, compiled or executed.** Its 21 CPU source/geometry tests and six
compile-harness tests pass. These are local preparation evidence only.

The working source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Its one official nonranking check `5090b83f1d63407b` passed 16 prerequisites
and 12 timed shapes: score **1.233633059**, 95% interval
[1.215128563, 1.252137555]. A2-11 remains the unchanged rollback.

## Actual baseline, before a proposal

All scored cases have D64, TP8 = 16 Q / 2 KV heads, page1, nominal ctx80K.
D means q_len1 and V means q_len8. Times below are the saved official
**adjusted whole-call** measurements. Shares are within each of six weighted
loads, not global score weights. Each load has equal weight in the geometric
mean. [baseline_matrix.json](baseline_matrix.json) preserves all inputs,
source-derived grids, separate activity reports and local score sensitivities.

| Family | Actual selected mechanism | Whole call, µs; own load share | Separate activity evidence, µs | Proposed missing capability and coverage | Cheap distinguishing test; current status |
|---|---|---|---|---|---|
| D1 sliding | Custom fused norm/RoPE/FP8/new-KV attention; GQA8; M16 with 8 valid rows; S16 gives 32 producer CTAs; 16 per-head merge CTAs include output norm/gate | 11.477; 54.97% | Producer 7.8, merge 1.7 | Source has padded arithmetic/state; a G8 SIMT family could remove it but loses tensor throughput. Only this one scored path is implemented. No broad gain established. | Compile exact same S16/N64/merge control and SIMT source; inspect real head state, reuse, reductions, resources before original oracle and actual-best whole-call test. **Prepared, unexecuted, held.** |
| D16 sliding | Same custom fusion/GQA/window pruning; S9 gives 288 producer CTAs, 256 merge CTAs | 13.921; 17.08% | Producer 9.0, merge 1.8 | Same source padding exists. B1's 32-CTA parallelism issue does not transfer to this grid. No missing capability established. | Do not extrapolate B1 result. Any later extension needs its own same-geometry comparison and actual selected baseline. No candidate. |
| D32 sliding | Same custom fusion; S5 gives 320 producer CTAs, 512 merge CTAs | 20.104; 15.97% | Producer 14.8, merge 2.2 | Same boundary as D16; no evidence of missing GQA, window pruning or preparation fusion | Source/ISA and complete-call accounting first; no split sweep or candidate. |
| D1 full | Fused Q/K normalization, Q/K/V FP8 conversion and live append; FA3 with supplied scheduler metadata; separate norm/gate | 27.031; 45.03% | FA3 forward 14.7, combine 6.8, preparation 1.2, gate 1.1 | Resolved B1 specialization/split/partial sizes and dependency costs remain unknown. There is no missing scheduler-preparation launch here. | Recover exact executed specialization and auxiliary shapes on same live fixture; separate dependency wait from work. No implemented replacement. |
| D16 full | Same fused preparation + FA3; supplied scheduler metadata; separate norm/gate | 194.272; 82.92% | Forward 297.3, combine 3.7, preparation 1.6, gate 1.6 | Dominant region is attention in this instrumented run; exact current page1 mainloop/ISA and reason for cost remain unknown | Explain the large activity-versus-benchmark discrepancy before treating it as recoverable loss. No demonstrated missing overlap or persistence. |
| D32 full | Custom fused producer, FP8-encoded Q/K, FP32 softmax/partials, BF16 PV; fixed 8192-capacity chunks, S11 / 704 CTAs; page-address lookahead; 512 merge CTAs | 304.032; 84.03% | Producer 297.2, merge 2.7 | A different mainloop might matter, but no current missing overlap/reuse capability is demonstrated | Existing donor codegen and composition results are the starting control; require actual-best full call and same math. Rejected BF16/WGMMA or alpha-one variants do not settle all architectures. No new replacement. |
| V1 sliding | Fused preparation + causal/windowed FA3; per-call schedule preparation; separate norm/gate | 16.988; 62.29% | Forward 7.8, combine 4.0, schedule 1.8, preparation 1.3, gate 1.2 | Small-work family may have decomposition costs; current exact split/partial geometry unknown. Already fused prologue. | Resolve current specialization/task shapes. Removing combine is not inherently beneficial; B16/B32 unsplit already regressed. No B1 unsplit result. |
| V16 sliding | Same fused preparation + causal/windowed FA3; per-call schedule; separate norm/gate | 21.740; 24.03% | Forward 10.7, combine 4.2, schedule 1.8, preparation 1.9, gate 1.4 | Launch/combine work is visible, but its critical-path cost and useful split parallelism are unresolved | Previous num_splits1 removed combine and regressed 31.49%; preserve that negative result. No repeat or new missing capability. |
| V32 sliding | Same fused preparation + causal/windowed FA3; per-call schedule; separate norm/gate | 27.468; 17.19% | Forward 14.5, combine 4.1, schedule 1.7, preparation 2.3, gate 1.5 | Same unresolved tradeoff | Previous num_splits1 regressed 7.86%; no repeat or new candidate. |
| V1 full | Fused preparation + causal full FA3; per-call schedule; separate norm/gate | 29.569; 37.71% | Forward 14.9, combine 7.4, schedule 2.7, preparation 1.2, gate 1.2 | B1 decomposition is incompletely resolved; gate fusion alone has small visible scope | Exact specialization/partials/dependencies first; live task maps must remain inside the call. No cache shortcut or new candidate. |
| V16 full | Fused preparation + causal full FA3; per-call schedule; separate norm/gate | 197.602; 75.97% | Forward 193.3, combine 4.2, schedule 3.1, preparation 2.0, gate 1.7 | No demonstrated missing persistence/staging. Attention dominates this instrumented report | Current specialization and page1 dataflow must be compared before claiming a missing mechanism. Rejected BF16 custom was slower. |
| V32 full | Same fused preparation + causal full FA3 | 380.381; 82.81% | Forward 372.2, combine 4.4, schedule 3.4, preparation 2.4, gate 1.8 | Same unknown. An older executed specialization already proves dynamic persistence and staged SM90 code | Old trace is a provenance-qualified reference, not this run's binary. A Gluon replacement failed actual-best timing by about 3.82× on nearby ctx4096. No production candidate. |

**Activity caveat:** the official reports are rounded, instrumented, sometimes
overlapping and symbol-truncated. The report explicitly says durations can
include predecessor wait. Never subtract their rows from adjusted whole-call
times, identify launch gaps by their difference, or call them achieved
occupancy, HBM bandwidth, tensor utilization or stall measurements.

Custom Q/K are E4M3 encoded with strict source accumulation, but that does not
prove native FP8 tensor instructions. The exact selected 3da control in the saved
common-PV compile probe has M16/N64/S11/lookaheadTrue codegen: mma.sync F16 QK /
BF16 PV, no WGMMA, 96 registers and 20,480 B shared; merge has 27 registers and
1,024 B shared, with zero reported spills. Its PTX also contains cp.async
shared/global copies and commit/wait groups. Adding asynchronous staging is
therefore not a demonstrated missing capability. These are source/geometry-bound
compile observations, not the official 5090 binary or achieved residency. The
three sliding custom specializations do not have identified exact-selected
resource receipts in this audit. Lookahead also changed generated buffering;
its timing cannot isolate address latency alone.

FA3 already has dynamic persistence and staged/async mechanisms: the older
executed full-B32-verify symbol names VarlenDynamicPersistentTileScheduler,
SM90 and tile64×128×64. Current FA3 split counts, task maps and all current
specializations are not recovered. Do not infer a missing feature from missing
source or claim the old trace identifies the current bottleneck.

## Scale and what the experiments actually established

The saved public snapshot at 2026-10-25 22:57 UTC is 2.9097, versus our
nonranking 1.233633059; no rank or private competitor per-shape data is known.
The ratio is 2.358643, equivalent to a uniform 57.6027% whole-load cut.
Halving all six full paths models only 1.8891. Replacing them with optimistic
TASK throughput estimates models 2.3585. These are fixed-reference scenarios,
not measured hardware bounds or forecasts. A one-path B1 sliding 15% cut
models only about 1.44% score improvement; even zero time there has a
conditional score factor of only 1.1422. It cannot be the complete winning plan.

| Evidence | Valid conclusion | Invalid extension to avoid |
|---|---|---|
| Fusion + lookahead: paired target reduction 4.8226% vs fusion; official score 1.23363; prior A2-11 1.18768 | Modest improvement over a relevant validated baseline; adopted working candidate | Award-level architecture, precise aggregate significance or isolated page-load causation |
| Two NCU counter sessions timed out; Torch/CUPTI activity worked | Hardware counter measurements were not obtained by those sessions | Counter permission denial, measured stall/HBM bottleneck or proof of no hardware headroom |
| Strict alpha-one first form exceeded prospective resource gate | Rejected before performance on the declared resource criterion | Measured slower, or proof every conditional form fails |
| Common-PV alpha-one compiled within gate, then had no paired target gain | That exact branch form was not useful under the tested conditions | Universal rejection of conditional rescaling or later changed-precision variants |
| Original purported serial control had K loads hoisted before wait | Claimed serial-versus-overlap contrast was invalid and was rejected | Passing the original control gate by renaming the claim after seeing speed |
| Revised V conversion/publication scheduling: 30 oracle calls pass; paired prototype improvement 5.04% / 3.89%; faster prototype remains 3.82× actual selected FA3 | Local generated scheduling effect, production replacement rejected | Progress against current best, whole-suite score or an architecture bottleneck diagnosis |
| B1 G8 SIMT: 21 CPU source/geometry + 6 harness tests pass; no native run | Reviewable narrowly scoped source and a cheap unexecuted falsifier | Resource/codegen success, numerical pass, performance gain or broad coverage |

## Method changes for the next research decision

1. Establish actual selected route, exact environment identity, available legal
   APIs and existing mechanisms before declaring a missing capability. Our
   Projection-only local static checker was not an official Attention validator.
   Official indexed rules permit load_inline/native CUDA and image libraries;
   a blanket compiler ban was an overrestriction. Exact current full-body
   exception/checker acceptance is still a separate unknown.
2. Rank questions by weighted-score coverage and the uncertainty they resolve.
   Require a numerical optimistic ceiling before implementation, explicitly
   labeling a one-family feasibility diagnostic as such. A working mechanism
   on a slower engine is not yet a competitive candidate.
3. Fix a same-work, same-geometry control and a current-best whole-call arm
   before execution. Confirm the intended native distinction first; compiler
   hoisting can falsify the control. Do not slow the control to create a win.
4. Keep codegen/resources, numerical/replay correctness, uninstrumented paired
   whole-call performance and hardware diagnosis as separate gates. Passing
   one does not establish the next. Include all live preparation, KV updates,
   scheduling and merge in the timed call.
5. Use one small representative screen, then confirmation with fresh fixtures
   before full official checks. Preserve reverse-order pairs, cold L2,
   unchanged-path controls and a prospective acceptance/rejection criterion.
   Explain short/padded geometry limits rather than project them to 80K.
6. Stop a failed hypothesis at its declared gate. A materially different design
   needs its own mechanism/evidence; do not reopen a failed branch by relaxing
   resource thresholds, precision or posthoc performance bars.

Original-oracle correctness remains a hard gate, not an optimization objective
to exploit. All keys, scale/quantization boundaries, last-split new-KV ownership,
cache updates and live replay metadata remain part of the contract. Later exact
3da DP robustness validation already covered page64 S2 and page1 S3/BS4 with
15 oracle calls; these two cases are not still outstanding. Actual hardware
overlap and arbitrary serving-state robustness remain unproven.

## Frozen preparation and handoff status

The parked B1 candidate is SHA256
`f5161be2e91b05fceb01e28564f96eecff471244704d036fa4cc7950dbe0ef34`;
compile worker is
`e75d54a10342fe0d982a85d1db68a21422dc63714abf4320bd27ce14956797f8`.
It changes only B1/L1/sliding2048/TP8/D64/page1-or64, preserving S16/N64,
four warps/three stages, FP8 encoding, FP32 softmax/partials, BF16 P/V,
unchanged merge and live cache semantics. Prospective native gates are
192 registers, zero spill/local traffic, ≤64 KiB shared and at least two legal
CTAs/SM, plus actual eight-head state/no3D/no-eightfold-KV-reload review.
No source-level proof substitutes for native dataflow inspection.

No B1 app exists, no quota was consumed and no B1 cost was incurred. Latest
completed attention runtime app `ap-Hy5O3TPhVTVKu0XyfsgJWT` was independently
confirmed stopped/tasks0 at 00:40:48 UTC. Its delayed workspace usage delta
was $0.05258797, actual paid $0; this is not an exact per-app invoice. Its
[reviewed result](../gluon-vstage-probe/RESULTS.md) is preserved at private
commit `350ae96bbde5287078f293b313326c31785aa235`. A later read-only supported Modal CLI closure check, recorded in
[METHODS_RESOURCE_CLOSURE.json](METHODS_RESOURCE_CLOSURE.json), returned nine
main apps, all stopped/tasks0, including this runtime, both earlier attention
compile apps and Projection P059. No B1 upload, admission, native compile or
execution occurred. That harmless closure read was the only service action
for this audit; it is not a new billing invoice or a complete historical list.

Primary local evidence: [official TASK](../official/TASK.md),
[current result](../fusion-lookahead-probe/RESULTS.md),
[B1 history](B1_ROUTE_AUDIT.md), [B1 budget](B1_WINDOW_BUDGET.md),
[saved specialization](LOCAL_SPECIALIZATION_AUDIT.md),
[gap arithmetic](GAP_MODEL.md), [compilation rules audit](RULES_COMPILATION_AUDIT.md),
[V-stage rejection](../gluon-vstage-probe/RESULTS.md),
[B1 source card](../b1-simt-probe/EXPERIMENT_CARD.md).
Run `python3 reviews/attention-v2/fa3-next-evidence/build_baseline_matrix.py`
to reproduce the 12-row numeric matrix without importing candidate code or
starting a service.
