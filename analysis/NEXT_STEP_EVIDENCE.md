# Offline next-step evidence analysis

No GPU/check/install/credential/push action. Candidate code and budget bytes unchanged.
Input receipts: attention e0b3cd9474084069; projection best48b1357016134e52 vs6799753227ed463e.
All source hashes and exact pinned Git input commit are in next_step_evidence.json.

## Attention: all12shapes (microseconds)

C=our candidate, S=same-receipt SGLang. Raw is unadjusted event median;
official adjusted C/S and paired S/C determine score. Raw medians must not
be averaged or divided to substitute for official paired/calibrated ratios.

| Load | Layer | Raw C / S | Adjusted C / S | Paired S/C | w×(C−S) µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 17.920 / 24.720 | 13.841 / 20.656 | 1.492338 | -313.473 |
| decode 1 | full | 45.424 / 35.744 | 41.167 / 31.552 | 0.766443 | +153.837 |
| decode 16 | sliding | 19.920 / 30.832 | 15.430 / 26.576 | 1.722392 | -512.730 |
| decode 16 | full | 259.744 / 204.608 | 256.005 / 200.288 | 0.782360 | +891.472 |
| decode 32 | sliding | 24.336 / 35.152 | 19.760 / 30.864 | 1.561945 | -510.785 |
| decode 32 | full | 373.120 / 389.536 | 370.063 / 385.440 | 1.041552 | -246.028 |
| verify 1 | sliding | 47.680 / 26.704 | 43.976 / 22.896 | 0.520642 | +969.701 |
| verify 1 | full | 80.512 / 40.208 | 76.491 / 36.048 | 0.471273 | +647.083 |
| verify 16 | sliding | 41.056 / 33.280 | 36.993 / 29.152 | 0.788037 | +360.694 |
| verify 16 | full | 209.280 / 208.848 | 204.304 / 204.272 | 0.999842 | +0.515 |
| verify 32 | sliding | 49.184 / 38.432 | 45.256 / 34.528 | 0.762954 | +493.472 |
| verify 32 | full | 393.872 / 393.888 | 389.661 / 389.856 | 1.000501 | -3.124 |

Negative gap means we win. Sliding weight46, full weight16; all rows are measured.

## Six loads and exact score arithmetic

`C_load=46*C_sliding+16*C_full; S_load=46*S_sliding+16*S_full`
`G=exp(mean(log(S_load/C_load)))=0.8919466283178157`

| Load | Candidate load µs | SGLang load µs | Signed gap µs | S/C |
|---|---:|---:|---:|---:|
| decode 1 | 1295.372 | 1455.008 | -159.636 | 1.123236 |
| decode 16 | 4805.846 | 4427.104 | +378.742 | 0.921191 |
| decode 32 | 6829.970 | 7586.784 | -756.813 | 1.110808 |
| verify 1 | 3246.768 | 1629.984 | +1616.784 | 0.502033 |
| verify 16 | 4970.554 | 4609.344 | +361.210 | 0.927330 |
| verify 32 | 8316.332 | 7825.984 | +490.348 | 0.941038 |

Largest load deficit: verify1,1616.784µs. Its sliding component969.701µs
is the largest single-path deficit; full verify1 adds647.083µs. Full decode16
adds891.472µs, but lives in a larger load. Equal-load geometric scoring means
absolute gap alone is insufficient: matching one path gives relative score
opportunities6.091% (verify1 sliding),3.774% (verify1 full),3.479% (decode16 full).

## One-target scenarios, NOT achieved speedups

`verify1 current =46*43.976461+16*76.490695=3246.768337µs`
`new_score=G*(3246.768337/new_verify1_load)^(1/6)`

| Assumption for verify1 sliding only | Target µs | New load µs | Modeled score | Relative change |
|---|---:|---:|---:|---:|
| target_latency_minus20pct | 35.181169 | 2842.184895 | 0.911952106 | +2.2429% |
| target_matches_same_receipt_sglang | 22.896000 | 2277.067134 | 0.946276626 | +6.0912% |
| unattainable_zero_target_time_ceiling | 0.000000 | 1223.851125 | 1.049444123 | +17.6577% |

All other11shape times and all SGLang times are frozen. Matching SGLang is a
conditional scenario, not a hard speed limit or performance forecast. The zero-
time row is an unattainable algebraic ceiling for changing only this path.
For the matching scenario:46*22.896000+16*76.490695=2277.067134µs;
0.891946628*(3246.768337/2277.067134)^(1/6)=0.946276626 (+6.091%).
20% latency reduction gives35.181169µs,2842.184895µs/load,0.911952106 (+2.243%).
Neither scenario by itself establishes an overall win over SGLang.

## Projection: what THIS failed candidate establishes

Official adjusted target30.262861→54.191390µs; normalized relative speedup
0.548359; overall1.148649952→1.112041018 (-3.187%).29controls move at most4.691%.
Both jobs passed9prerequisites+30shapes. Same evaluator/image, separate runs;
this is strong rejection evidence for the whole implementation, not TMA alone.

Target profiles: best one library kernel29.2µs; candidate one _pooler_pipeline
56.5µs. These are separate single instrumented samples, not steady-state
medians. They confirm the new GPU path executed. Host descriptor creation
and compilation are not replayed GPU-kernel time; neither is an established
explanation for the measured kernel regression.

**THIS candidate registers/shared memory/spill counts: unavailable, not zero.**
No compiled-resource fields, PTX/SASS, hardware counters, actual HBM traffic,
cache hit rate, achieved occupancy or stalls appear in its saved receipt.
GPU identity/driver/clock/temperature telemetry is available; it does not
provide those per-kernel performance counters or compiled resources.
Earlier M34 shows95regs/40960Bshared/0spills for BM32/BN128 and45regs/
16384Bshared/0spills for BM32/BN32; their source hashes differ from44312859.
Those values cannot be assigned to current BM64/BN64 TMA code or cuBLAS.

Confounders: new backend/geometry/pipeline versus opaque cuBLAS algorithm;
original[N,K] descriptors versus prepared contiguous[K,N] library layout;
compiler scheduling/accumulation order/resources. Both target profiles show
one kernel and no separate split-K reduce; m128 reduction evidence is not
evidence for m256. The ordinary-copy same-geometry control was never GPU-
tested. No conclusion that TMA itself, spills or bandwidth caused the loss.

## One recommended next experiment card (not implemented or run)

- Problem/parent: attention, adopted mixed_fa3.py SHA708d3f06/source51d7fca.
- Hypothesis: official FA3 can cut batch1/L8/sliding normalized latency≥20%.
- Change only this dispatch. Use official QK norm+RoPE, correct new-cache
  writes/scales, native-required FP8 query, causal window=(2047,0), and output
  RMSNorm/gate. Preserve all existing E5 paths/helpers, no competitor code.
- Expected affected shapes:1of12; every other11 is a control. Config-only
  RoPE table follows official baseline (~512MiB), not cached input/output.
- Value:20% cut models+2.243% score; matching reference models+6.091%.
- Reject predeclared goal if target cut<20%, correctness/cache/replay fails
  or another path regresses>5%. If global CI overlaps, global improvement
  remains inconclusive even when target improves. No guaranteed score.
- Before any future GPU admission: preserve BF16/FP8 order, no host value
  sync, verify RoPE positions, page64/nonunit scales, live cache row0,
  untouched rows, CPU argument/dispatch tests, fresh same-address A/B/A.
  Check actual native graph/table lifetime and all official cases if approved.
- Source profile alone does not diagnose registers/occupancy here either.
- Only one attention check remains: complete source/CPU review before any
  future admission; no spare correction check is assumed. This task authorizes
  analysis only, so the recommendation is not an instruction to launch it.
  Attention1/projection7 user checks remain untouched by this analysis.

## Exact saved sources and reproduction

- Attention receipt: `reviews/attention-v2/E5-official-result.json`, job
  `e0b3cd9474084069`; candidate `reviews/attention-v2/mixed_fa3.py`, source
  commit `51d7fca04ccc38a972039ef1b086011d8e456264`.
- Official implementation contract: `reviews/attention-v2/official/baseline.py`.
- Projection inputs are read with `git show` from immutable commit
  `75d1dc117aa40c401cddab015402ee75f8a2f934`: receipts
  `results/remote-P2-02-full-composition.json` and
  `results/remote-P2-03-pooler-m256-tma.json`; candidate sources
  `candidates/projection_full_composition.py`,
  `candidates/projection_pooler_m256_tma.py`, and
  `candidates/projection_pooler_m256_copy.py`.
- Earlier, different-kernel resource receipt: the same pinned commit's
  `results/remote-modal-M34-resource-probe-fixed.json`.
- `next_step_evidence.json` retains full-precision numbers, profile text,
  telemetry, source SHA256s, and every input SHA256.

Reproduce offline: `.venv/bin/python analysis/next_step_analysis.py`.
Checks: twelve shape ratios, raw sample medians, six weighted loads, exact
official score, evaluator identity, resource-source identity, and unchanged
candidate/budget bytes are asserted. No native kernels execute.
