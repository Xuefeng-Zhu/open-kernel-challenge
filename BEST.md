# Selected Projection candidate: P039

`candidates/projection_router_m1_fp32_gemv.py`

- Source commit: `cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`.
- SHA256: `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
- Official off-board job: `d5636ac8d1404abd`, done/pass, 9/9 prerequisites and
  30/30 scored cases. Score **1.173238998969285×**, interval
  **[1.1556404139847458, 1.1908375839538243]**.
- Authoritative selection: `experiments/projection-cycle2-budget.json:selection`.
  Receipt: `results/remote-P2-04-router-m1-fp32-gemv.json` and its CSV.

P039 retains the preceding composition and changes only router m1 to FP32 GEMV
on original FP32 weights. The target reference-normalized latency cut is
30.1058%. Preserve the original tolerance, all weights and all other dispatch.
The +2.1407% aggregate point movement versus the earlier composition cannot be
attributed entirely to the router change: unchanged controls moved too. Marginal
CI overlap alone is not a difference test; independent paired comparison is absent.

P040 passed but is not promoted. Its score 1.1758713266400707× is the highest
observed point, not the selected best; O-m1 improvement 6.5077% missed its 15%
predeclared target. Do not change the selection based on that small point gap.

Native canonical m1 correctness/fresh graphs are verified by the official job.
Dedicated native tail/stride/outlier/concurrency stress was not separately run.
No saved P039 library PTX/SASS/register/spill/shared-memory/counter evidence
exists. Older M34/M36 artifacts belong to different sources.

## Historical selection record (superseded, preserved)

Latest follow-up: QKV m128 single-pass035/M36 passed CUDA correctness but
regressed strongly; rejected/no officialcheck. [Exact results](QKV_M128_SINGLEPASS_RESULTS.md).
112localtests pass,all11apps stopped,conservativeModalguard$1.146412/$2,netpaid$0.
This officialbest source/result is unchanged.

Latest follow-on resource/timeline diagnosis: [HOPPER_DIAGNOSTICS.md](HOPPER_DIAGNOSTICS.md).
M34completed/all10apps stopped,108localtests pass; best and officialbudget unchanged.
M33CLI failure preserved/resolved. Conservative stageguard$1.034495/$2,netpaid$0.
Two future hypotheses assessed; no new algorithm benchmark or officialcheck.

# Best completed official point score

**candidates/projection_full_composition.py**, source commit
b9b0fd9fb0cf326b5836167ffc29d1a0d3386fa7; SHA256
c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7.

Off-board official job **48b1357016134e52**, done/pass, 9/9 prerequisites plus
30/30 scored-shape checks, no flags. **1.1486499523x SGLang**, 95% interval
**[1.1314202031, 1.1658797016]**. Original starter remains unchanged.

Compared with prior P2-01 point1.1290207051x, reference-normalized ratio1.0173861;
with R10 point1.1207320490x, ratio1.0249104; with original starter0.9076063758x,
ratio1.2655816. These are independent-run comparisons. Intervals overlap P2-01/R10;
no statistically resolved inter-candidate difference or ranking claimed. The
within-run improvement over SGLang has its interval above1.

Composition: R10 split32/wide hi+lo FP32-router paths, BF16 SIMT m1 pooler/QKV/
unpooler, and existing BF16 GEMM with pretransposed pooler weights above m1.
No precision reduction, input cache, activation precomputation or harness import.
Original source and every independent control remain in their own commits/files.

Actual Modal stability and isolated pooler decisions: MODAL_COMPOSITION_RESULTS.md.
Official full records/CI/profile/limits: OFFICIAL_COMPOSITION_RESULT.md and
results/remote-P2-02-full-composition.json/.csv, results/remote-P2-02-profile.json.
Current source is verified by both environments; Modal gross values are distinct.

Latest follow-up screens M26QKVlayout/M28unpoolerlayout/M30poolergrouped/
M32poolernarrow all passed actual CUDA correctness but none provides a meaningful
whole-workload gain. All rejected; no new official request.104local CPUtests pass.
Allapps stopped, exact costs/budgets/nextresearch in MODAL_FOLLOWUP_RESULTS.md and
STATUS.md. This source remains best official. No new work pushed.

P043 (`128378586ebe4730`) passed9+30 correctness but scored0.898428982; all three compressed pooler targets regressed substantially. It is rejected. Selected P039 source/hash/job above remain unchanged. See experiments/043-pooler-lossless-codec.md.
