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

Next local hypothesis candidates/projection_qkv_layout.py (experiment025) passes
39CPU+3focused checks but is **GPU-unverified**, not a replacement for this best.
95local regression tests pass. No new GPU run for025, scored entry or push.
Budgets/state: STATUS.md, CONTINUE.md and their explicit ledgers.
