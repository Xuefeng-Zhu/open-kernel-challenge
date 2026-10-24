# Official composition check — best observed passing point score

Job **48b1357016134e52**, mode **check** (off-board), completed2026-10-24 19:26:10UTC.
Started19:25:08UTC; no replacement or retry. Nine prerequisites and all30scored
shape checks pass, no flags/rejections. Source is the exact Modal-verified
**candidates/projection_full_composition.py**, commit
`b9b0fd9fb0cf326b5836167ffc29d1a0d3386fa7`, SHA256
`c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7`.

**Official speedup1.1486499523x SGLang**,95%interval
**[1.1314202031,1.1658797016]**. Lower bound above1 supports improvement over
SGLang in this run. This is an off-board result, not a public rank/final entry.

| Prior control | Official point score | New/control normalized ratio |
|---|---:|---:|
| P2-01 pooler layout |1.1290207051|1.0173861|
| R10 native router/pooler |1.1207320490|1.0249104|
| Original starter |0.9076063758|1.2655816|

These prior-run comparisons normalize each result by its own SGLang reference;
they are not same-input paired comparisons. New CI overlaps P2-01 and R10 CIs,
so point-score differences are not declared statistically resolved. Full exact
original-baseline score/records remain in results/h200-check.json. No score is
inferred from Modal gross timing.

| Load | New official SGLang/candidate ratio |
|---|---:|
|decode1|1.212103|
|decode16|1.086826|
|decode32|1.096358|
|verify1|1.350693|
|verify16|1.098073|
|verify32|1.072227|

The main change from P2-01 is at single-token decode: QKV12.0471us versus its
SGLang13.7280us (1.13953x); unpooler4.1234us versus7.0720us (1.71510x). QKV's
SGLang-normalized ratio over P2-01 is1.15348x; unpooler's over R10 is1.69937x.
Untouched paths also vary: larger verify loads slightly trail P2-01's previous
run. Preserve all shapes and overlapping CIs instead of crediting every difference
to source changes. Full tables: COMPARISON_OFFICIAL_FULL_VS_POOLER.md and
COMPARISON_OFFICIAL_FULL_VS_R10.md.

## Remaining opportunities and next reported hypothesis

Official mean per-load time shares: pooler32.14%, QKV22.42%, unpooler21.01%,
router13.49%, o10.95%. The same official profile shows QKV m8/16/32/128 uses a
cuBLAS split-K main kernel and1.6–3.0us separate reduction. This supports the next
isolated **BF16 QKV m>1 weight-layout** experiment025, not a performance prediction.
It changes only torch GEMM RHS layout, adds36MiB of weight-only storage, retains
m1/other families and original precision. Candidate025 passed39CPU+3focused cases,
static/syntax; no GPU compile/capture/timing or request has occurred.

Later opportunities: isolate unpooler large-M layout while preserving the <=8
slice contract; investigate a legal native BF16 tile only if library evidence
warrants it. No cross-layer activation fusion/cache is allowed by this one-layer
contract. BF16 weight compression to FP8/FP4 is forbidden. Profile kernel totals
are separate measurements and cannot be subtracted to invent a new official score.

## Ledger, cost and resume

Second projection cycle **2/10used,8remaining;0active,0uncertain**. This authorized
one-check exception is consumed; all other official admissions paused. Last
platform quota: checks13/20used,7left; scored0/10used,10left. Original ten-check and
attention one-check budgets remain separate/exhausted. Never resend an admitted ID.

Modal cumulative gross/credits$0.29, paid$0, balance$29.71, conservative stage
bound$0.44625<$2; all four apps stopped. No Modal GPU run added in this turn.
Existing credentials/cap unchanged; no paid compute, terms or push. Original
submission.py, R10, all candidate sources and attention branch preserved.

Exact official records/paired timings/CI/evaluator metadata:
results/remote-P2-02-full-composition.json/.csv. Sanitized emitted kernel profile:
results/remote-P2-02-profile.json. Hypothesis/source and exact commands:
experiments/024-official-full-composition.md and025-qkv-layout-local.md.

Local regression suite after candidate025: **95/95 tests pass**; CPU/static and actual H200 evidence remain separately labeled.
