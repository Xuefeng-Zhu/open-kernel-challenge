# Two kernel tracks — completed bounded experiments, 2026-10-25 UTC

Research proceeded in separate branches/worktrees; GPU measurements were serialized.
Attention E5 finished before projection P038 admission. All admitted jobs are
terminal; account jobs and shared GPU slot show no active measurement. No scored
entry, push, paid compute, new credential, terms/payment or spend-cap change.

| Track | Preserved parent | New experiment | Decision |
|---|---|---|---|
| Attention | E2 0.7958818× | E5 0.8919466×, CI[0.8785674,0.9053258] | Select complete mixed_fa3.py, +12.07% over E2; still below SGLang |
| Projection | 1.1486500× | P038 1.1120410×, CI[1.0953604,1.1287216] | Reject regression, keep parent |

Scores are per problem relative to SGLang. They must not be averaged into a new
challenge/leaderboard score. No ranked submission was made.

## Attention experiment card

Problem: our full verify B16/B32 path was much slower than the official baseline.
Hypothesis: route only these two shapes through official FA3 while retaining E2
elsewhere. Exact change: native QK norm, FP8-byte cache writeback (row0 remains
live), required BF16→FP8 query conversion for FA3, causal/paged attention with
supplied lengths/scales, then BF16 per-head RMSNorm/sigmoid gate. Full layers have
no RoPE; sliding retains E2. No competitor source was copied or executed, and no
separate experimental precision change was made. Official/SGLang attribution and
license text are included.

Predicted affected shapes: only full verify B16/B32 (2/12). Actual adjusted times:
339.220→204.304us and588.574→389.661us. Reference-normalized shape ratios versus
E2:1.6476 and1.5110. All10controls stay within5%. The hypothesis is supported.
The exact checked file includes all paths/entrypoint and is selected unchanged;
no source copy or known-duplicate final check is needed.

Source reviews/attention-v2/mixed_fa3.py, commit51d7fca04ccc38a972039ef1b086011d8e456264,
SHA708d3f06f46bfb24a7a43963022eb51426f3a68bb2155c77dea8e9b6ae255423.
Official checke0b3cd9474084069:16prerequisites+12timed shapes passed. All raw,
calibrated, per-shape and six weighted-load data remain in its receipt/summary.
Four CPU adapter/AST/oracle tests include page64, nonunit scales, slot0, untouched
cache rows and CPU same-address A/B/A with changing lengths. They use stand-ins,
not native FA3 or CUDA graphs. Actual target-path GPU A/B/A/page64/nonunit and
concurrent-call tests remain unverified; fallback E2 retains global ticket limits.

Read reviews/attention-v2/E5_MIXED_FA3.md and official-cycle-summary.json.
E2 final_attention.py and completed attention-v2-preflight branch remain preserved.

## Projection experiment card

Problem: medium/large pooler remains costly after prior layout/single-pass failures.
Hypothesis: an explicit descriptor/TMA-fed BF16 pipeline can improve pooler m256.
Exact change: only dense aligned m256 pooler, BM64/BN64/BK64,4warps,3compiler-managed
stages, BF16 operands/FP32 accumulation/BF16 output. Prepare and all other paths
are AST-identical to best. An identical-geometry conventional-copy control differs
only by USE_TMA and remains unmeasured. Documented Triton3.8 API, host descriptor
lifetime and narrow import allowance were checked; no allocator mutation or
manual barriers. MIT/Apache notices accompany the standalone sources.

Predicted affected shape: pooler m256 (1/30). Actual:30.2629→54.1914us; normalized
speedup ratio0.54836, a clear >5% regression. Overall candidate/best0.9681287
(-3.19%). All29controls stay within5%. The concrete hypothesis failed: do not
promote, run the copy control just to fill quota, or sweep more tiles. This does
not prove all TMA implementations slow or causally isolate loading from geometry.

Source candidates/projection_pooler_m256_tma.py on projection-pipeline-preflight,
commit60a8b4bdcdd4f3c48d2b7a8a0db4907c394ba84a,
SHA443128591305c4a8f79f2e8669d59b3cc402a5243b9920273676583b825f518d.
Official check6799753227ed463e:9prerequisites+30timed shapes passed. Profile shows
one _pooler_pipeline GPU call (~56.5us in a separate instrumented sample), proving
the new path executed. Use official paired timing for performance; that single
profile is not a steady-state benchmark. No PTX/SASS/register receipt, hardware
counter, achieved occupancy or DRAM/stall measurement was obtained.

Both sources passed39CPUoracles and117full tests (including5new focused tests),
plus2separate coordinator ownership tests. Native official compilation, published
fresh-input graphs and correctness now pass; nearby-tail/fallback/concurrency GPU
stress was not separately run. Negative results/raw samples are retained.

Isolated worktree: /tmp/okc-projection-pipeline-preflight. Source/result branches
are durable in the same Git repository even if the temporary checkout is removed.
Result commit73a79d8f28c40dd13bdd3930202990beab94705b. Read its
experiments/038-pooler-m256-pipeline-local.md and results/P038-official-assessment.json.
Preserved projection-bf16-local remains0eabf0e with best sourcec34c497b,job48b1357016134e52.

## Budget, resources and reproducibility

User remaining: attention1request of its six-check scope; projection7checks of its
second ten. UTC reset did not reset these budgets. Current platform daily limits:
19/20checks left per problem,0/10scored used. Actual quota/account snapshot and
all source identities are in DUAL_TRACK_RESULTS.json. No uncertain or pending job.

No Modal app launched in this cycle. Fresh read-only inventory shows recent A07/A09
apps stopped/tasks0; older stopped receipts are preserved. Net billed amount$0;
observed delayed gross$1.15. Cumulative conservative guard stays$1.708917914/$2,
never lowered from delayed billing, including the failed A09. No credential was
renewed. Prior unresolved operation: A09 return packaging BlobCreate returned
AuthError401. App stopped; report unavailable; no retry or access broadening.
It does not invalidate these independent official results.

Both scores are the geometric mean of six weighted loads (attention46sliding+
16full; projection62per layer). Within-run SGLang pairing is authoritative;
cross-run candidate comparisons are not directly paired trials. Full samples and
confidence intervals are preserved; no CPU timing is used as GPU evidence.

```sh
.venv/bin/python reviews/attention-v2/test_mixed_fa3.py
.venv/bin/python reviews/attention-v2/summarize_official.py
bin/okc status e0b3cd9474084069 --json
bin/okc status 6799753227ed463e --json
```

Projection offline summary from its worktree:

```sh
<primary-workspace>/.venv/bin/python summarize_check.py P2-03-pooler-m256-tma --ledger experiments/projection-cycle2-budget.json
<primary-workspace>/.venv/bin/python projection_admission.py
```
