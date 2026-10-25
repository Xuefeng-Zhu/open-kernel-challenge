# P046 narrow pooler composition — local ready

P045 is rejected and privately pushed through `7b1ca18`, remote verified at
21:41:07 UTC. Selected P039 remains 1.173238999×. P046 changes only P045's guard:
pooler M8/M16/M32 use BMM; M128 and all other paths restore P039.
Seven bounded CPU tests and independent review passed; source SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
No P046 native validation, request, reservation or job ID exists.

Read experiment 046 for newly declared three-target gates and explicit lower
whole-score upside (~1.237% conditional model, not measured). P045's broader
failure/gates remain unchanged. Current code is local; Attention owns the
measurement slot for robustness validation. Wait for fresh coordinator release,
then use committed source and the P2-09 plan with current jobs/quota checks.
Do not infer slot permission from old terminal records or repeat P045.

## Previous checkpoint (historical)

# P045 complete — preserve P039, measurement slot released

Job `b2c1e6c724144e87` passed all 9 prerequisites and 30 scored shapes;
score 1.179863621×, interval overlaps selected P039 1.173238999×. Reject promotion:
four-target normalized geomean gain 4.52% misses 10%; M128 regresses 5.54%;
target-only modeled score gain 0.964% misses 1.5%. All targets execute one main
kernel without a listed independent reducer; hardware causality remains unknown.
See experiment 045 and results/router-p045-official-assessment.json.

One check admitted, now terminal; no uncertain request, retry or active Projection
GPU job. Slot released at 21:36:40 UTC. Quota then 5/20 used, 15 left; reset 00:00 UTC.
P045 is preserved as a rejected experiment; P039 remains the selected source.
Finish reviewed result commit/private push. No immediate small-M rescue check:
its conditional modeled +1.237% is below the original gate. Next local mechanism
can examine QKV independent-output batching, with separate tests and a fresh
coordinator slot before any native measurement. Do not infer a new slot grant
from this terminal state or modify Attention worktrees.

## Previous checkpoint (historical)

# P045 official check accepted — 2026-10-25 21:34:36 UTC

Track only job `b2c1e6c724144e87`, experiment `P2-08-pooler-column-batch2`.
One request accepted, no uncertain admission. Source commit `a3aba7b`, SHA256
`c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58`.
Do not send again or change the candidate. Read the exact job to terminal,
then release the shared measurement slot before polishing/pushing results.
P039 remains selected; no P045 GPU result or speedup exists yet. Pre-admission
quota was 4/20 used, 16 left; refresh before reporting final availability.

## Previous checkpoint (historical)

# Current Projection checkpoint — 2026-10-25 21:26 UTC

P045 is local and ready for serialized GPU validation: output-column BMM,
source SHA256 `c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58`.
Thirteen bounded CPU/static tests passed; native CUDA correctness, graphs,
actual tactic, and performance remain unverified. Targets: pooler M=8/16/32/128;
all other scored paths stay P039. See experiment 045 for exact reduced-K limits.
No new GPU request or app. Attention currently owns the measurement slot.

P044 is complete and pushed privately through `aa730b8f1e7e0b000df41ea3abe7eb26da85efd9`,
verified at 21:16:50 UTC. App `ap-pF3Ammg5fT5638UgFJTrox` is stopped with zero
tasks. Its 610 oracle checks, 480 measured observations, ten sanitized traces,
and billing/shutdown evidence are preserved. The earlier P044 denials were
resolved by the explicit approved retries; neither operation remains pending.

Selected P039 remains 1.173238999×, source `cdf5bc2`, job `d5636ac8d1404abd`.
Latest historical quota read at 21:16 UTC: Projection checks 4/20 used, 16 left;
reset 00:00 UTC. Recheck current state before dispatch. No formal entry.

## Previous checkpoint (historical)

# Current Projection checkpoint — 2026-10-25 21:10 UTC

P044 diagnostic completed: frozen P039, app ap-pF3Ammg5fT5638UgFJTrox,
stopped/tasks0 verified, slot released. Five paths/610 oracle checks passed,
480 measured observations and10 traces preserved. No new official score.
P039 remains selected1.173238999×. See experiments/044-projection-evidence-diagnostic.md.

Cold paired baseline/P039 ratios: pooler8/128/256=1.05205/1.05239/1.07268;
QKV128=1.01354, unpooler16=1.00500. Warm controls were generally slower in both
runners, so no L2/HBM claim. Actual launch resource fields were captured; exact
vendor instructions, spills and hardware counters remain unavailable.

Reserve$0.709128<$0.71, observed delayed workspace delta$0.07047953, actualpaid0.
No new credentials, installs, NCU or officialcheck. Source/result push is now
explicitly authorized; review pending results before the one approved push retry.
Local output-column-sharded BMM prototype is being prepared; no GPU admission.

## Previous checkpoint (historical)

# Current Projection checkpoint — 2026-10-25 20:15 UTC

P044 is local-only preparation for unchanged P039. No app or official check has
been admitted. The shared GPU slot was released to Attention at 20:11 UTC while
this diagnostic launcher is completed; wait for a fresh coordinator grant.
Selected P039 remains 1.173238999×, and P043 remains a rejected terminal result.

The P041 worker now separately records synchronized preparation time and raw
activity traces. Eighteen plan/worker tests pass. Every representative retains
same-address A→outlier B→A and fresh-input oracle checks; no GPU result is implied.
See experiments/044-projection-evidence-diagnostic.md and projection_p044_analysis.py.

Fresh quota 20:05 UTC: Projection checks4/20,16left; Attention16/20,4left;
both ranked0/10. Supported official show at20:11 UTC confirms reset00:00 UTC.
Modal existing authentication works, browser credits28.40, net spend cap0,
paid0; no live apps at20:08. Refresh state at admission; no credentials changed.
Missing vendor instructions and hardware counters remain explicitly unverified.

## Previous checkpoint (historical)

# Current Projection checkpoint — 2026-10-25 19:14 UTC

P043 is terminal **done**, job `128378586ebe4730`: H200 compilation and9 prerequisites+30 scored correctness checks passed. Score **0.898428982×** [0.884952547,0.911905417]; rejected for performance. Selected P039 remains **1.173238999×**, source `cdf5bc2`, job `d5636ac8d1404abd`, SHA256 `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.

P043 source `candidates/projection_pooler_lossless_codec.py`, commit `10baddb`, SHA256 `9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837`. Three-seed bit-exact weights shrink about23.3% including metadata, but target m8/16/32 calibrated times are52.639/51.912/98.953µs versus P03920.162/19.932/20.237µs. All27 controls move less than5% normalized; profiles prove codec execution. No causal hardware-bottleneck claim. See experiment043 and full sanitized receipt/CSV/assessment.

Local16 focused tests and39 CPU oracle cases passed. Dedicated GPU exhaustive-bit/ABA/concurrency tests remain unverified. Current quota observed19:14:36UTC: Projection checks **4/20 used,16 left**, ranked **0/10**. UTC reset00:00; user history is separate. Seven cycle-two request records including one settled duplicate; all terminal, zero uncertain. A fresh account inventory at19:17:12UTC returned39 jobs and no nonterminal jobs. No Modal app was launched, no new credentials or paid resources, no ranked submission. Shared GPU slot terminal.

This bounded single-candidate experiment is complete. Reject the fixed codec implementation without a tile/split sweep; a future mechanism needs stronger dataflow/generated-code evidence. Preserve raw matched control, which is not GPU tested. Attention and other worktrees untouched.

## Previous checkpoint (historical)

# Current Projection checkpoint — 2026-10-25 18:48 UTC

P039 remains selected: source `cdf5bc2`, SHA256 `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`,
job `d5636ac8d1404abd`, **1.173238998969285×**, 9+30 native correctness passed.
P040 remains an unselected passing experiment. See BEST.md and original receipts.

The user's direct-official request sent unchanged P039 once. Request
**d9b940ff4fa14a1b** is terminal **duplicate**, pointing to the original P039 job;
`result=null`. **No new GPU execution, score, correctness or per-shape measurement**
was produced. No source nonce, second check or retry. See
[experiment 042](experiments/042-p039-unchanged-official-recheck.md).

Fresh platform observation at 18:48:29 UTC: checks **3/20 used, 17 remaining**,
scored **0/10 used**; duplicate consumed zero quota. Daily reset is 00:00 UTC.
Six request records, five non-duplicate results, all terminal. No uncertain request.
Historical numeric allowance fields are not platform availability.

P041 local evidence reconstructs the exact six-load score and freezes five
profiling representatives. Fifteen new offline/mock diagnostic tests and six
related P039 tests passed. **The new worker has never executed on GPU.** No Modal
app was created by this phase. The coordinator released the shared slot after
attention shutdown, but the user then selected the direct official workflow.
Do not automatically launch the deferred diagnostic.

Ordinary activity evidence exists; both attention-owner NCU attempts timed out
without metric values or a permission denial. Do not repeat that counter method
or infer measured bandwidth/occupancy/stalls. No credentials or billing changed.
Projection/attention sources outside this worktree were not modified.

## Historical checkpoint (superseded, preserved)

# Current checkpoint — 2026-10-24T20:29:48.540681+00:00

QKV m128 single-pass experiment035/M36 complete. CUDA correctness and36extra
checks pass, but target55.104us vsbest21.888us,0/15pairedwins; whole grossratio
0.9560322. Rejected, no officialcheck. NativeWGMMA/46regs/0spill observed, noTMA.
Details/source/hash/all samples: QKV_M128_SINGLEPASS_RESULTS.md.

Best unchanged: fullcomposition c34c497b,b9b0fd9,official48b1357016134e52,
1.1486499523xSGLang [1.1314202031,1.1658797016]. Originalstarter preserved.
112CPUtests pass; CPUmodels and actualCUDA checks are separate evidence.

Officialcycle2/10used,8remaining; daily freshquota13/20checksused7left,scored0,
observed2026-10-24T20:26:39.185676+00:00. No active/uncertainjob.
All11Modalapps stopped/tasks0. Latestgross0.89fullycredited/net0,conservative
stageguard1.146411930of2USD. Headroom0.853588070; never lower guard from delayed
billingreads. No cost/quota blocker. Stop this candidate for measured regression.

No newcredential/cap/payment/terms/push/rankedentry. Existingtokenexpiry06:26:47UTC
Oct25. Stagecomplete,no denied/pending/uncertainoperation. No broadpipeline rewrite
implemented; further architectural experiment would need separate source/evidence.
