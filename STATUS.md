# P051 compile gate passed — P046 remains selected

Native app ap-Iu9K2yefCGuDlalnHrLuhd completed and was independently verified
stopped/tasks0; all apps stopped. Shared measurement slot released. Source
commit144a0b8e99cdb145acfc3638952c4971012999ae.

Matched operand reversal changes BF16 mma.sync to BF16 WGMMA. Registers42→36,
shared10→20KiB, driver local allocation0both. Generated loop allows one WGMMA
group outstanding during next-buffer copies; no actual overlap/speed is measured.
No generated kernel executed; correctness, replay and performance remain open.
Read experiments/051-gap-closing-orientation.md and raw hash-bound artifacts
under results/remote-P051-compile/. The next narrow screen is specified there.

P0461.1841399409× selected; P039 rollback. P050 parked. No official checks used
in this phase. Delayed gross workspace delta$0.01155627, paid$0, reserve$0.278423;
billing is not a settled per-app invoice. No active/uncertain Projection app.

## Previous checkpoint (historical)

# P051 — major-gap audit and matched orientation diagnostic

P046 remains selected at official 1.1841399409×; P039 remains the rollback.
At 2026-10-25 22:54:34 UTC, the public Projection leader was 1.6099×.
Matching that score requires a modeled 26.4464% uniform effective latency cut,
or 34.8864% across pooler/QKV/unpooler while all other paths stay fixed.
These are conditional requirements, not forecasts or measured bottlenecks.

P050 is PARKED by the strategy change. Its reviewed source/harness at aecd1f6
and 40 passing local tests are preserved; no P050 GPU or official request exists.
Do not launch it merely because quota remains.

P051 prepares a matched old-R06 versus weight-first GEMM orientation diagnostic.
Only operand orientation/direct partial-store indexing changes; geometry,
partition, precision and reducer stay fixed. Compiler eligibility provides a
specific hypothesis; actual native instructions/resources are still unverified.
Read experiments/051-gap-closing-orientation.md and the reproducible gap model.
No new Projection app has been admitted. A bounded compile-only probe is the
next staged action after reviewed freeze and fresh guardrails; no official check.

Official read-only quota observation at 2026-10-25 23:10:08 UTC:
Projection checks 6/20 used, 14 left; Attention checks 18/20 used, 2 left.
Both scored limits are 0/10 used. TASK states reset at 00:00 UTC; the quota
response itself did not supply a reset timestamp. These are platform limits,
not a reason to run weak experiments. Re-read before any actual admission.

## Previous checkpoint (historical)

# P050 QKV-only expansion — reviewed local freeze ready

P049 rejected and result push verified at `4c438423a5f8f85ba4a8757326ea05f01c9ce375`
on22:40:48UTC. P046 stays selected at official1.1841399409×, P039 preserved.
P050 changes only canonical QKV M>1 to zero-bias addmm; unpooler fully restores
P046. Candidate SHA256 `b08b6e371fef83851268a8dbfa1159d1b6d5a8e141a410a596dfc3f159e0da8b`.
Forty local tests and independent reviews passed. Read experiment050 for fixed
score/latency gates, fresh-input coverage and raw-to-official modeling limits.

Nine-case native screen is ready: five scored QKV targets, nearby127/129,
unchangedQKV1/unpooler16 controls. Full completion would check728oracle outputs.
This has not run. No P050 app or official job exists. Attention owns the GPU
slot; wait for a fresh coordinator grant, then verify existing auth, credits,
paid cap$0 and stopped apps before one bounded invocation. Approximate reserve
$0.408freecredits; no new credentials or formal entry. Do not treat prior terminal
slot records as new authority. Passing native screen would warrant official
comparison, never automatic selection.

## Previous checkpoint (historical)

# P049 complete — rejected; P046 remains selected

P049 app `ap-qGKwQTPaVfoieaRJxN3Etw` completed and is independently stopped/tasks0;
all workspace apps were stopped. Slot released at22:37:36UTC. All326oracle
comparisons passed; QKV M128 paired cut3.7804%, unpooler M16 regression3.5852%,
two-case geometric cut0.1655%. Declared5%/2%gates fail. Reject, no official check.
Actual activity changes QKV split-K+reduce to one bias kernel; unpooler keeps one
main kernel with a bias variant. No claim of hardware causality or whole-score win.

Read experiment049 and results/router-p049-assessment.json. P046 remains selected
at official1.1841399409×, P039 rollback immutable. P049 source and failed evidence
are retained. Current quota snapshot6/20checks used14left, historicaltimestamp
in the ledger; reread before any new check. No active/uncertain Projection request.
No P049 broad all-shape or official verification is justified by this screen.

Delayed workspace gross delta$0.01641954, paid$0; reserve$0.407635 retained and
billing may lag. Finish reviewed private result push. Continue bounded local
next-mechanism review; wait for new coordinator slot before any compute.

## Previous checkpoint (historical)

# P049 local screen ready — P046 remains selected

P046 official 1.1841399409× is the working candidate; P039 remains rollback.
Its P048 paired confirmation and selection were pushed and remote-verified at
`26bec37aa5056375eb9dd262424a388a72078511` on 2026-10-25 22:23:49 UTC.

P049 source commit `9374d9c8e3a7dc4ec57ab897b184bd5f1285fe53`, source SHA256
`4c3b0ed4181cbdc70f8557fd7445b6937c1ae0b236724c87178faf9d02116ad7`.
Only canonical QKV/unpooler M>1 changes to public zero-bias addmm; 20,992 bytes
of immutable BF16 zero state, no precision policy or layout change. The actual
CUDA route and benefit are unverified. Read experiment 049 for fixed gates and
the two-case paired screen, including its separate route-equivalence control.
All 34 local candidate/worker/launcher tests passed. No P049 app/check exists.

Attention owns the measurement slot. Wait for coordinator release, then refresh
credits, paid cap, authentication and active apps before one bounded screen.
Do not use a terminal historical slot file as a new grant. P047 exact-judge
FlashInfer capability remains unresolved and is not required by this public API.

## Previous checkpoint (historical)

# P048 complete — P046 selected, GPU slot released

P046 source `candidates/projection_pooler_column_batch2_hybrid.py`, source commit
`fa56f2b`, SHA256 `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Official job `41ba778763a5433b`,9+30pass,1.1841399409×, CI[1.1663778418,1.2019020400].
P048 same-device H200 confirmation passed 480 oracle comparisons; three raw paired
target cuts10.74%/10.87%/9.88%, controls all within 0.7% median movement. All samples
and long tails retained. Select P046 based on target replication and controls,
not official CI overlap or the whole-score point alone. P039 stays immutable.

App `ap-lpXZhacMWtQNlM5oksAWRX` stopped/tasks0, all workspace apps stopped with zero tasks. Free-credit
reserve $0.407635, paid $0; no official check consumed. Latest quota 6/20 used, 14 left
at the ledger timestamp, reset 00:00 UTC. No active or uncertain Projection job.
Read experiment048 for raw/correctness/cost limits. P047 capability audit is
closed unresolved: exact judge image is internal, API/AOT unknown; no import,
JIT, package install or CPU external inventory occurred. No workaround.
Finish reviewed result push then continue bounded supported-mechanism research.
Wait for coordinator's next GPU slot; terminal file is not a new slot grant.

## Previous checkpoint (historical)

# P046 terminal — promising, P039 remains selected

Official job `41ba778763a5433b` passed 9+30 correctness, score 1.184139941×.
Three targets reproduce gains (11.52%, 8.32%, 8.72%), GM9.53%, target-only
model+1.553%. Declared gates pass; M128 original route is restored. But unchanged
O M1/unpooler M1 regress5.62%/10.60% normalized and shift exists before calibration.
Keep P039 selected at1.173238999× pending direct same-device control comparison.
Read experiment046 and complete assessment/control artifacts. Never discard
control samples or claim the aggregate win from its point alone.

Slot released on21:58:55UTC terminal observation. Projection checks6/20used,
14left; ranked0/10; reset00UTC. No active/uncertain Projection request. Finish
reviewed result push, then continue localP047cuBLASLt capability audit. Exactjudge
image isn't externally addressed by known tooling; do not run a different base
image as proof or triggerFlashInfer import/JIT. Attention owns the nextGPUslot.

## Previous checkpoint (historical)

# P046 check accepted — 2026-10-25 21:57:30 UTC

Track only `41ba778763a5433b`, plan `P2-09-pooler-column-batch2-hybrid`.
Source `fa56f2b`, SHA256 `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
One accepted request; never resend. Follow to terminal and release the shared
measurement slot before report/push. Pre-admission quota was 5/20 used, 15 left.
No P046 native result exists yet, and P039 remains selected. Capability research
P047 is local metadata-only preparation, with no external probe or GPU action.

## Previous checkpoint (historical)

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
