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
