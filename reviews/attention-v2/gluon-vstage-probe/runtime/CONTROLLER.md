# V-stage three-arm native validation controller

This local controller adapts the proven common-PV execution guards. It does not
infer execution admission from a compile pass. Before any authentication or
launch, it requires the parent's local `--native-gate-receipt`: explicit PASS
for mechanism/resource/lifetime/matched-work, exact kernel/config hashes,
compile-result SHA256`258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240`,
and hash-bound source/native review verdicts. Actual source, raw compile bytes,
resource pass and all review files are checked. The receipt stays local.

The native claim is qualified: raw K/V loads may be prefetched in both arms;
the admitted remaining contrast is conversion/shared publication around the
corresponding QK wait. No fully serialized control, complete V-load overlap,
universal concurrency or performance advantage is assumed.

The default action is frozen-Git CPU preflight. Explicit `--launch` requires
`--frozen-head`, `--native-gate-receipt`, `--slot-receipt` and
`--billing-receipt`. A new parent serialized slot and fresh free-credit/paid-$0
guardrails remain mandatory. The worker is invoked with
`--execute --output PATH --deadline160`; default worker behavior stays CPU-only.
This controller's author performed no GPU/service/network/credential/push.

Exactly nine allowlisted files are uploaded with their repo-relative structure:
new worker; kernel/config; working3da source; original modal helpers and official
reference/checks; license/notices. The controller itself is frozen too but not
uploaded. Native proof, slot/billing receipts and credentials are never payload.
Source maps and worker SHA256 must match the complete returned evidence.

Three arms are `fa3`, `v_after_wait`, `v_before_wait`. Three cases cover short
tails at unit/nonunit scales and a fresh coherent ctx4096 family, using seeds
135751/135752. The exact completion contract requires30 original-oracle calls
(27 ABA +1 setup +2 concurrent),9 return-A checks,36 K/V cache-byte comparisons,
18 cross-arm new-KV ABA comparisons,9 prototype-output ABA comparisons,4 post-
concurrent isolation checks/2 extra replays,48 checked timing replays,
16 prototype-pair timing comparisons and32 cross-arm new-KV timing comparisons.
Missing, unknown or mismatched success evidence cannot become a pass.

Correctness/timing completion is separate from the worker's boolean
`benefit.mechanism_passed`, `actual_fa3_passed` and
`prospective_followup_supported` fields. The latter must equal the conjunction
of the first two; either negative comparison is a valid completed result.
There is no `benefit.admitted` field. Raw three-arm timing, reference
comparisons and per-fixture arithmetic require independent parent review.
`performance_measurement` is set only after actual raw timing values exist,
not because a row was planned. No automatic performance, production-replacement
or official-check admission occurs; working3da remains preserved.

The existing image/H200/2CPU/8GiB, one app, zero retries,
160s worker/170s child/185s function/20s startup/210s watchdog and$0.58 free-credit
reserve remain unchanged. Actual paid limit is$0; supported existing protected
configuration is used in place without renewal or installs. Fresh CLI billing,
rates and all-app closure checks precede admission. No artificial stage ceiling.

Returned JSON is SHA256 checked and bounded500KB. Exclusive local ledger/result
creation prevents duplicate requests. Raw failed/partial evidence is preserved.
The watchdog/finally stop requests and independent app-state/billing observation
remain in force. `stopping` or nonzero tasks require separate reconciliation,
never rewritten receipts or an automatic retry. The parent reports stopped/
tasks0 and slot release before any next experiment.
