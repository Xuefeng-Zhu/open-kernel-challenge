# Attention corrected-v2 and separate-merge test — complete

Completed 2026-10-24T22:06:32.381804+00:00. Branch attention-v2-preflight, local commits only. No OKCcheck,
formal/ranked submission, push, credential creation/renewal or paid resource used.
Projection best1.14864995x and all its work remain unchanged on projection-bf16-local
at0eabf0e. Third-party1.4807x/1.4939x receipts are not our results; no competitor
repository code was executed or reused.

## Compiler baseline established first

Prior official checkbf4b49bd8bb34dae failed all16prerequisites at non-constexpr
LOG2E, with no numerical/timing result. Applied **only**
`LOG2E = tl.constexpr(1.4426950408889634)` in source commit
05922b7bf0353a3f4a54aaa27f9eef0692a4ee32, SHA256
3a86e83511725c581f288c4b5b07a8484a50e88cd396466d00a97f330cd2d353.
Original paste/source/hash remain in history; no algorithm fix silently bundled.

A03baseline app **ap-KGuppwXiAqOLZIE6m75wS0**, worker137.313s/app150s, completed
and verified stopped/tasks0. **All16official prerequisite cases and12scored shapes
pass** for correctedv2 and same-run officialSGLang. PublishedAPI files/TASK/AGENTS
were byte-for-byte verified against snapshots before running. Includes TP8/DP,
L1/8,page1/64,scales1and1/64,edges/soft/coherent,mixed/agenticlengths,millioncontext.

Whole-call raw graph medians in A03:

| full verify batch | corrected v2 | official SGLang | measured empty-graph median |
|---|---:|---:|---:|
|1|80.480us|43.632us|3.232us|
|16|395.104us|222.880us|3.168us|
|32|701.632us|407.040us|3.312us|

Raw weighted load geometric SGLang/v2 diagnostic ratio **0.76406**, so this v2 is
slower overall in this public-image test. It is **not an official contest score**.
No ranking or archival receipt is claimed. Calibration-adjusted per-shape values
are kept separately in modal-summary.json; no guessed judge subtraction.

## The single-factor comparison

After baseline passed, source commit3818c7255cd2d9aa5859b2e55ee83f2bd2f057d4 adds
separate_merge.py, SHA256
ddd9065c3fe444e7e5b97ed62101c7826b7745422e618aec80d910b19cbbb1a0.
It moves the identical serial merge recurrence/norm/gate/store to a second launch.
Partition count, attention/QKnorm/RoPE/precision/cachewrites/new-token owner and
partial layout remain identical. SPLITS1retains direct output. Scratch/output are
invocation-local; **no global ticket or atomic counter** exists in the variant.
Four AST invariants verify that exact scope; actual GPU verification followed.

The variant is deliberately **full-attention-only**, not a complete contest entry.
For sliding, both comparison arms call unchanged correctedv2 as a control. This
avoids changing sliding or retaining shared tickets in the new full-only function.
Do not submit this experimental file as an all-shape kernel.

A05comparison app **ap-sS1X8nAzzsQPkNzcNIhBGq**, worker108.805s/app118s, completed
and verified stopped/tasks0. All prerequisite/full numerical checks pass. Coverage
is20full prerequisite runner checks plus12sliding-control checks,12timed shapes
with192checked warmup/steady replays. Actual new variant covers10full prerequisite
cases and6full scored shapes; sliding uses originalv2, not the new function.

| full case | v2 raw median | merge raw median | paired v2/merge | merge wins |
|---|---:|---:|---:|---:|
|decode B1|45.392us|79.184us|0.57676|0/3|
|decode B16|289.872us|287.040us|1.00526|2/3|
|decode B32|510.768us|513.408us|0.99470|1/3|
|verify B1|80.848us|149.632us|0.54238|0/3|
|verify B16|393.056us|394.688us|1.00582|2/3|
|verify B32|691.456us|680.640us|1.02043|3/3|

Paired ratios are medians of reverse-order pair ratios, not ratios of marginal
medians; these can differ, as verifyB16 illustrates. VerifyB16is parity; verifyB32
has a small positive point estimate, but only3pairs and sliding control ratios
span0.9911–1.0347despite identical source. That control variation prevents a claim
of a demonstrated stable2%optimization. B1regressions are large in all pairs.

**Reject the variant as a whole**: raw weighted load geometric v2/merge ratio
**0.91640**. No follow-up tuning or new algorithm was tried. Do not multiply this
ratio by A03's SGLang/v2 as if it were a measured same-run SGLang/merge score.

Independent empty-graph calibration is separate:

| full verify batch | v2 calibrated median | merge calibrated median |
|---|---:|---:|
|1|77.616us|146.400us|
|16|389.808us|391.440us|
|32|688.224us|677.408us|

These subtract each case's measured empty-graph median from both arms, retain all
raw values and do not clip. This is a diagnostic calibration, not the private
judge's fixed-cost calibration. All cases resolved positive calibrated times.

## Compiler resources and limits of the diagnosis

For full verify B16/B32, correctedfusedv2 uses135registers/thread,0driver-reported
spill,24832Bshared. Newpartial uses128registers,0spill,24832Bshared; separate merge
uses66registers,0spill,256Bshared. Full PTX and hashes preserved in A03/A05 resultJSON.
Lower register count alone did not produce an overall win. No achieved occupancy,
actual HBM traffic or causal stall breakdown was measured; no restricted counters
were requested or bypassed. Do not invent the cause of B1regression from these data.

Actual environment: NVIDIA H200,132SM,Torch2.14.1+cu130,Triton3.8.0,Python3.12.3,
existing cached public SGLang imageim-cPTbGpNiX3RarKResPgX7h. Driver/clock counters
were not separately sampled by this attention harness. Public image is not the
private judge overlay. Compile/setup/wall time is separate from graph replay time.

Correctness uses unmodified official collect/check_implementation: cacheentries
within8%; output bound clamp(2.5*referenceFP8queryerror+1.5%,3%,15%) plus its
per-row/near-zero rules, against output attending the candidate's checkednewKV.
Each arm starts from restored immutable input/cache snapshots. Non-new cache rows
and all noncache tensors/configs must remain bytewise unchanged. New cache/output
is collected before restoration. No input-derived outputs cached outside kernel_fn.

Prerequisite cases each had one captured checked replay per arm after warmup;
all12scored shapes had2declared warmups retained/excluded and6steady fresh-input
replays alternating two runners,3reverse-order pairs. Every replay's output/cache
was checked. Fresh qkv,lengths,pages,caches and scheduler fields are copied into
fixed graphbuffers; shape changes fail visibly.256MiB coldL2read before events.
All seeds/samples/calibration observations retained, no outlier deletion or CIclaim.

Sequential graphs only: correctedv2 retains its documented device-global-ticket
concurrency defect, not exercised under overlap. The new full path has no shared
counter, but concurrent GPU stress was not performed. Numerical acceptance does
not settle earlier upstream intermediate-BF16 policy ambiguity; precision/math
ordering was intentionally unchanged in this narrow experiment.

## Local checks, failure history and exact commands

A01syntax passes;12relevant CPU/AST gates before baseline and16before comparison
pass. CPU equation cases pass the officialchecker but are not Triton/GPU evidence.
Historical normalization-only test initially failed because it asserts no compiler
repair; preserved cpu-constexpr-results.json, then replaced for this gate by an
exact five-link-normalizations-plus-one-constexpr assertion. Originalpreflight and
originalCPU evidence untouched. Initial A04scratch-prefix assertion counted3rather
thanpbase+3stores; corrected test, source unchanged, all4ASTinvariants pass. A final
ad-hoc inventory assertion expecting13historicalapps failed when the recent app
list shrank to5; exact attention IDs were rechecked stopped, all visible apps stopped,
older shutdown snapshots retained. No remote mutation or GPU retry resulted.

```sh
.venv/bin/python reviews/attention-v2/test_modal_worker.py
.venv/bin/python reviews/attention-v2/test_separate_merge.py
.venv/bin/python -m py_compile reviews/attention-v2/submission.py reviews/attention-v2/separate_merge.py reviews/attention-v2/modal_worker.py
.venv/bin/python reviews/attention-v2/summarize_modal.py
```

Exact historical allowlist/build/run/finalize commands: MODAL_PLAN.md andA05_PLAN.md.
Do not rerun accepted IDs. Source, harness and results each have separate commits:
05922b7A01,cf36a66A02,84e54d6A03,3818c72A04,0524fddA05prepare,eb0c095A05result.

## Accounting and shutdown

Original cumulative guard **$1.146411930** was carried across branches, never reset.
After both attention apps: **$1.565468698/$2** conservative cumulative guard,
remaining **$0.434531302**. Latest gross billing snapshot$1.21000000,
fully covered by free credits, **netpaid$0**; fine-grained observed highwater
$1.21273207. Counters settle asynchronously; no guard was
lowered after a smaller delayed read. No new resource can reserve$0.50under this
remaining ceiling; this cycle is complete regardless.

Both exact attention app IDs reverified stopped/tasks0. Current recent app list
contains onlystoppedapps; older projection shutdown evidence preserved in ledger.
No persistent deployment, detached run, volume, newcredentials/renewal, payment/
cap/terms change, push or OKCrequest. Existingtokenexpires2026-10-25T06:26:47Z.
Currentauthoritative ledger is modal-budget.json on attention-v2-preflight;
ignored local-notes/modal-budget-latest.local.json bridges later branch switches.
Do not restore projection's old1.146guard as currentaccounting.

No unresolved denied/pending/uncertain action. The bounded experiment completed;
the merge variant loses overall, so correctedv2 remains the attention control.
Neither has a new official score, and projection's1.14864995x remains preserved.
