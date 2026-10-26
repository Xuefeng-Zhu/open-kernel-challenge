# V-stage native publication gate passes; execution prepared, slot released

The [revised compile result](reviews/attention-v2/gluon-vstage-probe/COMPILE_RESULT.md)
passes the prospectively declared remaining conversion/shared-publication gate.
Both arms prefetch raw K and V; no V-load serialization is claimed. Resources
are166/168 registers,32KiB shared,zero spill/local,3 legal CTA/SM each.
Compile app `ap-XzBXIhHnYx5BgeBYIoyAkL` independently stopped/tasks0 at
00:10:53.666428 UTC; delayed workspace delta$0.01044102,paid$0,not exact invoice.
The older fully-serial contrast remains failed and preserved.

The [runtime worker/card](reviews/attention-v2/gluon-vstage-probe/runtime/CARD.md)
and independent semantic/controller reviews are ready for source freeze and
preflight:30 original-oracle checks before48 complete-call timing replays,
with actual unchanged FA3 as a third arm. Fixtures are padded nearbyctx4096,
not scored80K; no score or promotion is implied. No numerical job is admitted.
The parent requested slot handoff to Projection P056; no new GPU launch until
coordination. Working source3da22a/official1.233633059 and A2-11 rollback stay
unchanged. B1 route/window budget and rules-compilation audits are local evidence;
no checker was relaxed and no formal entry was made.

Earlier checkpoints below retain their original observations.

---

# Gluon compile: resource/ownership pass, strict native control fails

The [Gluon attention-core result](reviews/attention-v2/gluon-overlap-probe/RESULTS.md)
compiled two fixed arms and preserved ten native artifacts. Static precision,
buffer ownership and resources pass:167/168 registers,32KiB shared, zero
spill/local and legal maximum3 CTA/SM in both arms. The native control already
prefetches next K before its QK wait, so the declared fully serial control was
not retained. Partial V/conversion/store overlap is real codegen evidence;
no candidate invocation, numerical correctness or performance result follows.
The failed strict gate is preserved; the execution branch stops pending a
separately declared interpretation and review.

App `ap-AECevTBOrJ7vVRE4EWgk3R` was independently stopped/tasks0 at
23:44:21.845040 UTC and the slot released. The23:47:17.887804 UTC no-new-app
cutoff records delayed detailed workspace delta$0.01044224, paid$0, not an
exact app invoice. Working fusion+lookahead SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
remains selected at1.233633059x; A2-11 rollback and Projection records remain
unchanged. Main architecture research still covers B1/sliding/full; no missing
FA3 page1 mechanism or winning performance is claimed. No official quota was
used by this stage.

Earlier checkpoints below retain their original observations.

---

# Common-PV runtime rejected; broad architecture work continues

The [common-PV native result](reviews/attention-v2/alpha-one-common-pv-probe/RESULTS.md)
passed all 39 original-oracle checks and replay/cache/diagnostic gates. Target
alpha-one frequency was about 79.29%, but it won 0/6 paired timings; the two
fresh seeds were 0.54976% and 0.47631% slower. The declared benefit gate failed.
Stop this alpha-one route; no fresh confirmation or official check follows.

App `ap-wE6ZSXzC1dnP253vX0kM5l` is independently stopped/tasks0 at
22:51:11.150348 UTC. The closed delayed detailed workspace delta is $0.07047792,
paid $0; it is not an exact app invoice. Working composition SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
remains selected at official 1.233633059x, with A2-11 rollback and Projection
records preserved.

The main effort now targets broad B1/sliding/full attention architecture and
mainloop/data movement. The CPU-only [image identity audit](reviews/attention-v2/fa3-next-evidence/image-inventory/IMAGE_RESULT_REVIEW.md)
confirmed Gluon Hopper APIs and rejected installed XQA for page1. Exact FA3 C++
revision remains unknown; no missing FA3 mechanism is proved. Its app stopped/tasks0,
with no GPU request, observed delayed workspace delta $0 and paid $0. A matched
Gluon serial/overlap microprototype is local preparation, not a production change.
Use existing free credits with actual paid cap $0, existing credentials and
serialized measurement slots under the standing project scope.

Earlier checkpoints below retain their original observations.

---

# Alpha-one branch rejected before execution; working source unchanged

The [compile-only alpha-one probe](reviews/attention-v2/alpha-one-rescale-probe/RESULTS.md)
compiled all four same-geometry units and retrieved 24 hash-verified artifacts,
but failed its declared resource gate: producer registers rose 96 to 128, and
V asynchronous buffering was lost. Shared memory stayed 20,480 bytes and spills
zero. Actual rescale bypass and unchanged merge SASS do not establish a speedup.
No candidate/merge invocation, frequency run, native correctness test, timing
or official check followed. This is not a measured slowdown.

App `ap-uY5K1rBNFRR2K8rdo2F8tw` was independently stopped/tasks0 at
22:06:59.192653 UTC and its measurement slot released. Closed accounting showed
a detailed delayed workspace delta $0.02871385, paid $0; no exact app invoice
is claimed. The rejected source SHA256 is
`e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764`.

Attention still selects composition SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
with official score 1.233633059x; DP two/three-chunk validation remains passing.
A2-11 rollback, fusion control and Projection records are unchanged. A distinct
local lowering card considers conditional MUL followed by one common PV, based
on observed branch/liveness changes. No thresholds are relaxed, no failed
source is overwritten, and no further test is launched or reserved here.

Earlier checkpoints below retain their original observations.

---

# Attention DP multi-chunk robustness passed

The working fusion-plus-lookahead candidate is unchanged:
`reviews/attention-v2/fusion-lookahead-probe/candidate.py`, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Its [DP validation](reviews/attention-v2/dp-merge-validation/RESULTS.md) passed
two native B32/L1 DP cases at ctx8193/SPLITS2 and ctx16385/SPLITS3/mergeBS4:
15 original-oracle calls, four returning-A checks, twelve K/V byte comparisons
and four independent-graph isolation checks. Distinct graph pools and disjoint
outputs/caches were observed; actual hardware overlap and internal scratch
pointers were not. General serving concurrency remains outside this evidence.

App `ap-gXUKhAM11JBfvrA3h5rip9` was independently stopped/tasks0 at
21:52:08.731188 UTC; the parent received immediate measurement-slot release.
At the no-new-app accounting cutoff, delayed detailed credit usage increased
$0.03730389, paid $0; this is not an exact per-app invoice. No official check
or timing was performed, so the official score remains 1.233633059x and the
last observed Attention quota remains two checks. A2-11 rollback and Projection
records are preserved. The strict alpha==1 historical-numerator-rescale probe
is next local preparation; it has no compute slot or launched job here.

Earlier checkpoints below retain their original observations.

---

# Working attention selection: fusion plus lookahead

Attention now selects `reviews/attention-v2/fusion-lookahead-probe/candidate.py`
SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
frozen at `c9062363c665e37535d3370c1b6bfb54e04fc7c8`, on independent branch
`attention-fusion-lookahead`. See the [complete report](reviews/attention-v2/fusion-lookahead-probe/RESULTS.md)
and [selection/rollback record](reviews/attention-v2/fusion-lookahead-probe/WORKING_SELECTION.json).
Official nonranking check `5090b83f1d63407b` passed 16 prerequisites and 12 shapes,
score 1.233633059x, 95% interval [1.215128563, 1.252137555]. Native paired target
latency fell 4.8226% against fusion; official target reference-normalized cross-run
latency fell 3.8251%, with at most 1.8323% unchanged-path normalized regression.
This supports a working engineering choice, not universal concurrency safety or
a proven aggregate statistical difference. No ranked entry was made.

A2-11 remains the immutable rollback source at
`reviews/attention-v2/mixed_fa3_decode_parallel_merge.py`, SHA256
`c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`.
To roll back, select that path/hash in the attention record; do not overwrite
either source. Fusion remains the direct experimental control. Projection
records and original checkout sources are unchanged by this selection.

The native app `ap-EY8FNKYqqFMGgZAatYZXOQ` is independently stopped/tasks0;
the official job is terminal and its serialized slot released. No new job is
started by these records. A next local plan covers the actual DP multi-chunk
lookahead/merge gap; same-address live-metadata replay is already covered.
Remaining attention quota was 2 checks after this admission; duplicate/no-op
checks do not follow from that quota. Artificial stage-credit ceilings are
removed; use existing free credits with live paid cap $0, existing credentials,
serial measurement slots and reviewed private pushes under the standing project
scope. No payment, credential renewal, public visibility or formal entry.

Earlier checkpoints below are historical and superseded for attention selection.

---

# Pro research completed; independent review saved

Read [the Pro research review](reviews/attention-v2/next-plan/PRO_RESEARCH_REVIEW.md)
and its `pro-request-status.json` receipt. The single approved research request
completed in the user's Mac Chrome, with Pro selected and Response complete
visible. No duplicate request or browser action is pending. The live conversation
is https://chatgpt.com/c/6ade40a4-9438-83e8-b546-958708e69fd0.

Pro recommends extending existing fused preparation to full decode B16 and full
verify B16/B32. Independent source review confirms this needs only a dispatch
change: the existing helper is already batch-generic. Five to seven microseconds
saved per complete call would model +0.84–1.19% overall; this is not measured.
Keep the natural L8/M64 WGMMA idea as a separate compiler-feasibility probe.
The earlier PLAN.md preserves our original ranking and exact score calculations.

No candidate was changed, GPU job launched, official check consumed or credential
changed by this research stage. Dedicated batched-full cache/replay validation
remains necessary. A2-11 remains selected and projection is unchanged. The last
conservative Modal ledger observed by this task is $1.919496218/$2; this research
consumed no GPU credit. Do not infer another task's live app state from this file.

The former upload denial was resolved with specific user confirmation. A later
stop preserved the attached unsent draft; subsequent explicit reauthorization
permitted exactly one send. See the receipt history for the earlier states.

Earlier checkpoints below are historical.

---

# A2-15 paired diagnostic complete; selected A2-11 unchanged

Read [PAIRED_RESULT.md](reviews/attention-v2/a215/PAIRED_RESULT.md).
App ap-DjbFQfnZXQjpr1Zgt3rtFK stopped at 2026-10-25 16:17:37 UTC;
a separate 16:17:56 UTC read verified stopped, zero tasks and no active apps.
Two B32/L1/full cases (scored-family page1 and page64/scale1/64 boundaries),
both sources, four graphs: 12 ABA checks plus 32 warmup/timing calls passed.
Eight ABA checks used the official oracle; four A returns matched exact bytes.
All checked immutable inputs and untouched cache rows. No concurrency/control/
full-suite coverage is claimed.

Target raw medians: A2-13 312.032 us, A2-15 295.904 us. Median of six reverse-
order pair ratios is 1.054731x, a 5.189% latency cut. All 12 measured comparisons
favored A2-15. This misses the predeclared10% target and is not an official score.
Holding other paths fixed models only +0.750% overall; selected A2-11 stays.

Frozen worker/protocol source commit: 5cb8db07702dd9a69bd632b2e425198c0dfc5e91.
Candidate hashes remain unchanged. No additional OKC check or GPU run.
The 16:18:59 UTC billing snapshot increased $0.07, with $0 paid. This exceeds
our time model, so conservative accounting uses $0.07: cumulative $1.919496218,
remaining $0.080503782. Reservation released only after verified shutdown.
No active/uncertain request. No further run fits the unchanged $0.14637 reserve;
this bounded Modal cycle ends without spending on another small-score trial.

The completed experiment is being committed and privately exported after review.
Use the verified remote SHA receipt for delivery status; do not infer push from
this header. Both earlier native evidence and all historical failures remain.

Earlier checkpoints below are historical.

---

# A2-15 native diagnostic complete; no active GPU work

Modal app `ap-TQMFJknnfH0UZbG8lOBcBe` stopped at 2026-10-25 15:52:43 UTC;
a separate 15:53:12 UTC read confirmed stopped, zero tasks and no active apps.
Four exact-source H200 specializations compiled and disassembled. All 25 raw
files were retrieved and hash-verified. No candidate kernel was executed;
correctness, graph/cache behavior and performance remain untested for A2-15.

See [NATIVE_RESULT.md](reviews/attention-v2/a215/NATIVE_RESULT.md) and the saved
native artifacts. The compiler produces page-load/use separation across part
of PV plus double-buffered K/V, rather than the literal before-QK source order.
Producer shared memory rises 12,544 to 20,480 bytes; registers remain 96, raw
spill field 0. Both versions lower quantized FP8 Q/K to FP16 MMA in this
specialization. Merge SASS is identical. No new score or selection change.

Executed payload commit: 31010ce6b69d54767dc322a53b0cf6894b03d955.
A2-15 candidate hash remains fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c.
Post-run local warning hardening is commit aaa801a; it was not uploaded.
Billing at 16:00:27 UTC shows $1.23 gross and $0 paid, an observed $0.03 rounded
increase. Keep the larger $0.061676416 model allowance: cumulative $1.849496218,
remaining $0.150503782 under the $2 stage ceiling. Prior accounting was retained.
No active reservation, uncertain request, new official check or further GPU run.

The next justified experiment is a full correctness/paired-timing check of the
unchanged A2-15 source against A2-13, retaining A2-11 as selected best. This
completed diagnostic does not itself authorize a new launch. The private export
will include reviewed source, artifact hashes and results; verify delivery from
the remote SHA receipt rather than assuming it from this status file.

The earlier checkpoints below are historical and superseded by this result.

---

# A2-15: native diagnostic blocked; official capability audit complete

Read reviews/attention-v2/a215/OFFICIAL_CAPABILITY_AUDIT.md and
NATIVE_ATTEMPT.md. Modal app-list failed `Token has expired` at 08:31:05 UTC;
no upload or new app, no GPU use, no credential renewal. No launch outcome is
uncertain. Historical conservative Modal cost stays $1.787819802. Existing
account credits/cap/apps could not be freshly verified.

OKC read-only access works. At 08:36:21 UTC, attention checks were 16/20 used
(4 left), projection 3/20 (17 left), scored 0/10 each, all 37 returned jobs
terminal. No new check was reserved or sent. The inspected documented CLI/API
and saved receipts expose timings/profiles/logs, not native generated artifacts.
A timing-only result could assess net benefit but not the intended page-load
scheduling mechanism. The bounded decision is to preserve quota and prepare
native comparison when valid access can be established separately.

A2-15 source commit a2c9a91b1dd347d11cb333b2f96b8973951cfa03 and both frozen
candidate hashes remain unchanged. Seven source/address and three diagnosis
tests passed previously; no native compilation, GPU correctness or performance
result exists for A2-15. The narrow diagnostic program is still unimplemented;
old wrappers package extra files and omit current compile arguments.

The reviewed A2-15 export is being delivered to the existing private
attention-fa3-sanitized branch. Verify its actual SHA from the push receipt or
remote before any retry; this file does not assert that a pending push worked.
Selected A2-11 and projection P039 remain unchanged. No active GPU job or
uncertain request is introduced by this phase.

The prior completed local checkpoint follows as historical context.

---

# A2-15 local diagnosis and prototype complete; no external operation pending

Read CURRENT_AUTHORIZATION.md. No new OKC/GPU/Modal/push/credential operation
was attempted or reserved in this phase. Native compilation and execution are
not available in this Mac environment (no Triton installation). Do not upload
or launch without a separate exact approval.

Candidate: reviews/attention-v2/mixed_fa3_page_lookahead.py, SHA256
fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c.
Parent is preserved A2-13, not rejected A2-14. Only full B32/L1 prefetches next
page IDs before current QK/PV, consumes them after PV and carries int64 rows.
Math, scalar denominator, K/V payload loads, merge/cache ownership, geometry and
all other paths remain the parent. Seven AST/address and three diagnosis tests
pass. Independent review found no concrete semantic defect; native scheduling,
resources, correctness and performance remain unverified.

A2-14 diagnosis: denominator state16→1024 FP32 values and full-chunk logical
multiplies2048→131072, with max reductions unchanged. These are source counts,
not measured instructions/registers. Producer profile311.1→422.0us; no saved
candidate-specific compiler artifacts were found. Do not claim proven spilling.

See reviews/attention-v2/a215/README.md, diagnosis.json, preflight.json and
NATIVE_DIAGNOSTIC_PLAN.md for evidence, commands and the falsification gate.
The bounded local phase has one candidate; do not start a broader rewrite or
parameter sweep. Selected A2-11 and projection remain unchanged. Latest actual
verified private remote tip from the prior phase:
73d9d1acfa21722415bb0957f3ec9197c75c5855 (attention-fa3-sanitized).
New A2-15 work is local and unpushed.

The prior completed A2-14 checkpoint follows; its check must never be repeated.

---

# A2-14 complete: reject performance regression

Job `6d6019056cbc4f81` is terminal DONE; do not resubmit. Official H200 passed
16 prerequisites + 12 timed shapes. Score 1.1687086418×, CI
[1.1511780122, 1.1862392714]. Target full decode B32 latency 316.412→422.966 µs,
normalized +33.9263%, versus a 20% reduction goal. Aggregate −3.4915% vs A2-13;
no unchanged control regressed >5%. Independent analysis confirms rejection.

Frozen source: reviews/attention-v2/mixed_fa3_deferred_denom.py, commit
2092957b1c8312b1fc90b43f64c3ed5eed8f4658, SHA256
bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d.
Full receipt, all 12 timings, six weighted loads, numerical coverage and
commands are in reviews/attention-v2/A2-14-RESULT.md and associated JSON/CSV.
Source bytes have not changed after dispatch. Exact-source ABA/concurrency,
untouched rows and B32 page64/non-unit-scale stress remain unverified. No
current-candidate register/spill metadata; do not invent a hardware cause.

Selected attention remains A2-11, mixed_fa3_decode_parallel_merge.py,
job cfa8a8670e394f80, 1.1876787561×. Promising A2-13 FP8 QK remains separately
preserved, job c2fb7b23433f4888, 1.2109908035×. Projection P039 is unchanged
in its separate worktree, job d5636ac8d1404abd, 1.1732389990×.

The one exact A2-14 check permission is consumed; no new candidate upload,
Modal, credentials, paid resource or ranked entry is authorized by it. The
reviewed A2-14 source/tests/related analysis/sanitized evidence private push
completed at 07:55:09 UTC, verified first-delivery SHA
16a8c48296884950e308d8217918444ccefedb99 on attention-fa3-sanitized.
A confirmation-only commit may advance that branch; the final task/ignored
receipt records its exact latest SHA. No additional candidate transfer is granted.
Push verification belongs to the task and ignored local push receipt; inspect
actual remote state after any interruption, never blindly retry. Never push
original attention-fa3-local history. No unrelated projection push.

Actual platform snapshot 2026-10-25 07:50:53 UTC: attention16/20 used4left,
projection3/20 used17left, scored0/10 each. Reset October26 00:00 UTC.
No active jobs among37returned; zero uncertain requests. These counts are a
timestamped observation, not a perpetual grant. Historical Modal apps were
stopped; no Modal action here. Conservative cost remains $1.787819802.
