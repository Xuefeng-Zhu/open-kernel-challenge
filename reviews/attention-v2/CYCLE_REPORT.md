# Attention E0–E4 cycle

Scope: muse_attention_decode only. Six newly authorized off-board checks maximum,
with one reserved for final complete-source verification. Projection's remaining
8 checks and best 1.14864995× branch are unchanged. No scored entry, push, new
credentials, or payment/cap change.

| Experiment | Change / parent | Official result | Decision |
|---|---|---|---|
| E0 | Original v2 + only LOG2E constexpr repair | 5b0baa286659421a; passed; 0.7224690513×, CI [0.7116320155, 0.7373518839] | Preserved baseline |
| E1 (historical) | Full-only separate merge from E0 | No new official check; Modal weighted paired raw v2/merge 0.916402 | Reject overall regression; not stacked |
| E2 | E0, only full B16/32 fixed 8192-token historical chunks | 7edded086c8a47f2; passed; 0.7958818006×, CI [0.7839435736, 0.8078200276] | +10.16% score vs E0; E3 parent |
| E3 | E2, only full B16/32 history tl.range num_stages=3 | d4e3fc9a2c234d72 accepted; consult ledger | Await terminal evidence |
| E4 | Final complete file, chosen after E3 | Not dispatched | Final reserve retained |

Sources: E0 submission.py at05922b7; E1 separate_merge.py at3818c72; E2
fixed_chunks.py at2613417; E3 pipeline.py atf009696. Full hashes, exact commands,
parents, job IDs and terminal states are in attention-cycle-budget.json and receipts.
Every experiment has separate local source/result commits; no history fabricated.

## Coverage audit and additional replay evidence

Historical A03/A05: each app checked 16 prerequisites ×2 runners, and 12 scored
shapes ×2 runners ×8 distinct fresh seeds (2 declared warmups +6 steady samples),
224 checked replays/app. Three eager warmup calls precede capture. The same graph
and work buffers are reused, but no explicit A→B→A and no actual-length vector logs
were saved. Ten unique prerequisites use page64; two use scale1/64 (sliding B3 L8
page1; full B2 L1 page64). No joint page64+non-unit-scale+L8 case. All scored cases
page1/scale1. See COVERAGE_AUDIT.md for precise historical limitations.

A07 adds explicit same-address A1/B/A2 on one captured graph per runner/case,
12 checked replays total for E0 and E2. Actual A/B length vectors are logged and
changed, then restored; A2 output plus new K/V are bitwise equal to A1. Cases:
full B16 L8/80K/mixed/coherent/page1, and full B32 L8/80K/agentic/normal/page64/
scale1/64. Covers empty non-owner chunks. Official output and new-cache oracle
passed every replay; all untouched cache rows and non-cache inputs/config stayed
bytewise unchanged. A07 did not time kernels. It does not establish explicit ABA
for L1 or every shape. Original device-global tickets still limit simultaneous
calls/streams/graphs; sequential execution is the tested case.

A09 attempted the same replay/compiler probe for E2/E3 plus sliding control. Its
report was not returned: container push_outputs → BlobCreate received401. Existing
client status/billing reads work; the exact app is stopped/tasks0. No retry,
credential renewal or permission broadening occurred. Thus no E3 Modal correctness,
PTX-difference, register, spill or shared-memory claim is made. The independent
approved official check establishes its own evidence. This transfer failure is
preserved, not treated as a numerical pass or failure.

## Measurement and reproducibility

Official scores use six weighted loads (46×sliding +16×full per load), geometrically
combined. summarize_official.py reproduces E0/E2 scores exactly. All12 raw and
paired/overhead-adjusted timings, sample counts, errors and receipts remain saved.
Independent official candidate runs are not directly paired with each other;
0–2% differences without stronger repetition are inconclusive. >5% regressions
must remain explicit. E2's target normalized shape gains vs E0: decode B16+13.04%,
B32+38.59%, verify B16+11.88%, B32+15.52%. No >5% control regression; B1 sliding
decode -3.36% illustrates cross-run control noise. Both are slower than SGLang overall.

Modal public-image results are diagnostic, not leaderboard scores. Actual A07
H200 environment: Torch2.14.1+cu130, Triton3.8.0,132SM; exact frozen image/manifest
in its receipt. E0/E2 verify compiler metadata in A07:135/134 registers,0 driver-
reported spills,24832 B shared. Metadata cases are the two A07 cases, not all12
scored shapes. Official receipts preserve private evaluator/image hashes and GPU/
driver information; exact official Torch/Triton versions are not exposed there.

```sh
.venv/bin/python reviews/attention-v2/test_fixed_chunks.py
.venv/bin/python reviews/attention-v2/test_pipeline.py
.venv/bin/python reviews/attention-v2/test_modal_worker.py
.venv/bin/python reviews/attention-v2/check_cycle.py poll E3
.venv/bin/python reviews/attention-v2/summarize_official.py
```

Nine CPU/AST/harness tests pass; syntax and whitespace checks pass. CPU evidence is
separate from actual GPU checks. check_cycle.py records a source hash/commit before
one POST, reserves uncertain requests, queries daily quota and unchanged rules, and
keeps at least one final slot. No blind resubmissions.

## Cost and stopped resources

Modal cumulative conservative guard $1.708917914/$2, remaining $0.291082086;
includes all projection experiments and failed A09 app. Latest observed bill net$0,
free credits applied; delayed gross counters do not reduce the guard. Existing12h
auth expires2026-10-25 06:26:47UTC; no renewal authorized. Mac browser last verified
credit$28.89 and net spend cap$0. No new Modal runs planned for this cycle.

Exact apps stopped/tasks0: A07 ap-c7hDHYlZH7Ja3VAJNM4Iur; A09
ap-R6eHydPQwAqE8eg0p5lSCi. Older A03/A05 stopped evidence preserved. Prior projection
apps were stopped and current visible app inventory contains no active resource.

Pending: E3 terminal result, final selection and E4 verification; no other uncertain
submission. See CONTINUE.md and durable ledgers before resuming.
