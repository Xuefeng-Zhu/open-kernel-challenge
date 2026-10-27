# P053 — distinguish partial-kernel cost from reducer and launch cost

Status: diagnostic completed. All 598 current oracle comparisons passed; the
predeclared reducer-only rescue gate failed. App stopped/tasks0 and measurement
slot released. No official check or new score. P046 remains selected at official
1.1841399409×, with P039 preserved as rollback. The following plan and its gates
were committed before execution; terminal findings are appended below.

实验卡：P052 证明了指令变化和正确性，却没有得到足够速度。下一步不改
tile，也不凭 WGMMA 名字继续重写。用完全相同的计算主体，分别测“写出
split-K 部分结果”和“完整调用”，再用独立时间线确认归约与间隙；同时保留
P046 对照。若只省掉归约仍追不上 P046，就停止这条补救路线。

## Existing evidence, with provenance

P052 timing receipt SHA256:
`2ae054ec40d6f39a119dfe0c97fd330dfde09e315ba51d45b0cffb6cc71cd30f`.
Its 390 GPU oracle checks passed. Paired M8 weight-first latency was 21.3177%
higher than P046; matching P046 requires a **17.5718% reduction in the candidate's
own latency**, not 21.3177%. M7 was 4.8785% slower. Matched-control cuts were
only 2.2448%/1.7574%, nine of twelve pairs each. Keep all tails and the rejected
selection decision; the new diagnostic cannot retroactively change those gates.

`projection_p053_analysis.py` reproduces the following source model and verifies
the immutable receipt and PTX. Reproduction is CPU-only:

```bash
python -B projection_p053_analysis.py --output results/router-p053-local-analysis.json
python -B -m unittest tests.test_projection_p053_analysis -v
```

| Fixed source quantity | Pooler M8 | Pooler M7 |
|---|---:|---:|
| Partial CTAs / K iterations per CTA | 512 / 16 | 512 / 16 |
| Useful floating-point operations | 536,870,912 | 469,762,048 |
| Padded tensor operations / useful | 2.000× | 2.286× |
| Logical valid weight reads | 64 MiB | 64 MiB |
| Logical valid activation reads across output tiles | 8 MiB | 7 MiB |
| Unique activation storage | 128 KiB | 112 KiB |
| FP32 partial write / reducer read | 1 MiB / 1 MiB | 0.875 MiB / 0.875 MiB |
| Partial epilogue shared read + write | 4 MiB | 4 MiB |

These are source-address counts and padded arithmetic, **not measured HBM bytes,
transactions, cache misses or achieved throughput**. Both orientations preserve
the same valid global elements. Their lane-to-address mapping can still change
transaction efficiency, which has not been measured.

The M8 executable PTX text matches P051 after source-location/debug data is
removed. Both have ten static async data-copy sites, four commits and two waits;
counting the entire `cp.async` prefix incorrectly gives sixteen data copies.
Control has two vector-four shared stores/loads in the partial epilogue;
weight-first has four vector-two stores/loads. Both move 32 bytes per thread in
each direction, but instruction/address work differs. Static instruction-line
counts are 191 versus 213, not dynamic cost. Register/shared figures remain
42/10,240 bytes versus 36/20,480 bytes at M8, with zero raw driver local allocation.
No achieved occupancy or spill-traffic inference follows.

P052 has **no phase trace**. P044's pooler M8 main/reducer observation
17.919/2.272 µs belongs to a vendor kernel in P039, not this Triton reducer.
P048 has no activity evidence. P049 removed a QKV reducer while changing the
main kernel, so it cannot supply the missing causal reduction-removal result.
The source and trace differences are why another arbitrary tile choice is not
the next experiment.

## One discriminating test

Freeze the exact P051/P052 kernel bodies and launch geometry, the P046 source,
original BF16/FP32 precision, split partition and reducer. Test only M8 and its
nearby M7 prerequisite. Each graph owns its scratch and output. The same fresh
X and immutable original/prepared weights feed all comparisons.

Four balanced comparison series per M:

1. Token-first partial-only versus weight-first partial-only.
2. Token-first whole call versus weight-first whole call.
3. Actual P046 whole call versus weight-first partial-only.
4. Actual P046 whole call versus weight-first whole call.

“Partial-only” includes the full original partial kernel, including its shared
layout conversion and global FP32 scratch store. It is **not pure matrix time**.
After both members of a timed pair, run the original reducer outside timing to
validate each partial-only output against the original oracle. Do not run oracle
or reducer work between the timed members, and do not reuse an output before it
has been checked. Full calls remain unchanged.

Use six predefined warmup rounds, 24 measured rounds / twelve reversed pairs,
rotating/reversed comparison order, initialized CUDA events and 256 MiB L2
eviction before every timed replay. Retain every raw sample. Before and after,
check A→B→A, sparse and zero inputs for all five graphs and exact immutable input
and state. The earlier five-case/tail coverage is inherited from hash-bound P052;
the new screen does not claim it was re-executed. The current timing protocol
plans 580 oracle checks, including 220 partial-only outputs reduced outside
timing. No overhead constant is guessed or subtracted.

Only after timing completes, collect a separate Torch/CUPTI activity subprocess:
three fresh single-replay profiles for each of the three whole paths and each M
(18 traces). Oracle validation happens outside each profiler context. Record
actual kernel symbols, intervals, gaps and available resource metadata; absent
fields remain missing. Custom partial/reducer symbols must be identifiable.
Do not assume a vendor P046 path has a particular split or number of kernels.
Instrumentation is explanatory, never an acceptance speedup, hardware counter
or substitute for the uninstrumented paired samples. No NCU retry is proposed.

## Predeclared interpretation and limits

- If weight-first partial-only has ≥10% latency reduction and ≥10/12 paired
  wins over token-first while whole-call improvement stays <5%, investigate
  dilution by the separate reducer, launch or graph costs. This does not isolate
  the partial epilogue, which remains inside both partial-only calls.
- If partial-only improves <5%, orientation alone has insufficient observed
  main-kernel benefit. Do not scale this same design into a broad rewrite.
- A reducer-removal follow-up needs weight-first partial-only at least 5%
  faster than actual P046 whole-call with ≥10/12 wins in both cases. If even
  that optimistic diagnostic lacks headroom, reject reducer-only rescue of
  this fixed route. This is not a strict physical bound: changing fusion or
  scheduling can change cache state and execution costs.
- Near-threshold or contradictory samples are inconclusive. A passing diagnostic
  justifies a specific next design, not promotion or a whole-problem score.

Do not subtract medians of separate graphs and call the difference “reducer
latency.” Use separate profiler kernel intervals, retaining profiler overhead
and variation. No test here independently identifies memory transaction costs,
shared bank conflicts, matrix utilization or achieved copy overlap. These remain
explicit limitations rather than reasons to invent a bottleneck.

The immediate whole-score value is zero: this is causal evidence, not new code
for submission. A resulting mainloop design must later cover several high-weight
families; P051's model requires roughly 34.9% cuts across pooler/QKV/unpooler to
match the observed public leader. Simply eliminating one small reducer cannot
meet that strategic requirement.

## Execution readiness and operational boundary

Worker/controller are prepared; independent worker and controller reviews found
no concrete blocker. Native behavior is unverified.
One ephemeral H200 uses the existing image/authentication, verified free credits
and paid $0 only. Timing and activity have separate 90-second internal deadlines
and 105-second hard bounds; function 230 seconds, startup 45 seconds, watchdog
275 seconds. The 305-second cost reserve is $0.45788342 at the known rate floor,
increased if live rates require it. This is a reserve, not incurred cost.

The launcher permits only 16 committed payload files and 23 named artifacts,
including 18 traces. Activity requires the timing receipt's source identity and
all 580 passing checks; a negative performance result still permits explanatory
activity. No retries or persistent app. Fresh credits/cap/authentication/stopped
apps and coordinator slot must be verified at launch. A historical terminal slot
record is not a new grant. No official check is proposed.

Final local verification and source hashes are recorded below before admission.


Local verification at 2026-10-25T23:51:32.457123+00:00: 24 worker protocol tests, 17 launcher tests,
4 offline analysis tests and 20 P054 inventory fixtures passed (65 total,
0.259 seconds); four local payload guards also passed. These execute no CUDA,
Modal SDK or external operation. Initial worker testing had one false-positive
AST guard that mistook the original oracle helper `safe.check_output` for a
subprocess call; the guard was corrected, without changing kernel behavior.
The explicit two-case follow-up gate was also made machine-readable before any
measurement. Raw verification is `results/router-p053-local-verification.json`.

```bash
python -B -m unittest tests.test_projection_p053_phase_screen tests.test_projection_p053_modal tests.test_projection_p053_analysis tests.test_flashinfer_capability_inventory
python -B projection_p053_modal.py self-test --control-root /PATH/TO/LOCAL/CONTROL
```

Worker SHA256: `e494b809981de64afde229b12bee43035ec408fce85dee85f314c5bb457ee7db`.
Launcher SHA256: `047a790324ae4961b897f16aaff55b2e90aed3c524cb24eccc62a48a86c49a8f`.
The reviewed allowlist is 16 files, 366,448 bytes at this preparation point.
The actual launch requires an exact committed HEAD and revalidates every byte.


## Terminal result — 2026-10-25 23:56 UTC

Frozen launch commit `e6804b0d7909a6401396d2ef8a854967477fcdf8` contains the
reviewed P053 source from `8ee2fa7`. App `ap-UShS6clh30RqRtKHHcoB4A` was admitted
at 23:54:50.811138 UTC. Timing took 30.9143 s; separate activity took 20.5335 s.
Worker wall time was 51.4524 s and client wall time 69.2823 s, including lifecycle
and result transport. These are not measured allocated-GPU seconds.

All 580 timing oracle checks and 18 activity checks passed. Original tolerances,
sparse/outlier/zero fixtures, fresh inputs, same-address A→B→A and immutable
weights were retained. Earlier P052 prerequisite coverage was inherited, not
rerun. Concurrent calls and the complete official 9+30 suite were not tested.
An independent reviewer reproduced all eight paired summaries and verified
all source/receipt/18 trace bindings. The Mac performed analysis, not CUDA.

The table reports `1 - 1 / median(opponent/candidate paired ratio)`. Positive
means weight-first reduces latency; negative means it is slower. Twelve reversed
pairs use all 24 measured samples. Six warmup rounds were predefined/excluded.

| M | Partial vs matched control | Whole vs matched control | Partial vs P046 whole | Whole vs P046 |
|---|---:|---:|---:|---:|
| 8 | +1.4593%, 9/12 wins | +0.3135%, 6/12 | −14.5968%, 1/12 | −22.5155%, 2/12 |
| 7 | −0.9310%, 6/12 | −1.4756%, 6/12 | −0.6417%, 6/12 | −4.5991%, 5/12 |

**Decision: stop reducer-only rescue of this fixed mainloop.** At M8, even the
incomplete partial-only computation is 14.6% slower than the actual P046 whole
call. It needs a materially different main-kernel design, not merely a cheaper
final reduction. M7 provides no robust contrary evidence. This rejects the
specific tested route; it does not prove every fused or asynchronous design loses.
All tails remain, including candidate M8 whole 70.624 µs and M7 partial 67.584 µs.

Separate instrumented medians, three single replays per row:

| M/path | Main/partial µs | Reducer µs | Actual observed sequence |
|---|---:|---:|---|
| 8 control | 21.789 | 1.814 | custom partial → custom reducer |
| 8 weight-first | 19.957 | 1.759 | custom partial → custom reducer |
| 8 P046 | 17.394 | none | one vendor kernel |
| 7 control | 21.376 | 1.728 | custom partial → custom reducer |
| 7 weight-first | 19.872 | 1.792 | custom partial → custom reducer |
| 7 P046 | 17.892 | 2.208 | vendor main → vendor reducer |

These traces explain actual sequencing; their apparently larger main-kernel
improvement is not a replacement for uninstrumented paired statistics. No graph
median subtraction, hardware-counter inference or achieved-overlap claim is made.
No visible copy events occurred; this does not prove absence of hidden work.

Resource metadata in actual traces agrees with the earlier custom compilation:
M8 control 42 registers/10,240 shared bytes; weight-first 36/20,480; M7 weight-first
34/20,480; custom reducer 20/2,048. Both custom partial grids are [64,8,1] with
128 threads. P046 M8 has grid [4,16,1], 384 threads, 168 registers and 164,308
shared bytes. Its M7 main uses [2,64,1], same threads/registers and 164,292 shared
bytes. Geometry and resource allocation are observed metadata, not measured
occupancy, bandwidth, transaction efficiency or stall attribution. Profiler
occupancy estimates were deliberately excluded from conclusions.

## Reproduction, shutdown and costs

```bash
python -B projection_p053_modal.py launch --frozen-head e6804b0d7909a6401396d2ef8a854967477fcdf8 --control-root /PATH/TO/LOCAL/CONTROL
```

This is the historical single launch, **not an instruction to launch again**.
The durable record rejects duplicates. Source/controller hashes and the exact
16-file payload are in `experiments/projection-p053-phase-modal.json`.

An offline publication/audit command is:

```bash
python -B projection_p053_result.py --raw local-notes/projection-p053-phase-artifacts --dest results/remote-P053-phase --ledger experiments/projection-p053-phase-modal.json --output results/router-p053-assessment.json
```

It refuses to overwrite a publication. All 23 returned artifacts are retained;
published trace copies remove only top-level container hostname and trace ID.
Every event, timing and resource field is preserved. Assessment maps original and
published SHA256 values; the activity receipt's hashes refer to original raw
traces, retained in ignored local storage. Timing/activity receipts are unchanged.

The explicit stop succeeded. The first immediate postcheck still saw
`stopping.../tasks1`, so the launcher correctly exited 1 and required reconciliation.
A separate read at **23:56:28.401787 UTC** verified the exact app `stopped/tasks0`
and all workspace apps stopped/tasks0; its recorded stop time is 23:55:58 UTC.
The shared slot was released. No retry, duplicate invocation or repeated stop.

Paid usage remains $0. Delayed workspace gross delta at 23:59:15 UTC was
$0.07830934; it is not a settled per-app invoice. The $0.45788342 reserve remains
in the conservative ledger. No official quota was consumed.

No new performance candidate is justified by this result. The next high-information
prerequisite is P054's bounded metadata inspection for a genuinely precompiled
vendor-tactic route; it remains unexecuted and is not proof of judge eligibility.
The prior GitHub push denial remains unresolved; completed results stay local.
