# Attention architecture test

The BF16 custom full-verify route is rejected on performance. Its requested
strict M64 BF16 QK mechanism works and lowers static resources, but paired
whole-call timings are 61.68% slower than FA3 at B16 and 48.89% slower at B32.
All representative correctness/replay checks pass. All 24 measured comparisons
favor FA3. No official check or formal routing change is justified; A2-11 stays
selected. The scheduling hypothesis also lacks evidence for a CUDA prototype,
and the actual FA3 B32 verify kernel already uses a dynamic persistent scheduler.

## Source and isolation

Independent worktree: `attention-architecture-probe`, based on private remote
`attention-fa3-sanitized` commit `f67c58987cab6416080ef0dcb8f345f9700a95c7`.
Original checkout: `/Users/frank/Documents/Codex/2026-10-23/task`, initially clean
at `7290d1cf7638581f1b8c594d7fb2b424f3c2f139`. Root and attention
AGENTS/TASK instructions were read. No repository `.agents` skills were found.
An unrelated `next-plan/pro-request-status.json` changed externally during the
run; it was not edited, staged or reverted here. Original selected source and
original HEAD remain unchanged.

Exact source SHA256:

- A2-13: `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
- Selected A2-11: `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`.
- Compile worker: `b980e6229db9043ea795aa1cd26c110cf9326c0ec757cb89294f6465889b0c4c`.
- Runtime worker: `3818221cb069ed1d605c37871183aebc5cf711bdcf22fd28fcc6768b3dc120c7`.
- Runtime BF16 helper: `59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82`.

The existing A2-13 `QK_DOT` parameter permits compiling the requested probe
without changing candidate source. Q remains BF16→E4M3 before BF16 widening;
K widens from E4M3. M64/D64/BN64, 11 × 8192-token chunks, four warps, three
stages, strict QK accumulation, BF16 PV and FP32 softmax/partials are preserved.
L8 keeps its integrated final-CTA ticket merge. There is no A2-15 lookahead.
The runtime experimental helper is AST-equivalent to dormant `_e2_kernel_fn`
after reversing qualification, function renaming and its BF16 operand selector;
it does not mutate the original module or formal FA3 route.

Compile payload freeze: `34c3ee1bda2ffccf7958588114d5896d1286ba39`.
Runtime payload freeze: `7c544c918996739863f126771f4ebe8fcdc7f122`.
All new deliverable hashes are recorded in the final `manifest.json`; native
files have individual hashes in `native_artifacts/summary.json`.

## Native result

One H200 app compiled four records in 21.18 worker seconds. No candidate kernel
was executed in that phase. B16/B32 differ only in launch grid and share the
same cubin within each operand variant. All 25 native files were retrieved and
hash-verified; TTIR, TTGIR, LLVM, PTX, cubin and SASS are preserved.

| M64 variant | Historical and new-token QK | PV | Registers | Shared bytes | Raw driver spills |
|---|---|---|---:|---:|---:|
| E4M3 control | FP16 MMA | BF16 WGMMA | 172 | 33,024 | 0 |
| BF16 widening | BF16 WGMMA | BF16 WGMMA | 135 | 24,832 | 0 |

The two QK products are individually mapped through source locations in IR/PTX;
PV WGMMA alone is not used as proof. The source explicitly retains accumulation
bound zero. TTIR omits that attribute while showing BF16 operands/FP32 outputs;
omission alone is not independent proof. No local loads/stores were found in
PTX/SASS. WGMMA commit is closely followed by wait-group-zero, so useful QK/PV
pipeline overlap is not established. See `probe/native-audit.md`.

Installed environment: H200 capability 9.0, Triton 3.8.0, Torch 2.14.1+cu130,
CUDA 13.0, driver 610.57.04. Exact Triton Git revision is unavailable;
distribution METADATA/RECORD and compiler-source hashes are recorded. Static
resource reduction does not establish occupancy or end-to-end performance.
Raw compiler/disassembler whitespace is retained byte-for-byte for artifact
hash reproducibility; source whitespace checks exclude those immutable files.

## Dynamic scheduling result

The two primary vectors in `scheduling/results.json` are actual recorded H200
inputs from the historical paired run, not guessed CUDA RNG outputs. Four
CPU-generator samples are supplementary. For the two B32 vectors, 704 CTAs
contain 594/604 full 128-tile history tasks, 64 partial tasks and 46/36
empty-history tasks. Every empty-history task is final split 10 and still owns
new-token cache writes/attention; none can simply be deleted.

Under the idealized 132 identical-slot model, x-fastest FIFO reaches the fixed
full-task pigeonhole lower bound of 640 tile units. Reordering existing
8192-token tasks has zero modeled headroom. This remains true across all 40
recorded residency/prologue sensitivity cases. It is not a measured CTA trace.

Arbitrarily divisible work has about 4.9–5.5% ideal headroom in the zero-prologue
132-slot examples, with much larger bounds at other hypothetical residencies.
Actual residency and per-CTA service times are unknown. Assignment, descriptors,
gather, partial aggregation and merge costs are unmeasured. Existing partial
output is 2,973,696 bytes (~2.84 MiB); naive partials per 64-token tile exceed
300 MiB. Aggregation is necessary. This evidence does not justify a dynamic CUDA
prototype. See `scheduling/RESULT.md` for full assumptions and sensitivities.

The actual unchanged FA3 B32/L8/D64/page1 graph was profiled once outside the
paired timing interval. Its complete, untruncated kernel name explicitly
contains `flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true,
true, true, true, true, true>`. Length preparation and combine kernels are also
present. This verifies the scheduler type for that observed B32 call; it does
not identify boolean-template meanings, prove B16's exact specialization or
establish held-out performance. `paired/FA3_PROFILE.md` preserves the evidence.
The installed wrapper defaults remain `num_splits=0` and `pack_gqa=None`.

A new dynamic planner must consume live GPU lengths/page tables inside each
timed call and graph replay, without cross-layer/taskmap caching. H20/D128/page64
results or timings excluding assign are not Muse/D64/page1 performance evidence.

## Correctness, replay and complete-call timing

The one runtime job used the unchanged official oracle/tolerances. Three cases
cover scored-family B16/B32 full verify at80K/page1/coherent, plus B16 page64,
scale1/64, lengths0/1/63/64/65/8191/8192/8193, valid cache row0 and unused pages-1.
All six independently captured graphs replayed A→B→A with changing real lengths
and cache content at fixed addresses. All18 oracle replays passed; all6 return-A
checks matched exact output/new-KV bytes. Immutable inputs and untouched cache
rows were checked. All64 warmup/timing replays were additionally validated.

Only after all three cases passed did timing begin. Each family used four
warmup two-arm samples and12 measured two-arm samples, balanced into six
reverse-order pairs using the same input version. Complete `kernel_fn` graphs
include preparation, cache stores, ticket initialization, partials and merge.
Input/cache restore,256MiB L2 eviction, validation and calibration are outside
the event interval. Raw timings are primary; all samples are retained.

| Full verify family | Unchanged FA3 raw median µs | BF16 custom raw median µs | Custom slowdown from paired ratio |
|---|---:|---:|---:|
| B16/L8 |211.584|342.480|61.683%|
| B32/L8 |396.080|588.400|48.891%|

There is no confidence interval or official score claim. The observed losses
are far larger than sample variation, and every measured comparison favors
FA3. This does not compare BF16 custom against the FP8 custom route at runtime;
it rejects replacing the current FA3 route with this complete BF16 custom
implementation. Native mechanism success did not translate into competitive
end-to-end latency. See `paired/ANALYSIS.md`, raw `paired/result.json`, exact
payload hashes and independent analyzer admission checks.

## Guardrails and resources

The initial manual balance question was resolved after the user's direct reply
and a narrow parent authorization to inspect a separate Modal billing tab.
Supported Chrome extension reads verified $28.74 free credits, spend cap $0 and
paid spend $0 at 17:41:26 UTC. ChatGPT/Pro tabs were untouched. The native desktop
was locked; it was not unlocked or bypassed. No billing, payment, credential or
visibility setting changed. The artificial $2 stage ceiling is removed; actual
paid authorization remains $0. Existing credential configuration was used in
place via `MODAL_CONFIG_PATH`, without copying or renewing secrets.

Each job uses one H200, two CPU cores, 8 GiB RAM, max one container, no retries,
explicit file allowlists and independent hard deadlines/watchdogs. GPU runs are
serialized. The compile job stopped at 17:43:26 UTC and subsequent independent
CLI reads confirm stopped/tasks 0. Its initial stopping/tasks 1 observation is
preserved rather than overwritten. Rounded workspace usage increased $0.04;
detailed ephemeral usage increased $0.03393386, paid cost stayed $0. Conservative
time allowance is $0.06599165, below its $0.19663884 reservation. Billing is
workspace-level/rounded, not an exact per-app invoice. See `control/compile-receipt.json`.

The paired app `ap-Cox9fy63pE0bZocdwG9e0e` stopped at17:50:36UTC. A separate
17:51:51UTC CLI read confirmed every returned app stopped with tasks0. Both
controllers' initial immediate reads saw stopping/tasks1 and exited with their
shutdown flag false; these records are retained. Independent later reads
resolved teardown to terminal. Worker exit codes are0, with full native and
runtime results retrieved; there is no pending or uncertain resource/request.

Total observed workspace usage rose approximately$0.18 ($0.18011257 detailed),
with paid charges$0. Refreshed UI still shows$28.63credits and spend cap$0;
its header total$1.37lags CLI/ephemeral detail$1.44. This discrepancy is recorded,
not treated as a refund or exact per-app bill. Conservative accounting takes
the larger time/observed allowances per job: $0.20873 total (rounded up). Historical
conservative cumulative accounting becomes$2.12822152 (rounded up), with no artificial$2
ceiling and no active reservation. `COSTS.json` and the two reconciled run
receipts retain all inputs and qualifications.

Local validation passed5 compile-source guards,6 compile-controller guards,
7 runtime-worker protocol guards,6 runtime-controller guards and4 result-analyzer
negative admission checks. Scheduling arithmetic/hash/ownership checks and
native role-mapping/hash checks passed. These local checks supplement the82
validated GPU replays; they do not provide full-suite or concurrency coverage.

## Decision and next experiment

Stop this BF16 custom route and do not spend an official check on it. Keep
selected A2-11 and current FA3 full-verify routing. No new CUDA scheduling
prototype is justified by the current model. No formal contest entry or
visibility change occurred. Reviewed source, raw native artifacts and sanitized
results are intended for the existing private repository; delivery is established
only by the verified remote SHA receipt after push.

The next useful bounded experiment is a short actual CTA residency/service-time
trace of the current selected B32 full-decode path, compared with these real
length vectors. It should measure recoverable tail loss and include the cost
of any live per-call assignment/descriptor/aggregation before proposing a
planner. For full verify, start from the measured FA3 dynamic-persistent path;
another custom WGMMA compile pass is not enough evidence for replacement.
