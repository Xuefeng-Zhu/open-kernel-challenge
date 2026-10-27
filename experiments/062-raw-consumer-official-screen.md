# P062 — missing raw-consumer baseline, not a compression retry

Prepared 2026-10-26. Selected P046 remains unchanged at 1.1841399409×.
This is one native baseline ablation to close a concrete evidence gap identified
in P061. It does not retest the failed compressed candidate, add a codec, retile,
or claim a new optimization before measurement.

## Experiment card

- **Problem:** P043's compressed consumer passed correctness but lost badly;
  its exact raw BF16 consumer was never GPU tested. Decoder costs and an
  uncompetitive custom GEMM are confounded.
- **Change:** use the already committed raw-control source, differing from
  P043 only by `POOLER_COMPRESSED=False`. S8/BM16/BN64/BK64, four warps, two
  stages, weight preparation, profitability guard, split scratch, reducer and
  every other path are identical. No diagnostic logging is added to the file.
- **Primary question:** does this raw consumer remain competitive enough to
  justify investigating its decoder/layout? This is information gain, not a
  predicted score improvement.
- **Targets:** pooler M8/M16/M32 and prerequisite M7. The other 27 scored paths
  are controls. P046 uses a different, already successful full-K BMM there.
- **Coverage:** these three scored rows have mean within-load share 14.456%.
  A hypothetical 20% cut on P046's three targets models +3.024% score; even
  zero target time gives only 1.40515×. No such cut is predicted for this raw
  control. A broad future compression route needs a different coverage case.
- **Cost/risk:** same scratch write/read and original raw weights; the codec
  state remains prepared and resident to preserve the matched control. Native
  compilation, numerical reduction order and dispatch still need validation.
- **Cheap falsifiers:** exact source diff/hash, static preflight, full CPU
  oracle, fresh weights/activations, existing bit/address/fallback tests. All
  original tolerances remain unchanged. CPU tests do not execute Triton.

## Frozen source and comparisons

Source `candidates/projection_pooler_lossless_raw_control.py`, commit
`10baddbce8a690814cbf24452529b9836b404d51`, SHA256
`8ee5189831d7dcd209284b14a1ed907140d45ad9beba8c266122dc07e93a43de`.
It has not been modified for this screen. The plan/assessment gets a new
experiment commit; historical source and failures remain intact.

- Compressed P043: job `128378586ebe4730`, source SHA
  `9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837`.
- Selected P046: job `41ba778763a5433b`, source SHA
  `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.

For each job `r = SGLang / candidate`, so the normalized raw/P046 latency
ratio is `r_P046 / r_raw`. Report compressed/raw as `r_raw / r_compressed`,
plus both jobs' actual calibrated times and uncalibrated graph medians. The
historical receipts are separate runs, not new paired raw/compressed/P046
measurements. Normalization cannot eliminate input-, compiler-, allocation-,
cache- or clock-dependent differences. Preserve all raw samples and profiles.

## Predeclared interpretation

1. Require all 9 prerequisites and 30 scored cases to pass. A native failure
   yields no performance conclusion and no automatic retry.
2. Require `_pooler_codec_partial` and `_reduce` activity on all three targets.
   The raw arm retains that symbol; **source identity plus the false constant**
   establishes the raw branch. The profile establishes that the static guard
   did not silently use a library fallback.
3. Report all 27 controls. More than 5% movement or changed evaluator identity
   limits cross-run attribution. Do not discard outliers or infer a difference
   test from overlapping aggregate intervals.
4. Raw near P046 (no target more than 5% slower) supports obtaining exact compiler
   artifacts before a decoder change. A modest mixed outcome is inconclusive.
   If all targets exceed 1.30335× P046, deprioritize this consumer geometry:
   even the P061 optimistic storage-proportional zero-decode model cannot
   recover parity. That model is not a physical bound against all compression.
5. Raw faster than compressed but slower than P046 is mechanism evidence only.
   No outcome from this single cross-run screen automatically promotes it.
   Preserve P046; do not turn a failure into a tile sweep.

## Compiler/layout status

P043 source reconstructs BF16 immediately before `tl.dot` and has no decoded
global weight buffer. Its matrix transpose is a logical operand operation;
the physical lane layout, shared staging, vector widths, spills and issue
ordering are unknown. No exact P043 compiler artifact is present in saved
results. P051's related control is not identical and cannot supply P043's
register/shared-memory numbers.

The official TASK promises activity profiles and a bounded log tail. It does
not promise PTX/SASS export. This exact-source check will not manufacture such
evidence or read evaluator internals. Triton 3.8's owned `CompiledKernel.asm`
and warmup metadata can support a future diagnostic, but the Mac has no Triton
package/CUDA host, and existing Modal authentication was reported expired.
No package install, Modal call, credential action, or hidden compilation
instrumentation is part of P062. The full three-arm same-input protocol from
P061 remains unfulfilled by this narrower official-runner screen.

## Local results and commands

Mac arm64, Python 3.12.13, torch 2.14.1 CPU, CUDA unavailable:

```sh
python -B static_check.py --submission candidates/projection_pooler_lossless_raw_control.py
python -B check.py --device cpu --suite all --submission candidates/projection_pooler_lossless_raw_control.py --weight-seed 6262 --seed-offset 62000 --output results/router-p062-raw-control-cpu.json
python -B -m unittest tests.test_projection_weight_codec tests.test_projection_pooler_codec tests.test_projection_gap_model -v
python -B -m unittest tests.test_p062_assessment -v
```

Static preflight passed. All 39 fresh CPU cases passed; activations and weights
were unchanged. The 22 existing codec/score tests passed in 0.910 s; nine new
offline assessment error-path/ratio tests passed in 0.042 s. Independent review
found no admission blocker and prompted frozen historical identity checks and
explicit evaluator/control-drift tests, now included. Synthetic test
receipts are never persisted or represented as GPU evidence. NumPy-unavailable
warning is environmental; no installation was needed.

The initial sandbox read-only OKC requests failed DNS before reaching the
service. The approved network permission succeeded: login Xuefeng-Zhu, both
tracks 0/20 checks used, and no active jobs in the returned account list.
`whoami --json` returned human text, so the first wrapper's JSON parser failed;
the supported plain `whoami` read verified identity. This was not an auth failure.
Quota and job state must be refreshed immediately before admission. Reset is
00:00 UTC per the saved official contract, independent of user authorization.

Exact proposed command, after shared-slot and source/quota gates:

```sh
OKC_CLI=<existing-cli> OKC_GPU_SLOT=<coordinator-slot> python -B cycle2_check.py send P2-10-p043-raw-control candidates/projection_pooler_lossless_raw_control.py
```

The wrapper journals a reservation before its sole POST. An uncertain outcome
must be reconciled using existing jobs, never blindly resubmitted. Only mode
`check`, no paid resources, no formal entry. Assess a terminal sanitized receipt
with `python -B assess_p062.py`. Admission and outcome will be recorded below.

## Terminal result: diagnostic passed; no promotion

Plan commit `4025e73` froze the source, interpretation and offline assessment
before dispatch. Exactly one request was accepted at 2026-10-26 22:22:55 UTC:
`de8dcd1a4d5a43b2`. The server reports start 22:23:04 and finish 22:23:56;
the terminal read at 22:24:04 confirmed **9/9 prerequisites and 30/30 scored
cases passed**, score **1.1365058657×**, interval
**[1.1194582777, 1.1535534537]**. No formal entry was made.

The official evaluator code, file hashes and image digest match both historical
receipts. The three raw paths executed `_pooler_codec_partial` plus `_reduce`;
the source hash and `POOLER_COMPRESSED=False` establish the raw specialization.
This is actual native correctness and timing, unlike the earlier CPU models.

| Pooler M | P046 calibrated µs | P062 raw calibrated µs | P043 compressed calibrated µs | Normalized raw/P046 latency | Normalized compressed/raw latency |
|---|---:|---:|---:|---:|---:|
| 8 | 17.5593 | 23.0469 | 52.6389 | 1.30364 | 2.24020 |
| 16 | 18.0427 | 23.4695 | 51.9122 | 1.28527 | 2.22684 |
| 32 | 18.3095 | 26.1860 | 98.9527 | 1.41025 | 3.75679 |

Ratios above 1 mean the numerator is slower. The normalized columns use each
receipt's paired SGLang ratio, not division of the displayed cross-run µs.
Raw uncalibrated graph medians are 27.344/27.488/30.544 µs; P046's are
21.584/22.208/22.400 µs. Activity profiles report raw partial times
21.5/21.8/24.1 µs and reducer times 1.4/1.7/2.3 µs. These profile totals are
separate observations, not an additive decomposition of the paired calibrated
score. Even the raw partial's observed time exceeds P046's whole-call time;
removing only the reducer is therefore not a credible rescue from this evidence.

All 27 unchanged controls are retained. Raw/P046 normalized control latency
ranges from 0.94652 to 1.03024, median 0.99496. Router M128 is 5.35% faster
despite unchanged source. Relative to P043, unpooler M1 is 9.20% slower,
router M1 5.92% slower, and router M128 5.58% faster. Thus the predeclared
control-drift qualification applies; **there is no evaluator-identity change**.
All 60 candidate samples, 60 SGLang samples, 60 reported ratios and three
outside timing bursts per scored shape remain in the sanitized receipt.
None were discarded. Dedicated same-address A→B→A and overlapping-stream
coverage was not added by this official run, nor was all-bit GPU codec testing.

The six load ratios are decode 1: 1.27986660; decode 16: 1.04442771;
decode 32: 1.01011449; verify 1: 1.29482259; verify 16: 1.13123830;
verify 32: 1.08956220. `summarize_check.py` recomputes each load as
`sum(62 * SGLang time) / sum(62 * candidate time)` and verifies their geometric
mean equals the official score. This is not a 30-shape unweighted mean.

**Decision:** P046 stays selected at 1.1841399409×. The raw consumer misses the
near-baseline gate on all three shapes. The stronger all-target storage-screen
rejection does **not** trigger: M16 is below 1.30335, and M8 is only marginally
above it. Do not change that predeclared threshold after observing results.
This is a control-qualified mixed deficit, not proof that all compression is
unviable. It does establish that P043 cannot be explained only as a good GEMM
with a cheap decoder: the measured raw backend is already behind P046, while
the complete encoded branch is substantially slower still. The size of that
encoded-branch penalty is suggestive, not an isolated instruction-cost estimate.

## Next discriminating step and exact blocker

Do not submit unchanged P043 or tune its tile sizes. Before another compression
candidate, compare **these exact two source hashes**, same M16 inputs and launch
geometry, in an owned compiler-artifact diagnostic. Capture TTGIR/PTX/SASS and
register/shared-memory/spill metadata for `_pooler_codec_partial`, retaining
the reducer and P046 as controls. M16 has a complete BM16 tile and avoids M8's
masked rows and M32's additional M tile. Test whether encoding changes matrix
instruction family, operand-layout conversions, async-copy scheduling or
spills. A source-level integer-operation count cannot answer those questions.
If neither the arithmetic path nor compiler resources/ordering change
materially, the remaining hypothesis is decode/address/load instruction cost,
requiring controlled same-input timing; do not claim a measured hardware stall.
If an adverse change is found, only a single fix that preserves the raw path's
layout is worth a prototype. That fix must beat actual P046, not merely P043.
The three-target 20% improvement model remains only +3.024% overall, so a narrow
decoder repair is not by itself a competition-gap strategy.

No exact P043/P062 compiler artifacts were returned by this check. The official
contract promises profiles, not an arbitrary native-artifact export channel.
The Mac has no Triton/CUDA backend; existing Modal authentication was reported
expired and this turn did not call Modal or renew credentials. Thus the exact
native compiler comparison is **blocked by unavailable authorized native
diagnostic access**, not by an observed compiler failure or proof of spills.
There is no pending upload, denied action, uncertain request or active
Projection measurement. No further check is needed just to repeat this result.

Reproduction after the terminal reply is saved locally:

```sh
python -B summarize_check.py P2-10-p043-raw-control --ledger experiments/projection-cycle2-budget.json
python -B assess_p062.py
python -B -m unittest tests.test_p062_assessment -v
```

Evidence: `results/remote-P2-10-p043-raw-control.json` and `.csv`, all-30-shape
analysis `results/projection-p062-assessment.json`, original CPU receipt above,
and the durable request ledger. The post-run snapshot is **1/20 checks used,
19 remaining; 0/10 ranked runs used** for Projection, observed 22:24:04 UTC.
Reset remains 00:00 UTC. This task consumed one free off-board check and $0 paid;
no Modal app or new credential was created. The shared measurement slot is
terminal, so other coordinated work may proceed.
