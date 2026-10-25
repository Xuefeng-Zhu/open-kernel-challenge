# P046 — validate the three-path pooler composition

Status: official check complete; declared local-path gates pass, control drift remains unresolved. P046 is promising; P039 remains selected. Preparation sections below are historical. P045 remains a rejected broad candidate; its gates are not changed.

实验卡：P045 的 M8/M16/M32 路径分别快约 8.93%/6.04%/7.97%，但 M128 回退。
本实验仅保留这三个小 M 的输出列 BMM，其他形状全部使用 P039。验证实际组合
能否复现局部收益，不把约 1.24% 的条件模型当成已经实现的整体成绩。

## Parent, change and provenance

Code parent: P045 `candidates/projection_pooler_column_batch2.py`, source commit
`a3aba7ba7d70ef84bacd957ced35c0f0c9939c0b`, SHA256
`c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58`.
Its official job `b2c1e6c724144e87` passed 9 prerequisites and 30 scored shapes,
but failed the broad performance gates. Selected performance parent and fallback
remain P039, source `cdf5bc2`, job `d5636ac8d1404abd`, 1.173238999×.

Candidate `candidates/projection_pooler_column_batch2_hybrid.py` changes only the
P045 dispatch predicate: canonical BF16 contiguous pooler M in {8,16,32} uses
output-column batching. M128, M1, M256, nearby M7/9/31/33 and every other layer
use the unchanged P039 expressions. Other layout/dtype/prefix fallbacks remain
unchanged. No tile, shard count, precision, kernel math or packing change.

Static weights remain packed into [2,8192,2048], with one invocation-local output
written through disjoint view strides [2048,4096,1]. Full K remains in each GEMM.
The extra static weight storage is 64 MiB. The original and transposed pooler
weights are retained. No activation-derived state persists and no output cache
or work outside the timed call is introduced. Applicable Apache-2.0 text remains.

## Why perform this composition check

This is a new, narrower dispatch experiment, not a retroactive acceptance of
P045 or a shard/tile sweep. Its purpose is to reproduce the three observed local
wins while restoring the measured losing M128 route. A single whole-file check
also validates that all paths coexist correctly. There is no guarantee that
prior per-shape gains reproduce on a different run.

Using the observed P045 three-target normalized ratios while holding every other
P039 time fixed models factor **1.0123702024**, score **1.1877522029×**. This is
conditional arithmetic, not a verified combined score. It is below P045's original
1.5% aggregate gate and may be inside the official aggregate confidence interval.
The broad P045 candidate remains rejected regardless of P046's result.

## Gates declared before admission

- Original official correctness: all 9 prerequisite and 30 scored cases pass.
- Three targets M8/16/32: reference-normalized geometric-mean latency cut at
  least 5%, with no individual normalized latency regression. Define a target's
  ratio as `P039 paired speedup / P046 paired speedup`, and its cut as 1−ratio.
- The three-target-only frozen-P039 score model must improve at least 1%.
  This is a smaller-scope gate, not a claim of a statistically resolved aggregate win.
- Report all 27 controls, especially restored M128. Investigate any absolute
  normalized movement over 5%; do not attribute unchanged-path drift to code.
- Compare the three target directions with P045's independent receipt and review
  actual native profiles. Source dispatch plus timings are not proof of hardware
  traffic, occupancy, spills, or hidden library workspace.
- Do not promote from a higher aggregate point alone. Preserve P039 if target
  gains fail to reproduce, an individual target regresses, correctness fails,
  or control changes leave the interpretation unresolved. If shape gates pass
  with overlapping aggregate intervals, explicitly qualify the result as local
  path evidence with unresolved whole-score significance.

No tolerance change, duplicate source nonce, new credential, paid resource,
formal entry or simultaneous GPU measurement is part of this experiment.
Local correctness and independent source review precede one possible official
check after the coordinator releases the measurement slot.

## Local verification

Candidate SHA256:
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Test SHA256:
`a726e0cf8868f7951958fd90541efebeac6c893f47603022593c0e5d5f29d43e`.

Seven focused tests passed in 8.192 seconds, completion observed at
2026-10-25 21:44:34 UTC. Static checks, official snapshot hashes, and whitespace
checks passed. Independent review found no source-level blocker and confirmed
the executable AST differs from P045 only in the M dispatch predicate.

With the existing Python 3.12.13 / PyTorch 2.14.1 environment activated on the
Mac (Darwin arm64, CUDA unavailable), reproduce with:

```bash
python -B -m unittest discover -s tests -p test_projection_pooler_column_batch2_hybrid.py -v
python -B static_check.py --submission candidates/projection_pooler_column_batch2_hybrid.py
```

Actual full-K candidate coverage is **M8**, normal/outlier/sparse/zero and
same-address A→B→A, with state/input immutability. M16/M32 use K131 helper
models with real output layout; they are not full-K CUDA tests. Other checks
cover exact three-path dispatch, nearby fallbacks and unchanged-parent AST.
P045's exhaustive BF16 packing evidence applies to the identical preparation
function; it is inherited evidence, not a new native test. This complete file
has not executed on CUDA, and dedicated graph/concurrency checks remain unverified.

Gate formula: `1 - geomean(P039_speedup / P046_speedup) >= 0.05` across the
three targets, each ratio at most 1, and frozen-parent score factor at least 1.01.
No actual check request exists. The planned one-check command is:

```bash
python -B cycle2_check.py send P2-09-pooler-column-batch2-hybrid candidates/projection_pooler_column_batch2_hybrid.py
```

Run only after a fresh coordinator slot release, source commit and current
quota/job/rules preflight. The previous P045 request is terminal and must never
be duplicated. This report's arithmetic and local tests do not reserve quota.


## Compact experiment card before native admission

- **Evidence:** independent P045 official profiles replace the two-kernel
  split-K path with one main call on all three selected shapes; normalized
  cuts were 6.04–8.93%. M128 failed, so it is restored, not retuned.
- **Single hypothesis/change:** the same three kernels coexist with P039
  fallbacks and reproduce their direction on fresh official inputs. Only one
  metadata predicate changes from P045; no geometry or arithmetic change.
- **Optimistic latency/value:** a uniform 10% cut is about 1.99–2.02 µs per
  target and models +1.6059% overall. TASK's optimistic 14.0/14.1/14.1 µs
  floors imply 5.83–6.16 µs savings and +5.0833% overall if reached; the
  candidate is not expected to reach that hardware model. An impossible zero
  latency ceiling is +20.7712%. These ceilings are not performance evidence;
  the observed-component composition model remains only +1.2370%.
- **New costs/risks:** retain 64 MiB extra static packing; no new timed
  allocation beyond P045's invocation-local output. Library heuristics, cache
  state and cross-run drift may change. Selecting the three P045 winners
  creates selection bias; this fresh check is confirmation, not proof from
  the original selected observations alone.
- **Control:** P045's same geometry/math/preparation is unchanged on targets;
  P039 remains the frozen performance reference and all 27 other timed paths
  are unchanged. Report prior-P045 and new-P046 directions without claiming
  a direct same-device paired P039/P046 experiment.
- **Cheap falsifier:** seven local CPU/AST tests check the exact predicate,
  full-K M8 edge/ABA behavior and state immutability before spending a check.
  The earlier P045 native result covers these unchanged kernels but does not
  prove the complete new dispatch file. Full official original-oracle and
  fresh graph checks are still required.
- **Acceptance:** the pre-admission 5% target GM, no target regression and
  1% target-only model gates above are fixed. Report unresolved aggregate
  significance, all control drift and profile limits; do not retrospectively
  rescue P045 or lower these gates after a result.


The offline P046 assessor has ten passing focused tests for three targets,
27 controls, normalized ratio direction, exact source identity, six-load
scoring, P045 comparison and malformed receipts. It never automatically selects
an implementation. After a terminal passing receipt, reproduce with:

```bash
python3 -B assess_p046.py results/remote-P2-09-pooler-column-batch2-hybrid.json --p045 results/remote-P2-08-pooler-column-batch2.json --output results/router-p046-official-assessment.json
python3 -B -m unittest discover -s tests -p test_projection_p046_assess.py -v
```

These tools have not assessed an actual P046 result yet.


## Official result — completed 2026-10-25 21:58:38 UTC

One request was accepted at 21:57:30 UTC, job `41ba778763a5433b`, source commit
`fa56f2b3958b0c8d358c906a0b6cbb8e741a150e`, exact SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
The H200 job ran from 21:57:52 to 21:58:38, about 46 seconds. All **9 prerequisite
and 30 scored correctness cases passed**. No retry or duplicate admission.

Official score **1.184139941×**, interval **[1.166377842, 1.201902040]**.
P039 is **1.173238999×**; the point-score increase is **0.9291%**, with overlapping
intervals. All 30 rows and 60 raw samples/ratios per shape are preserved in
`results/remote-P2-09-pooler-column-batch2-hybrid.json` and its CSV.

| Pooler M | P039 calibrated µs | P046 calibrated µs | Reference-normalized latency cut |
|---:|---:|---:|---:|
| 8 | 20.162 | 17.559 | +11.523% |
| 16 | 19.932 | 18.043 | +8.317% |
| 32 | 20.237 | 18.310 | +8.719% |
| 128, restored control | 22.550 | 23.158 | +1.149% |

All three target directions reproduce P045's gains. The target geometric-mean
cut is **9.5311%**, each target improves, and the frozen-parent target-only model
factor is **1.0155331934** (+1.5533%). Thus every predeclared correctness and
numerical target gate passes. The original conditional P045 composition model
was +1.2370%; neither model is the observed complete score change.

### Actual route verification

The three targets use the same single-main symbols observed in P045:

- M8: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_NNT`, 17.5 µs activity.
- M16: `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_NNT`, 18.1 µs.
- M32: `nvjet_sm90_tst_64x16_64x16_1x2_h_bz_NNT`, 18.5 µs.

M128 restores `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_splitK_NNT` plus
`cublasLt::splitKreduce_kernel`, 20.9 + 3.0 µs activity. No separate copy is
listed in these traces. These are instrumented samples, not calibrated medians,
PTX/SASS, scratch allocation accounting or hardware-counter evidence.

### Controls prevent an unqualified promotion

All 27 controls are accounted for. Two unchanged M1 paths regress beyond 5%
after reference normalization:

| Control | P039 raw median µs | P046 raw median µs | Calibrated normalized regression |
|---|---:|---:|---:|
| O M1 | 10.848 | 10.976 | 5.621% |
| Unpooler M1 | 7.712 | 8.064 | 10.599% |

The raw paired reference-normalized increases are approximately **3.447%** and
**3.762%** before overhead subtraction. Calibration amplifies the change but
does not explain all of it. Unpooler's shift persists across both sample halves,
so discarding a single outlier would not resolve it. The raw-pair formula and
all derived pairs are saved in `results/router-p046-control-analysis.json`.
No adjusted or synthetic score replaces the official score.

Shared Triton helper ASTs and non-pooler `kernel_fn` computation exactly match
P039. Control profile symbols are identical: the O library kernel and `_gemv`.
Unpooler activity moves from 2.9 to 3.1 µs; the O activity sample moves opposite
to its replay benchmark. Single profile samples cannot establish the cause.
All P039/P045/P046 endpoint snapshots report SM 1980 MHz, memory 3201 MHz and
700 W limit. Temperatures differ (P039 42→41°C, P045 31→33°C, P046 35→38°C).
These are endpoint snapshots, not per-sample throttling, utilization or clock
traces. They neither prove nor exclude transient runtime/hardware effects.

**Decision: retain P046 as a promising, officially correct candidate; keep P039
selected.** This follows the predeclared unresolved-control clause, not interval
overlap alone. Independent review concurs. A direct same-device P039/P046 probe
of the three targets, O M1, unpooler M1 and restored M128 is the minimal useful
confirmation. Preserve balanced ordering, cold-L2 graphs, fresh inputs, raw
samples and calibration separately. Do not request a nonce official rerun.

The full file passed the official fresh-input graph protocol, with matching
P039 evaluator/image digests. Dedicated same-address graph A→B→A/concurrency
stress remains separately unverified for P046. CPU and inherited P045 evidence
are not relabeled as this full file's GPU stress tests.

Slot released at the 21:58:55 UTC terminal observation, before analysis/push.
Actual Projection quota then: **6/20 checks used, 14 left**, ranked **0/10**;
reset **00:00 UTC**. No Modal app, paid compute, credential action or formal
entry occurred. There is no active or uncertain Projection job.
