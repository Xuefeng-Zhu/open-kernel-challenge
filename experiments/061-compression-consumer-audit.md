# P061 — compression must earn a competitive consumer

Local methodology audit, 2026-10-26. **Do not relaunch unchanged P043 as an
optimization.** Preserve selected P046, official score 1.1841399409×, job
`41ba778763a5433b`. No candidate was changed, no GPU or official check was
launched, and no third-party implementation was executed. Attention retains
the coordinator's diagnostic priority.

The missing comparison is real: P043's matched raw consumer exists but has no
saved GPU result. That comparison can attribute failure within its custom
geometry. It cannot by itself establish a new mechanism capable of beating
P046. The papers do not supply that missing competitive evidence.

## Frozen sources and what they actually do

All paths are relative to this repository. Full hashes and receipt hashes are
in `results/projection-compression-audit.json`, reproduced by
`python -B projection_compression_audit.py` without importing any candidate.

| Arm | Source / SHA256 | Role |
|---|---|---|
| Selected P046 | `candidates/projection_pooler_column_batch2_hybrid.py` / `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e` | Actual working baseline, not SGLang |
| P043 compressed | `candidates/projection_pooler_lossless_codec.py` / `9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837` | Officially correct, performance rejected |
| P043 matched raw | `candidates/projection_pooler_lossless_raw_control.py` / `8ee5189831d7dcd209284b14a1ed907140d45ad9beba8c266122dc07e93a43de` | Exact one-constant change; GPU unmeasured |

P043 source lines 231–261 encode each row's K256 block as sign/fraction bytes,
packed four-bit exponent deltas, a base and an escape index. Blocks spanning
more than 15 exponents use full exponent bytes. Escape blocks still reserve
delta slots. Preparation at 404–426 retains original and transposed BF16
weights, transposes only metadata, and uses a static stored-size guard. The
32-byte header is an accounting allowance, not a GPU header tensor.
TASK.md explicitly allows static lossless weight compression in untimed
`prepare`; all activation-dependent work remains timed. Finite arithmetic and
original numerical tolerances remain required. Bit-exact NaN/Inf *storage*
tests do not imply valid arithmetic outputs for nonfinite inputs.

The timed path at 364–401 loads streams, reconstructs each uint16 BF16 pattern,
bitcasts it, and feeds `tl.dot(x, tl.trans(w), acc)`. **No decoded global weight
matrix is materialized.** The source has no explicit shared decoder buffer;
the register assignment, shared staging, conversion cost, vectorization and
overlap chosen by the compiler are unknown. `num_stages=2` does not prove
decode/compute overlap. “Fuse decoding” would repeat an existing feature.

At 489–500, pooler uses S8/BM16/BN64/BK64, four warps, two stages; the file's
global SPLITS=32 is for router. N4096/K8192 yields 512 CTAs at M8/M16 and 1024
at M32, with 16 K64 iterations per CTA. M8 masks half the row tile; M32 repeats
weight reconstruction for two row tiles. FP32 partials occupy 1/2/4 MiB and
their write/read traffic is 2/4/8 MiB. Those are logical sizes, not measured HBM.

The raw source differs only in `POOLER_COMPRESSED=False` at line 218. It keeps
the same preparation, stored-size guard, launch, scratch and reduction; only
the compile-time weight-load/decode branch changes. Both arms must use the
same compressible weights and prove that this branch ran. Incompressible
weights must separately exercise the truthful fallback, not count as codec
timings. Historical R06 is not this exact raw control.

P046 instead packs weights as [2,8192,2048], broadcasts X and uses full-K BMM
for disjoint output-column halves at canonical M8/M16/M32. Its profiles list
one vendor main kernel and no separate reducer. P043 also covers compatible
M2–32, including prerequisite M7. Thus P043 versus P046 changes the consumer,
partitioning, layouts and reduction as well as representation.

## What the failed run proves

P043 job `128378586ebe4730` passed all 9 prerequisites and 30 scored cases.
Its profiles prove all three codec branches executed. The generic assessment
decision string mentions missed execution gates, but the actual execution
gate is **true**; correctness/execution passed and performance failed.

| Pooler M | P046 calibrated us | P043 calibrated us | P043/P046 normalized latency | P043 own cut needed for parity |
|---|---:|---:|---:|---:|
| 8 | 17.559 | 52.639 | 2.920× | 65.76% |
| 16 | 18.043 | 51.912 | 2.862× | 65.06% |
| 32 | 18.310 | 98.953 | 5.298× | 81.12% |

Normalized comparisons divide each job's SGLang/candidate paired ratio; these
are separate-run comparisons, not new same-GPU pairs. P043 partial activity
51.2/50.4/97.0 us versus reducer 1.3/1.6/2.5 us cannot support a reducer-only
rescue. Activity samples are not calibrated medians. No P043 PTX/SASS,
register/spill/shared-memory data or hardware counters are saved. The cause
cannot be assigned to integer decoding, spills or HBM from these receipts.

Stored weights shrink by 23.2746–23.3043%, but original plus transposed weights
remain, increasing total pooler residency to about 177.1 MiB. Source address
arithmetic gives a different quantity: counting unique stream addresses per
K64 iteration, repeated metadata four times per K256 block, and masked escape
deltas yields 53,194,368–53,204,352 bytes per row tile, or 79.27–79.28% of raw
64 MiB. Counting literal per-lane operands instead gives 103.91% because two
adjacent lanes name the same nibble byte. The compiler may deduplicate/vectorize
those loads; neither number is an issued-transaction or DRAM measurement.
This distinction is independently checked with normal/escape address enumeration.

## Primary-source comparison, not transferred speedups

[ZipServ, arXiv v1 §§4.2–4.3 and 7](https://arxiv.org/html/2603.17435v1#S4)
aligns its 8×8 fragments and bitmap representation to the registers consumed
by `mma.m16n8k16`, with aligned staging and per-thread BF16 reconstruction.
This is more specific than merely fusing decode with GEMM. Its main small-batch
experiments use RTX4090/L40S; it explicitly reports cases below cuBLAS on
A100/H800. P043's generic row/K-block format and compiler-assigned dot layout
do not establish equivalent consumer alignment. The paper does not prove an
H200 gain or justify copying its reported kernel speedup into our score model.

[Unweight, technical report §§8 and 11](https://research.cloudflare.com/papers/unweight-2026.pdf)
uses TMA staging and reconstructs weights into swizzled shared memory before
Hopper WGMMA consumption. Its H100 SXM5 measurements nevertheless lose speed:
Table 11's gate kernel at batch 1 is 86.1 us cuBLAS versus 109.1/128.1 us for
the two reconstructive paths; end-to-end throughput falls 40.7% at batch 1
and 29.6% at 1024. Its capacity benefit and cross-layer scheduling are different
objectives from this isolated projection. H200 compatibility is not H200
performance validation. Neither external implementation was copied or run.

## Whole-score coverage rebased to P046

Six equal-weight load ratios are geometrically combined; all five family
times inside a load have weight 62. Every other timing is held fixed here.

| Scope | Scored paths | Mean within-load share | If target latency falls 10% / 20% | If it falls 23.2746% |
|---|---:|---:|---:|---:|
| Existing M8/16/32 pooler | 3 | 14.456% | +1.478% / +3.024% score | +3.546%, score 1.22613 |
| All pooler, hypothetical extension | 6 | 31.047% | +3.205% / +6.625% | +7.796% |
| Pooler + QKV + unpooler, hypothetical extension | 18 | 75.801% | +8.202% / +17.870% | +21.423% |

The last column deliberately assumes latency reduction equals stored-byte
reduction. It is an optimistic **conditional scenario**, not an attainable
forecast or physical upper bound. Compression ratios for other families are
not established by this pooler experiment. Even zero time for the three old
targets models only score 1.40515. That is a mathematical ceiling for changing
only those rows, not a kernel design target or a current leaderboard claim.

A useful screening model is `Tcompressed = Traw - (1-r)*Tweight + D`, where
`r` is the stored-byte ratio and D is incremental reconstruction cost. Neither
Tweight nor D is measured. Under the maximally generous assumptions Tweight=
Traw, D=0 and r=0.76725435, a raw consumer must be within 1.30335× P046 just
to reach parity, or 1.17301× to reach a 10% target cut. This is a reason to
deprioritize an uncompetitive raw consumer, not a proof that all compression
architectures are impossible. Cache/layout changes violate this simple model.

## Decision tree and the one discriminating protocol

1. **Now: do not admit another optimization run.** The old source already fuses
   reconstruction, and no new consumer mapping has been demonstrated. Preserve
   the exact raw control and independent bit/address/score checks. A new format
   must specify which operand shuffles, duplicated loads or reconstruction steps
   it eliminates; smaller storage alone is insufficient.
2. **If attribution will change the architectural decision**, the minimum native
   diagnostic is three arms on the same inputs: P046, exact raw P043, exact
   compressed P043. Start with M16 and M32 to expose repeated reconstruction;
   M8 is confirmation only if the first screen survives. Freeze all hashes,
   tiling, stages, precision, partitions and reducers. Capture generated loads,
   operand layout conversions, registers, shared memory and spills for both
   custom arms before interpreting timings. No counters are assumed available.
3. Check output against the original FP64 oracle before/after timing, sparse
   power-of-two rows, outliers/cancellation/zero, escape-heavy finite weights,
   nearby M7/M17/M33, changed prepared weights and same-address A→B→A graphs.
   Check immutable state and independent scratch. Storage tests include all
   BF16 patterns; nonfinite arithmetic is not an allowed-output test. Timing
   counts all decode, escape, layout, partial and merge work inside whole calls.
4. Predefine 6 discarded warmup rounds, then 24 balanced-order measured rounds
   per target on fresh data, cold L2 and raw graph times; retain every sample.
   Any calibration is separately reported and equally applied. Trace separately
   to avoid mixing profiler overhead with paired timings. Compute both ratios
   `Traw/Tcompressed` and `TP046/Tcompressed`; do not choose between controls.
5. If raw loses badly and compressed remains slower than P046, close this
   consumer geometry. If raw is competitive but compressed loses, allow one
   future decoder change only when codegen exposes a concrete extra cost. If
   compressed beats raw but not P046, record mechanism evidence, not progress.
   Only correctness plus repeatable whole-call wins against P046 justify a
   broader-family prototype and a later official verification. A near-noise
   effect is inconclusive; do not spend official quota to force a verdict.

This protocol is specified, **not scheduled or implemented as a new harness**.
The immediate recommendation is to keep compression off the optimization queue
until consumer codegen/competitive raw evidence distinguishes a new mechanism.
Do not start a new tile or codec sweep to fill that gap.

## Local validation and handoff

Reproduce arithmetic with `python -B projection_compression_audit.py`. It
verifies all three immutable sources, the exact one-constant control diff,
P046's score reconstruction and source address counts. CPU codec and score
tests use:

```sh
python -B -m unittest tests.test_projection_weight_codec tests.test_projection_pooler_codec tests.test_projection_gap_model -v
```

Result on macOS arm64, Python 3.12.13, torch 2.14.1 CPU: **22 tests passed in
0.681 s**. The independent normal/escape address enumeration, immutable source
hashes and score reconstruction also passed. The existing environment emitted
its NumPy-not-installed warning; no dependency was installed or needed.
Independent review found no blocker and separately recomputed score changes
from the official JSON benchmark rows, plus normal/escape/mixed/multiblock
address counts. A second read-only reviewer checked the cited primary papers.
These checks do not execute Triton or establish GPU correctness/performance.
No new app, job, credentials, quota use or spending. The last saved Projection
app is P059 `ap-vbrU0sPpoRjkmN7e7OgK4R`, independently stopped/tasks0 at
2026-10-26T01:10:24.310883 UTC; this audit does not claim a fresh global app
inventory while Attention's coordinator owns subsequent work. The coordinator
reported at 07:27 UTC that its read-only Modal app inventory failed because the
existing token expired. This task did not call Modal or renew credentials;
future native execution is gated by that prerequisite and scheduling. No GPU
diagnostic was attempted in this audit.

**中文实验卡：** 旧 P043 已经在矩阵乘法内部解码，重新“融合解码”不是新方案。
真正缺口是相同几何的 raw 对照和编译后的操作数布局证据。当前三条路径即便快
20%，总分模型也只增约 3.02%。本轮不重跑失败方案；先保留可区分“慢消费内核”
与“解码额外成本”的三方协议，只有能战胜 P046 的证据才值得扩大覆盖。
