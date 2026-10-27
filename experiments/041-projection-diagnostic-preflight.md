# P041 — profile the selected Projection paths before another rewrite

2026-10-25. Local preparation only. No new candidate algorithm, GPU launch,
official check, credential action or performance result in this record.
The selected P039 source remains unchanged.

## Experiment card

**Problem:** pooler, QKV and unpooler account for 32.6156%, 22.5420% and
20.9258% of the mean within-load time fractions, respectively. Most of those
paths still use library GEMMs. We lack their exact generated instructions and
resources. Failed custom kernels do not explain the library's remaining cost.

**Question:** does the selected library implementation show a repeatable cache
sensitivity and a specific split/reduction or scheduling cost across several
loads, large enough to justify a shared data-reuse/pipeline change? This is a
diagnostic hypothesis; it does not assert that HBM or stalls are the bottleneck.

**Change:** none to `kernel_fn` or `prepare`. Profile fixed P039 and the preserved
official SGLang baseline on five preselected cases. Measure whole-call cold
graph replay separately from instrumented activity and warm-cache diagnostics.

**Decision:** admit one new implementation only after this evidence suggests
a distinct mechanism, not another arbitrary tile/split sweep. A useful pooler
target is at least 2% modeled aggregate value: approximately 7.56% lower latency
across all five m>1 pooler loads. This is a value threshold, not a predicted win.

## Frozen parent and verified selection

- Source: `candidates/projection_router_m1_fp32_gemv.py`.
- Source commit: `cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`.
- SHA256: `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
- Official job: `d5636ac8d1404abd`, 9/9 prerequisites and 30/30 scored cases pass.
- Score: 1.173238998969285×, interval [1.1556404139847458, 1.1908375839538243].

P040's higher 1.175871327 point is not selected. Its 6.5077% O-m1 normalized
improvement missed the predeclared 15% target. P039's 30.1058% router-m1 target
reduction is concrete, but unchanged QKV-m1/unpooler-m1 controls also shifted;
the entire aggregate point increase cannot be assigned to that change. Marginal
CI overlap is not a difference test; independent paired candidate evidence is
missing. Original sources and failed experiments remain untouched.

README.md, BEST.md and STATUS.md have been corrected from the immutable selection
and receipts. Earlier scores and dated budget/resource statements remain
explicitly historical, not current account observations.

## Reproducible score and compulsory-data analysis

`projection_p041_plan.py` reads the frozen sanitized CSV/JSON, extracts dimensions
from the official reference AST without executing it, checks the selected source
hash and reconstructs all six loads:

`load = 62 × sum(five family times); score = geomean(reference_load / our_load)`.

It exactly reproduces 1.173238998969285×. Attention's 46/16 weighting is inapplicable.
The complete 30 rows, six loads, models, activity reports and evidence hashes are
in `results/router-p041-plan-cpu.json`.

| Held-fixed scenario | Modeled aggregate gain from a 10% target cut |
|---|---:|
| Pooler, all five m>1 loads | 2.6638% |
| QKV, all five m>1 loads | 1.8717% |
| Unpooler, all five m>1 loads | 2.0046% |
| Pooler m256 alone | 0.5382% |

These are conditional sensitivity calculations, not achieved gains or likelihoods.

| Representative | Official adjusted time, µs | Unique weight bytes | Dense GFLOP | FLOP / unique byte | Official ideal floor, µs |
|---|---:|---:|---:|---:|---:|
| Pooler m8 | 20.1617 | 64 MiB | 0.537 | 7.98 | 14.022 |
| Pooler m128 | 22.5503 | 64 MiB | 8.590 | 122.27 | 14.636 |
| Pooler m256 | 30.1783 | 64 MiB | 17.180 | 234.06 | 17.371 |
| QKV m128 | 17.2094 | 36 MiB | 4.832 | 119.49 | 8.424 |
| Unpooler m16 | 18.4869 | 64 MiB | 1.074 | 15.91 | 14.063 |

BF16 compulsory bytes are `2NK + 2MK + 2MN`; arithmetic is `2MNK`. Each weight
can ideally serve M rows and each input value N outputs, but the actual reuse,
repeated requests, split scratch and cache residency are unmeasured. The official
floor uses 4.8 TB/s and 989 TFLOP/s ideals and excludes launch cost. Neither that
floor nor FLOP/byte establishes achieved bandwidth. Unpooler at m≤8 uses N1024;
the selected m16 case exercises the N8192 boundary's full-width path.

## Exact evidence available and missing

Saved P039 profiles show split-K plus reduction for pooler/QKV m8/16/32/128;
pooler/QKV m256 use a single GEMM. Full-width unpooler m16/32/128/256 also uses
one GEMM. Pooler m>1 is the lossless pretransposed layout; QKV/unpooler retain
their original library layouts. Symbol names and instrumented activity durations
exist, but P039 library PTX/SASS/register/shared/spill/counter data do not.

M34's 95/45 registers and 40,960/16,384 shared bytes belong to rejected BM32
pooler prototypes. M36's 46 registers, zero raw spills and WGMMA belong to its
failed QKV full-K prototype. They cannot be assigned to P039 or P038.

Avoid repeating failed shortcuts:

- QKV transpose had no robust aggregate win and regressed m256.
- Broad unpooler transpose was aggregate parity; its small m128 gain is not a
  demonstrated multi-load mechanism.
- Narrower pooler tiles increased CTA count but remained much slower.
- QKV m128 full-K was 55.104 versus 21.888 µs, 0/15 paired wins, despite WGMMA.
- P038 changed geometry/backend together, and its ordinary-copy control was
  never GPU-tested. Its loss does not isolate a TMA effect.

The coordinator reports that ordinary Torch/CUPTI activity succeeds, while two
bounded Nsight Compute runs—including one `dram__bytes_read.sum` metric—timed
out after connecting to the process. No metric and no permission-denial evidence
were returned. **Do not repeat that method for P041 or call counters available.**

## Native protocol prepared; not executed

The local `projection_p041_probe.py` worker is intended for the existing approved
H200 image/tools after the coordinator releases the serialized slot. It has no
Modal provisioning or account operations. Capture its exact source hash and the
complete allowlisted payload before upload. Use only the selected P039, official
baseline/oracle/checks, required local harness modules and their license notices.
Never upload the repository, ignored logs, auth files or unrelated candidates.

1. **Identity and correctness.** Refuse source mismatch/non-H200. Record Torch,
   Triton, CUDA, device and precision flags, source/reference hashes and actual
   kernel symbols. Validate fixed-address A→B→A graphs against the official
   FP64 oracle, including immutable inputs and original weights. Keep ordinary
   and outlier inputs and official column probes. These five cases are diagnostic
   coverage, not a replacement for all 9+30 or independent concurrency tests.
2. **Timing.** Predefine warmup and reverse-balanced candidate/reference order;
   retain every gross CUDA-event whole-call sample. Read 256 MiB before each
   cold replay. Do not subtract a guessed fixed graph cost or call this an
   official score. A future candidate needs independent repeated sessions before
   accepting a small gain; unexplained >5% control shifts require investigation.
3. **Cache sensitivity.** A separate condition immediately replays the identical
   working set before timing without eviction. Warm samples are never scored or
   used for acceptance. Record actual exposed L2 capacity when available; 36/64
   MiB weight footprints do not guarantee complete residency. A warm improvement
   indicates cache sensitivity, not measured DRAM traffic or a causal stall type.
4. **Activity/instructions.** Profile separately from timing. Save actual symbol,
   launch metadata and resource information only where the installed tools expose
   them. Bind extracted binaries/PTX/SASS to their hashes. If library instructions
   or function attributes cannot be obtained, mark them absent. Never substitute
   a rejected Triton kernel's resource data or retry restricted/timeout counters.

The worker's monotonic checks bound host progress but cannot interrupt a stuck
CUDA call. The parent launcher must therefore enforce a hard function timeout
and one ephemeral H200 container, no retries/detach/deployment, then verify stopped
and zero tasks. Before launch, freshly verify existing auth, sufficient free
credits and net-paid-$0 limit. This local phase does not assert those observations.
The historical $2 artificial stage cap has been removed by the user; that does
not authorize paid usage, new credentials or bypassing the shared GPU slot.

## Acceptance and next implementation gate

The diagnostic succeeds by establishing trustworthy exact-source evidence,
including a negative result; it does not need a faster kernel. Cache sensitivity,
activity structure and generated instructions can eliminate explanations but
cannot replace missing bandwidth/occupancy/stall counters. If they still do not
support a concrete multi-load mechanism, report that limit instead of rewriting
an optimized library from a roofline estimate.

For any subsequently justified candidate, change one mechanism, preserve all
precision requirements and ordinary fallback semantics, and require all 9+30
cases plus changed-dispatch nearby/strided, sparse-column and fresh/replay cases.
Measure paired cold graph calls against P039; retain all samples and evaluate
the six-load effect, not only the target's best sample. Keep P039 unless the
predeclared target and correctness gates are met. No candidate has passed these
new gates yet.

## Local commands and state

From this worktree, use the existing Python interpreter (`$PY` below), without
installing dependencies:

```sh
$PY -B projection_p041_plan.py --output results/router-p041-plan-cpu.json
$PY -B -m unittest tests.test_projection_p041_plan -v
```

The six offline analysis tests passed. They cover exact score reconstruction,
selection identity, weights, per-load fractions, multi-load sensitivity,
unpooler width boundary and source-mismatch refusal. Nine mock/static worker
tests also passed. The worker has correctness/timing/activity modes, six declared
warmup rounds and 24 reverse-balanced measured rounds, an optional immediate
warm replay control, and explicit source/device/timeout guards. ABA uses normal
A and outlier B. These local checks do not execute Triton, FA3 or CUDA.

**Launch state:** no Modal app was admitted. The coordinator released the GPU
slot after attention app shutdown, but the user's subsequent instruction changed
the workflow to a direct official check. This diagnostic remains unexecuted;
do not start it automatically. The separate unchanged-P039 official request is
recorded in experiment 042, not counted as a new optimization or this diagnostic.
