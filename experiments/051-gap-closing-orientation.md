# P051 — gap-closing strategy and matched GEMM orientation diagnostic

Status: local research and executable diagnostic preparation. No P051 GPU,
Modal, official check, or new performance result. Selected P046 and rollback
P039 remain unchanged. P050 is parked; its source and tests are preserved.

实验卡：Projection 目前官方检查为 1.18414×，公开榜首为 1.6099×。需要的是
覆盖多个高权重路径的大幅改进，不是继续把约 1% 的总分微调作为主线。
第一步只检验一个编译机制：将旧内核的 `X @ W.T` 改为 `W @ X.T`，结果直接
写回相同布局，保持分块、split-K、精度和归约不变。先看是否真正改变 GPU
矩阵指令及资源；若没有，就否定这条依据，不以源码写了异步就宣称更快。

## Measured gap and conditional arithmetic

Public CLI observations on **2026-10-25 22:54:34 UTC**:

| Problem | Our selected off-board check | Public leader | Leader / ours | Uniform effective latency cut to match |
|---|---:|---|---:|---:|
| Projection | P046 1.1841399409× | colinlikescode 1.6099× | 1.35955× | 26.4464% |
| Attention | fusion + lookahead 1.2336330589× | pjmullin3 2.9097× | 2.35864× | 57.6027% |

Sources: `okc leaderboard muse_bf16_proj --json` and
`okc leaderboard muse_attention_decode --json`. Sanitized public snapshot and
raw-response hashes: `results/public-leader-gap-20261025.json`. No competitor
private job was queried, and neither our selected candidate has a ranked entry.
Leader scores are rounded to four decimals and provisional. Cross-run score
ratios are not same-device paired comparisons. Public pages:
[Projection](https://openkernelchallenge.dev/leaderboard?problem=muse_bf16_proj),
[Attention](https://openkernelchallenge.dev/leaderboard?problem=muse_attention_decode).

The Projection model uses all 30 P046 timings from job `41ba778763a5433b`,
CSV SHA256 `4538b39e8710fa88864127fea35e37598b361838ffcce9ca88d856894e292a35`.
It exactly reconstructs the official score. Each load sums its five shape times
with weight 62; the six reference/candidate load ratios are geometrically
combined. It does **not** use Attention's 46/16 weights.

| Load | Pooler share | QKV share | Unpooler share | Combined |
|---|---:|---:|---:|---:|
| decode 1 | 39.494% | 25.777% | 8.785% | 74.057% |
| decode 16 | 26.740% | 21.989% | 27.189% | 75.917% |
| decode 32 | 26.618% | 22.182% | 27.077% | 75.877% |
| verify 1 | 33.378% | 26.689% | 13.492% | 73.559% |
| verify 16 | 28.996% | 22.780% | 25.302% | 77.078% |
| verify 32 | 31.054% | 21.333% | 25.930% | 78.317% |

For a uniform latency cut `c` in a group occupying fraction `f_l` of load `l`,
the score multiplier is `exp(-sum(log(1-c*f_l))/6)`. Holding everything else
fixed, all three families need a **34.8864%** cut to match the observed leader;
limiting the change to M>1 raises this to **40.4808%**. QKV alone or unpooler
alone cannot close the gap even in an artificial zero-latency sensitivity.

| Hypothetical cut across all three families | Modeled score increase | Modeled score |
|---:|---:|---:|
| 10% | 8.2020% | 1.28126× |
| 20% | 17.8700% | 1.39575× |
| 30% | 29.4361% | 1.53270× |
| 40% | 43.5202% | 1.69948× |

If these three improve 30%, O plus router would still need a conditional
15.3135% cut; with a 20% improvement, the complement would need 46.6334%.
These are arithmetic requirements, not attainable forecasts.

Replacing just the three families with TASK's rounded ideal floors gives
**1.608190×**, only 0.1063% below the rounded leader. That difference is below
official uncertainty and **does not prove other families must improve**. The
useful conclusion is that a three-family-only strategy depends on approaching
the optimistic model almost completely. TASK's bandwidth/compute model is not
a strict physical bound: it omits scheduling, launch, scratch and rereads, and
does not establish measured HBM traffic or attainable implementation efficiency.

P050's prior native observation, a 3.7804% cut on QKV M128, models only +0.1443%
whole-score improvement if reproduced on that shape; applying it hypothetically
to all five QKV M>1 targets gives about +0.73%. Its frozen work remains useful
as a route-selection study, but is no longer the main competitive experiment.

Reproduction (standard library only):

```bash
python -B projection_gap_model.py --output results/router-p051-gap-model.json
python -B -m unittest tests.test_projection_gap_model -v
```

## Ranked mechanisms and capability decisions

1. **Own exact-BF16 GEMM dataflow covering multiple families.** The immediate
   experiment isolates compute orientation; a later matched explicit Hopper
   pipeline is warranted only if native lowering and correctness support it.
   The longer-term target is multi-shape instruction utilization, operand reuse
   and overlap, with M1 handled separately. A first narrow shape cannot establish
   the required 35–40% multi-family gain. Vendor kernels may already use equivalent
   orientation, so outperforming them is a high-risk research objective.
2. **Direct vendor algorithm selection: capability blocked, not disproven.**
   Inspected PyTorch v2.14.1 public matmul/mm/addmm/linear schemas have no tactic
   argument. Triton v3.8's bundled wrapper exposes matmul/gemm, requests one
   heuristic, and launches on stream 0; it does not provide a caller-selected
   algorithm or establish current-stream graph safety. FlashInfer's upstream
   internal runner is interesting, but exact judge availability, compatible AOT
   loading without subprocesses, reduction precision and workspace isolation
   remain unknown. P047 records this gate. Do not substitute ctypes, load_inline,
   installation or a global backend switch for missing supported capability.
3. **Lossless weight traffic reduction: potentially broad, currently poor evidence.**
   P043 reduced its encoded weight bytes about 23.3%, yet its changed custom
   matmul was much slower. Its same-geometry raw control was not GPU-tested, so
   decode overhead was not isolated. A decoder integrated into a demonstrably
   competitive raw kernel could be valuable; repeating P043's losing geometry
   is not justified. Full representation, metadata, escape/decode cost and
   arbitrary BF16 values must count. Lossless packing is explicitly permitted
   in untimed `prepare`; no activation-derived persistent state is permitted.

The user's external research suggestions—cluster split-K merge, warp
specialization and router epilogue fusion—remain hypotheses. Their supplied
P039-based single-digit modeled score effects do not independently close this
gap. No external research software was executed here.

Primary API sources: [Torch schemas](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/native_functions.yaml),
[Triton bindings](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/triton_nvidia.cc#L331),
[one-heuristic implementation](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/include/cublas_instance.h#L165),
[FlashInfer public BF16 interface](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_bf16.html).
The absence statement is limited to these inspected interfaces, not every
possible installed library. Exact judge-image inventory remains unresolved.

## Minimum executable experiment: orientation before pipeline

Mechanism parent: our historical R06 `_partial`,
`candidates/pooler_triton_split8.py`. Performance/selection parent: P046. The
diagnostic is **not a replacement submission** and does not alter P046.

Fixed controls: logical output tile 16 tokens × 64 columns, K tile 64, split-K 8,
four warps, two stages, original BF16 inputs, FP32 accumulation, identical K
partition and order, partial storage `[split, token, column]`, same reducer and
BF16 output. Only the dot orientation and equivalent direct partial-store
indexing change:

```text
control:      accumulator[token, column] = X[token, k] @ W[column, k].T
orientation: accumulator[column, token] = W[column, k] @ X[token, k].T
```

There is no full activation/output transpose, packed-data change, new split
choice, persistent scheduler, precision change or added staging in this test.
The diagnostic accepts M1..16 and masks N/K tails; it does not pretend to cover
M32/128/256. Input-derived scratch is invocation-local.

The pinned compiler provides a specific falsifiable basis. Its
[MMA support predicate](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Analysis/Utility.cpp#L1209)
requires the first result CTA dimension to be divisible by 64 and the second by
8 for MMA v3, along with type, K and warp conditions. R06's 16×64 fails that
dimension condition; the reversed 64×16 satisfies it. On SM90 the
[selection pass](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L43)
tries v3 before v2. This is compiler eligibility, not proof of emitted WGMMA,
useful overlap, occupancy, fewer HBM bytes, or a faster call.

This differs from M36's already-WGMMA 64×32 full-K QKV kernel and P038's
changed TMA geometry. Those failures show that an instruction name or fewer
launches alone is inadequate. Historical M34 native pooler artifacts used
32×128 / 32×32 tiles and mma.sync/cp.async; they are not R06 PTX. Exact R06 PTX
was not saved. P051 compiles both orientations anew instead of substituting
another kernel's evidence.

First gate: source-bound PTX/TTGIR and compiled register, raw spill and shared
memory metadata for both matched kernels. Inspect matrix/copy instructions,
wait placement and layout/store conversions; static counts are not runtime
counts. If the predicted lowering distinction does not occur, reject this
specific compiler rationale. If it occurs, original-oracle/fresh-input and
same-address graph replay checks are required before timing. A codegen pass
alone never promotes the candidate.

Only after that gate should a same-geometry synchronous versus explicit
TMA/WGMMA ring-buffer comparison be implemented. A proposed first full-K
unpooler M16 design would use physical 64×16×64 tiles; a three-buffer BF16
operand ring accounts for about 30 KiB plus barriers/compiler overhead, with
4 KiB of FP32 accumulator values. Those are source sizes, not compiled resources.
Gluon/Hopper API availability in the exact judge is still unverified. This ring
buffer is not implemented in P051.

Accepting an eventual architecture requires original correctness, no hidden
copies, preserved nearby-shape behavior, cold-L2 paired **whole-call** timing
against both matched control and P046, retained tails, and broader multi-load
confirmation. No arbitrary register-count threshold substitutes for timing.
No instruction-based performance benefit is assumed. The first diagnostic has
no official-score acceptance gate because it is not a performance experiment.

## Provenance, eligibility and remaining work

Saved official TASK authorizes normal BF16 computation, metadata dispatch,
lossless static weight preparation and graph replay. P051 uses standard Triton
JIT as existing accepted candidates do; it adds no C++ extension, child process,
network call or harness introspection inside a candidate. Live `/rules` could
not be read by the web text tool during this audit; saved TASK is the cited
contract, not a claimed new live legal interpretation.

The implementation is our own narrow variation of repository R06. No competitor
or tutorial kernel was copied or executed. Existing repository attribution is
preserved. Future copied Triton source would require its MIT notice. Relevant
upstream examples establish primitives, not drop-in BF16 implementations:
[Hopper WGMMA](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py),
[explicit persistence](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/07-persistence.py).

Native compilation, emitted instructions/resources, GPU oracle/replay coverage,
concurrent calls and performance all remain unverified for P051. No GPU slot
has been used. Next action is a bounded native compile comparison after review
and serialized slot coordination, not an official quota-consuming timing run.

## Local implementation and review checkpoint

`projection_p051_orientation.py` is the executable diagnostic. Its original
control/reducer ASTs exactly match R06; 13 standard-library tests cover indexing,
empty splits, tails, integer algebra, frozen source hashes and native-mode
guards. The prepared correctness mode has 70 oracle comparisons across five
cases, including sequential same-address graph A/B/A, but has not run. Review
found and fixed the K=1 fixture producing identical A/B values before any native
measurement; it now requires distinct bit patterns. This is a fixture correction,
not a kernel algorithm change.

The compile-only launcher uploads six files: the diagnostic, its historical R06
source reference and four license/notice files. It does not upload the whole
repository, credentials, oracle or benchmark harness. One H200, 75-second
internal / 90-second child / 105-second function bounds, 45-second startup,
150-second app watchdog and no retry. Resource-based conservative reserve is
$0.278423 free credits. Account observation on the user's Mac and read-only CLI
check showed $27.87 free, $0 paid, $0 net spend cap and all apps stopped. No cap or
credential was changed.

```bash
python -B -m unittest tests.test_projection_gap_model tests.test_projection_p051_orientation tests.test_projection_p051_modal -v
python -B projection_p051_modal.py self-test --control-root /path/to/control-workspace
# Coordinator only, after fresh budget/slot checks and committed-byte review:
python -B projection_p051_modal.py launch --frozen-head <40-character-commit> --control-root /path/to/control-workspace
```

The compiler probe's result is deliberately separate from numerical correctness
and timing. Its exact frozen commit and native outcome will be recorded in a
separate result commit; no result is assumed here.

Final coordinator preflight: **34/34 local tests passed in 0.262 s** on Python
3.12.13; four no-network launcher guards and `git diff --check` passed. Exact
source/test hashes are in `results/router-p051-cpu.json`. Review also fixed
timeout partial-artifact retention. The new test initially hit a Python 3.12
pathlib mock error; its test factory was corrected before this passing rerun.
Worker SHA256: `445ab7d44e36cdceafdf1ecdbe19d7fafa247d8033ab580ab38f74b98f3533db`.
Launcher SHA256: `961345c72351eb929e2ea88268d0e4b35fda7f64045527d417c0662af6f4a9e3`.
