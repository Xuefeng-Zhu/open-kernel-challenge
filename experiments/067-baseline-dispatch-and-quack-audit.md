# P067 — selected baseline audit and pinned Quack comparison

Decision: preserve P046 at **1.1841399409×**. No new performance candidate,
GPU launch, official check, dependency installation or credential action occurred.
Quack provides a concrete implementation reference, but neither its source nor
our saved traces establish a capability missing from P046. Do not start a
Stream-K or pipeline rewrite on that assumption.

## What the selected baseline actually proves

Selected source: `candidates/projection_pooler_column_batch2_hybrid.py`, SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Official job `41ba778763a5433b` passed 9 prerequisites and 30 scored cases.
The following audit checks immutable receipts and the six published P046 traces:

```bash
python -B projection_p067_audit.py --output results/projection-p067-audit.json
```

This is saved-data analysis, not a new measurement. It also reconstructs the
six-load score through P066: five family times, each weighted 62, per load;
geometric mean of six reference/candidate load ratios.

| Evidence | Established fact | Not established |
|---|---|---|
| P046 source, lines 272–388 | Small pooler M8/16/32 packs two output-column batches and calls full-K BMM into disjoint output views. Other pooler M>1 uses prepared transpose. QKV/unpooler M>1 and O use vendor matmul. Custom GEMV and router hi/lo routes remain. | Vendor tile/K mapping, persistence, buffer ownership or tactic ID. |
| Exact official profile | One reported kernel for small pooler; main + reducer for QKV M8/16/32/128, pooler M128 and unpooler M8. Large unpooler and M256 pooler/QKV have one main kernel. | A one-kernel call is not proof of no internal synchronization; a split symbol is not an exact work partition. |
| P049, exact P046 source | QKV128 main/reducer 14.527/3.327 us; unpooler16 one event, 17.952 us. | Collector saved symbols/durations only, not launch resources or binary identity. |
| P053, exact P046 pooler M8 | Grid [4,16,1], 384 threads, 168 registers/thread, 164,308 shared bytes; one vendor kernel. | Achieved occupancy, utilization, traffic, stalls or producer/consumer organization. |
| P053 nearby M7 | Main grid [2,64,1], 384 threads, 168 registers, 164,292 shared bytes; reducer [128,1,1], 512 threads, 32 registers, zero shared. Three same-stream traces show no main/reducer overlap, with 0.318–0.320 us gaps. | Inter-kernel order does not expose internal barriers or CTA assignment. |
| P044 / P059 | P044 resources belong to P039. P059's native split2 ID66/tile27/custom2 recovered matching visible main/reducer names and near-flat whole-call timing. | Neither proves P046's exact tactic, executable or scheduling. |

All phase durations above are instrumented observations, not calibrated official
times or a decomposition of event-timed medians. P053's `blocks per SM` is simply
grid size / 132 (64/132 for M8), not measured occupancy. Its zero occupancy
estimate is not used. The saved Modal environment recorded Python 3.12.3,
Torch 2.14.1+cu130, CUDA runtime 13.0, Triton 3.8.0 and H200 SM90. That is not
proof of the exact official image's installed Quack/CuTe dependencies.

Missing: exact P046 vendor SASS/PTX or module hash; internal tile/K mapping,
cluster scheduling, persistent work assignment and buffer protocol; achieved
occupancy, per-SM work distribution, memory bytes and stall samples. Offline
Triton compilation of our kernels cannot reconstruct a vendor executable.

## Quack versus P052/P053: a real structural distinction, not a speed claim

Public source pinned to **`77a750c0d9548d4a6e23c0f64062d61b3ea420ff`**,
committed 2026-10-15. File blob IDs and source URLs are in
`results/projection-p067-source-provenance.json`. No third-party code was run,
installed or copied into a candidate. Quack is Apache-2.0; future source reuse
must retain its notices and any underlying attribution.

| Mechanism | Our P052/P053 weight-first arm | Pinned Quack SM90 |
|---|---|---|
| Algebra/orientation | W × X-transpose, then stores transposed FP32 partial indices; fixed S8, BM16/BN64/BK64, four warps/two stages. | `swap_ab` swaps operands and output views. The same mathematics does not fix tile geometry, lane layout, pipeline or epilogue cost. |
| Loading and ownership | Ordinary Triton pointer loads lower to asynchronous copies. One 128-thread CTA, no explicit separate producer warpgroup. | One producer warpgroup alongside 1–3 MMA warpgroups; distinct AB-load, scheduler and optional C-load warps; TMA full/empty barriers and stage ownership. |
| Matrix overlap | Saved PTX already has WGMMA wait-group 1 in the loop and drain-to-zero at exit. P052 was not fully serial. | Its BF16 mainloop also retains one MMA group, but releases old AB buffers after completion while a separate producer fills later stages. Ping-pong optionally overlaps tile MMA and another group's epilogue. |
| Layout/epilogue | Compiler-selected shared operand layouts; FP32 partial epilogue uses a shared layout conversion, then global scratch; separate reducer. | Explicit staged swizzled A/B layouts and a staged register-to-shared-to-TMA output epilogue. Layout conversion is managed, not universally eliminated. |
| Assignment | Fixed output-tile × equal-K-split grid, no persistent work queue. | Static persistent scheduling is the SM90 configuration default. Dynamic semaphore scheduling is optional, not an inherent/default gain. Fixed Split-K divides K tiles with balanced remainder handling. |
| Small M | 64×16 weight-first MMA tile pads M8/M7; P053 raw partial remains slower than complete P046. | Lower-level tile constraints allow narrow N multiples of 16; the stock SM90 tuning list mainly uses large tiles (128/192/256 axes). Swapping alone does not prove an appropriate small-token configuration. |

Primary implementation links: [roles and pipeline](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_sm90.py#L75-L106),
[register budgets](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_sm90.py#L274-L325),
[actual BF16 wait/release loop](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_sm90.py#L1611-L1676),
[shared layouts](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_sm90.py#L2173-L2205),
[interface routing](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_interface.py#L630-L790),
[SM90 configurations](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm_config.py#L115-L159),
[actual static/dynamic assignment](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/tile_scheduler.py#L885-L916).

Quack's dynamic queue is distinct from work-centric Stream-K: its ordinary
split range still belongs to a fixed output tile. SERIAL split mode uses FP32
partials and ordered completion; PARALLEL uses arrival-order accumulation;
SEPARATE writes disjoint FP32 partials and launches a reducer. Those semantics
and costs must be named explicitly. Per-call semaphore/partial allocation is
visible in [gemm.py](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm.py#L292-L327)
and [the launch path](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm.py#L604-L627).
Reset/zeroing and finalization must remain inside the graph/timed invocation as
required; caching input-derived results is prohibited. A graph-safe invocation
must own its scratch. No concurrency or correctness claim is made for unexecuted Quack.

**Feasibility gate still open.** Pinned [pyproject](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/pyproject.toml#L5-L17)
requires `nvidia-cutlass-dsl>=4.7`, `apache-tvm-ffi>=0.1.6,<0.2`, Torch and einops;
its [README](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/README.md#L24-L28)
states CUDA 12.9+ and Python 3.12 (package metadata allows >=3.10).
Our historical Modal CUDA/Python versions satisfy those two version floors,
but neither exact judge DSL/FFI versions nor pinned Quack availability is verified.
Do not silently install/update dependencies or equate generic CuTe support with
this revision's compatibility. P055's documented native-compiler exception
remains valid historical evidence; today's web read of the rules returned no
extractable text, so no new rules-version equality is asserted.

## Two dispatch traps verified at fixed upstream commits

**CUTLASS:** at `91843b71d90737e0a9b86f7d9df213cff5a3526f` (v4.8.1),
[splits > 1](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp#L185-L253)
selects Split-K under Heuristic, and is invalid with explicitly requested StreamK.
Even splits=1 can resolve to data-parallel or ordinary Split-K in
[the actual selector](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/include/cutlass/gemm/kernel/tile_scheduler_params.h#L801-L925).
The class name is not proof of work-centric scheduling. Genuine Stream-K also
uses workspace/barriers. This updates P066's broad API description; the newer
public revision is not substituted for any installed judge version.

**DeepGEMM:** at `5081f8513af57e61696de4f4f04b0e707ffb503e`,
[BF16 dispatch](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/apis/gemm.hpp#L488-L534)
first uses cuBLASLt when nondeterministic mode, full-device runtime availability
and a supported alpha epilogue permit it. Identity supplies alpha=1; determinism
defaults false. Only after bypassing that route does SM90 use its native TMA
implementation. The wrapper asks for one heuristic, allows NONE/COMPUTE_TYPE
reduction, and uses a FAST_TF32 compute enum for non-blockscaled inputs. That
enum alone does not prove BF16 precision loss, but needs exact-input review.
Calling default DeepGEMM BF16 is not automatically a distinct native-kernel test.
See [wrapper](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/impls/smxx_cublaslt.hpp#L42-L105)
and [runtime workspace/availability](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/runtime/runtime.hpp#L24-L91).

## Compression conclusions are implementation-specific

P043 already fused reconstruction. P062's raw GPU control establishes that its
backend was weak; this does not logically rule out enough compression benefit
to overcome a raw deficit. P065 removed decoded-weight shared staging while
also changing lane mapping and losing ordinary Triton's automatic pipeline.
Its matched Gluon raw **GPU timing is still missing**. Do not label P065 as a
test of all register-consumer compression or all lossless formats.

Pinned ZipServ `6f8a209bfe46b0905f90d61b0796e352147ff041` reconstructs BF16 into
MMA registers, but **still double-buffers compressed streams in shared memory**
using async copies. [Its pipeline](https://github.com/HPMLL/ZipServ_ASPLOS26/blob/6f8a209bfe46b0905f90d61b0796e352147ff041/csrc/L_Kernel.cuh#L480-L539)
interleaves decode/computation; it is not the P065 direct-load pipeline. Its eager
cold-cache complete-GEMM benchmark includes the optional reducer but is not the
judge's paired calibrated graph method. Its GDDR-focused results do not transfer to H200.

Pinned Unweight `db0e6d3d4505cf71379f6b9b4567acf598bc186c` explicitly rebuilds
swizzled BF16 shared tiles before WGMMA; [reconstruction and handoff](https://github.com/cloudflareresearch/unweight-kernels/blob/db0e6d3d4505cf71379f6b9b4567acf598bc186c/src/reconstructive_matmul.cuh#L536-L556).
Its H100 measurements and cross-layer overlap cannot be transferred to one
isolated projection invocation. These sources establish different design tradeoffs,
not a reason to reopen the closed prototype without a new discriminating test.
ZipServ is Apache-2.0; Unweight is BSD-3-Clause with its own pinned dependencies.

## Actionable next decision and exact blocker

The minimum next observation is **the actual selected QKV128/pooler128 dispatch,
launch/binary identity and phase behavior**, with small pooler M8 as a known
single-kernel control. The bounded procedure and rejection conditions are in
`docs/P067_BASELINE_PROFILING_PROCEDURE.md`. Public Quack source can guide an
eventual controlled pipeline comparison; it cannot fill missing vendor evidence.

Existing Modal credential metadata expires **2026-10-26 03:39:46 UTC**; the
23:32:08 UTC local metadata check confirmed that past expiry without reading a
secret. No authentication call, renewal or launch was attempted. Prior NCU
attempts timed out with no metrics, so unchanged retry is also not a valid plan.
The official status/profile interface has no documented native artifact/counter
export; no check was spent merely to obtain another unchanged profile.

There is no newly admitted or uncertain job, and no new cleanup obligation.
Historical app-stop receipts remain historical; this audit did not query live
service state. Exact judge dependency inventory and functioning authorized H200
profiling access are the missing prerequisites. Current P046 remains selected.

Local verification: the audit reproduced the exact score and all 30 profiles,
verified six published trace/raw-receipt bindings, and ten existing score/phase
analysis tests passed. Independent review corrected the shared-memory field to
"reported" rather than "static", clarified optional counter access, and scoped
the six-path value model to pooler/QKV/unpooler (not router). No CPU check is
reported as native execution. No sensitive credential file was read or staged.

实验卡：这轮先确认基线已经做了什么。Quack 的独立加载线程组和显式缓冲交接，
确实区别于 P052/P053；但 P046 内部是否已有同类机制仍未知。当前不把类名、
WGMMA 指令或较弱对照上的提升当成进步。下一步只补基线的调度与阶段证据，
认证过期期间不续建凭证、不盲跑新方案。
