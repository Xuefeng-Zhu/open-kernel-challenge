# M64N8 transpose: offline mechanism gate passed

The pinned CPU-only compiler emitted **both reversed QK and PV as native
M64N8 WGMMA**. Independent PTX/SASS dataflow review passed the declared
precision, cross-warp reduction, publication/lifetime and zero-spill/local
gates for this **one unbuffered historical tile**. This resolves a compiler
expressibility question; it is not a GPU correctness or performance result.

Selected1.233633 source `3da22a…` and A2-11 rollback remain unchanged.
No official check, GPU or Modal app was used by this probe.

## What was compiled

Q is post-norm/RoPE BF16 `[8,64]`, rounded to E4M3 then widened to BF16.
Encoded E4M3 K/V `[64,64]` widen to BF16. The two operations are:

`K[64,64] @ Q.T[64,8] -> scores[64,8]`

`V.T[64,64] @ P[64,8] -> raw_attention[64,8]`.

Scores, max/exp2/sum across64 keys and accumulator state remain FP32; P is
rounded to BF16 before PV, and scalar K/V scale placement is explicit.
Both operations request `max_num_imprecise_acc=0`. There is no new-token tail,
online previous-tile carry, page gather, norm/RoPE implementation, cache append,
split/merge, producer dispatch, launch or timing in this source.

The distinction from earlier experiments is the native8-query N dimension,
not padded64-query M, not scalar SIMT attention and not a modification to FA3.
It does not establish that the selected baseline lacks asynchronous staging.

| Frozen input | SHA256 |
|---|---|
| Owned `kernel.py` | `118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033` |
| Offline helper | `e7a489a899d7eae222250ca546d00bfefbb3435cf4ad25c8a79c96d0a099fc37` |
| Existing public Triton3.8 ARM64 wheel | `a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50` |
| Generated PTX | `822c2b004ffb9066ecec394e3c6448ed68b9f26bc7effe13f123b1ed97a77ecd` |
| Generated SASS | `3000d4eaef7b3f4bba348c84336d7aaf991518c591000a25865dfde787515e87` |
| Locally retained cubin | `449d38c546da620b172d6ba50782fd79ed0801022527d4023b5e5712c0155ac7` |

Seven CPU source/model/protocol tests and actual preparation smoke passed
before compilation. Source and cached API review found no precompile blocker.
Preparation hashes remain separate from generated artifacts.

## Actual generated evidence

| Observation | Result | Interpretation |
|---|---:|---|
| PTX `wgmma.mma_async...m64n8k16...f32.bf16.bf16` sites |8|Four QK plus four PV sites, without query-N padding |
| SASS HGMMA sites |8|Corresponding native instructions, not executed counts |
| `mma.sync` sites |0|No matrix-family fallback in this primitive |
| ptxas registers |32|This tiny primitive only; not comparable as a reduction from the full producer's96 |
| Stack / spill store / spill load bytes |0 /0 /0|Static assembler evidence |
| Cubin LOCAL field / PTX local accesses |0 /0|No observed local-memory traffic sites |
| Compiler dynamic shared-memory request |18,432 B|K8KiB + V8KiB + Q1KiB + P1KiB |
| cuobjdump SHARED field |1,024 B|Retained separately; not a replacement for dynamic launch request |

The [artifact review](ARTIFACT_REVIEW.md) verifies all seven generated-file
hashes and distinguishes the QK and PV instruction groups. It traces the
FP8 query roundtrip and BF16 K/V/P operands, and checks complete FP32 reductions
over all four warps. PTX/SASS contain the shared publication fence/barrier
before PV and waits before consuming accumulators. No overwrite or barrier
blocker was found for this one-tile source.

All TTGIR, LLVM IR, PTX, SASS, assembler/resource logs and the compiler manifest
are retained under `evidence/`. The cubin remains in the ignored local payload
directory with its published hash. No generated window was deleted to change
the conclusion. Static counts do not establish runtime throughput, stalls,
actual CTA residency, memory traffic or beneficial overlap.

## Credential-free execution and closure

The setup reuses Projection's independently published
[offline SM90 guide](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/c9c0907/docs/OFFLINE_SM90_COMPILATION.md)
and the already cached official wheel/image, without changing its sources or
containers. No dependency was downloaded or installed on the Mac host.
Only the disposable container installed the verified wheel into temporary RAM
storage, with `--no-index --no-deps` and no network.

The existing Python image digest is
`sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70`.
The Attention-only container `okc-attention-m64n8-offline` used Linux ARM64,
two CPUs,4GiB,128 pids, dropped capabilities, no-new-privileges, read-only root
and input mounts, temporary package/cache storage and one writable artifact
directory. No repository, home, credential or Docker socket was mounted.
The explicit target was `GPUTarget("cuda",90,32)`, with four warps/three stages.
There was no active-driver binder, CUDA availability patch, kernel handle,
device load, launch or NVIDIA device node.

Python3.12.15, Triton3.8.0, bundled ptxas12.9.86 and cuobjdump13.1.80 produced
`sm_90a` PTX/cubin. Tool and API source hashes are in the compiler manifest.
Host compile plus disassembly took about0.764s; this is **not kernel latency**.
A matching version or selected Python API hashes do not establish the judge's
compiler/binary identity. No offline cubin was executed on H200.

The container exited0 and its exact-name Docker listing was empty afterward.
OrbStack was started because it was stopped. This task did not stop the shared
engine or any other container. A later read observed the engine **Stopped**;
Docker was consequently unavailable, and it was not restarted for this report.
Exact Attention-container absence had already been verified while it was running.
The invocation and cleanup record is `evidence/execution-receipt.json`.
Paid spending, credential actions and official-quota use are all0.

## Next testable candidate mechanism

The proposed coverage is the four **currently custom decode** paths: sliding
B1/B16/B32 and full B32. If all four complete calls were10% faster with other
paths fixed, the selected-score model would improve about3.03%;50% would
model about20.20%. These are coverage scenarios, not forecasts or a complete
plan to reach the leaderboard leader.

The next gate is a source-bound **full producer integration compile**, before
any free official check. Preserve original Q/K normalization and quantization,
live page-table/length reads, split/chunk/grid policy, strict FP32 online
max/sum/carry, last-split new-token ownership, original tail semantics and
partial/merge ABI. The N16 new-token tail is not covered by this probe and
must not be silently expanded or omitted. Preserve or explicitly audit the
selected producer's existing asynchronous K/V buffering and wait lifetimes.
Gluon does not inherit ordinary Triton's automatic layout/software-pipeline
passes; the same `num_stages=3` is not evidence of matching buffering.

Compile against the actual selected control under the same pinned offline
toolchain/options. The prospective integration target remains the selected
96-register resource envelope and zero spills/local, with buffering and
shared-memory/residency limits reviewed separately. The primitive's32-register
count does not waive any integration gate. Stop if native N8 disappears,
query padding/fallback returns, precision changes, the resource/buffer gate
fails or numerical/address models disagree; do not sweep flags to rescue it.

Only an integrated, independently reviewed candidate passing those gates
could justify a free official correctness/performance check against original
tolerances and actual-best comparisons. Original1.233633 remains selected.
GPU correctness, replay/concurrency and complete-call performance are still
unresolved; the compiler result must not be reported as a speedup.
