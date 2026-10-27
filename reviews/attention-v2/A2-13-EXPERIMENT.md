# A2-13: FP8 QK on the preserved Triton fallback

Parent: A2-11 `mixed_fa3_decode_parallel_merge.py`, source commit
`7e302d2366b194c32c433f6d6f9cc7b0b59b3928`, SHA256
`c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`.
Official job `cfa8a8670e394f80`, 1.1876787561x; parent remains selected.
Candidate: `mixed_fa3_fp8_qk.py`, SHA256 `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.

Problem: four scored paths still expand FP8 cache keys to BF16 for QK dot.
Hypothesis: retaining E4M3 keys and encoding Q as E4M3 may reduce QK work or
operand movement while leaving BF16 PV unchanged. Source types alone do not
prove native FP8 tensor-core instructions or a bottleneck; small-M/N lowering
may upcast. No hardware-counter evidence establishes a compute-bound kernel.

Exact change: Q = E4M3(BF16(norm_rope(Q))); logit scale = sm_scale *
qk_scale_factor * k_scale * LOG2E. Historical and new-token QK use E4M3 operands,
FP32 results and max_num_imprecise_acc=0. The latter requests FP32 promotion
without long imprecise FP8 accumulation; emitted instructions remain unverified.
Parent used BF16(qk_scale_factor * norm_rope(Q)). Moving the BF16 boundary is
explicitly part of this encoding experiment. BF16 then FP8 can double-round
relative to direct FP32-to-FP8 reference encoding; neither bitwise identity nor
GPU tolerance is inferred. The official SGLang path also supplies BF16 Q before
its FP8 cast. Do not apply qk_scale_factor or k_scale twice.

Unchanged: BF16 P/V operands, FP32 softmax/partials, key normalization and FP8
cache writes, V scaling, masks, partition, new-token ownership, scratch/tickets,
merge, warps/stages/tiles, all FA3 functions and dispatch. QK reduction D=64
satisfies the reviewed Triton3.8 source FP8 minimum K=32 even for new-token N=16; that
is only a static compatibility check. Optional interpreter DOT conversion is
preserved but no interpreter execution is claimed.

Expected affected scored paths: sliding decode B1/B16/B32, full decode B32.
All eight FA3 scored paths unchanged. Twelve of sixteen prerequisite cases
also take the modified fallback, including DP heads, page64, non-unit scales
and some L8/full cases. Geometry is not narrowed to scored shapes.

Predeclared useful result: >=10% geometric-mean normalized latency reduction
across the four affected scored paths, no >5% path regression, all official
correctness checks. Small aggregate changes/overlapping CIs are inconclusive.
Model only: cutting each of the four target times by10%, holding the remaining
same-receipt times fixed, gives 1.2239352772x. This is not achieved performance.
All six loads use46*sliding+16*full, then geometric-mean(S/C).

Local commands:
```
.venv/bin/python -B reviews/attention-v2/test_fp8_qk_source.py
.venv/bin/python -B reviews/attention-v2/test_fp8_qk_numerics.py
```
These are syntax/source invariants and CPU rounding/arithmetic models only.
Native CUDA compilation, output/cache correctness, performance, independent
graph replay and concurrency for the new bytes require native evidence.
A13's passes belong only to the exact parent, not this candidate.

One exact candidate off-board check and reviewed private source/tests/results
push were explicitly approved after previous delegation denials. Earlier
blocked delegations executed no external action. No new credential, Modal,
paid compute, ranked entry, force push or visibility change is authorized.
The coordinator must verify current quota, no active/uncertain job, committed
hash, official rules version and private destination before the respective action.

Primary contracts: [official task](https://openkernelchallenge.dev/api/v1/problems/muse_attention_decode),
[official rules](https://openkernelchallenge.dev/rules),
[Triton3.8 dot semantics](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/semantic.py),
[Triton3.8 NVIDIA minimum dot dimensions](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/compiler.py),
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
Saved reference.py307–312 places FP8 query rounding before qk_scale_factor.
No third-party competitor software is executed or copied.

## Official coverage audit before result

The saved public checker collects output and new K/V cache locations. It checks
new cache rows within8% relative L2, then compares output using our own new cache
entries. Default oracle Q is BF16-rounded; its arithmetic uses FP32 and final
output is BF16. The alternate E4M3 query oracle establishes the adaptive bound,
not a bitwise target. The worst-row output check is a token row with all query
heads concatenated, not an independent bound on each head.

TASK documents fresh-input CUDAgraph comparisons, but this receipt cannot prove
explicit same-address A/B/A, untouched cache rows, immutable input invariance or
actual overlap of independent graphs. Those remain unverified for A2-13. The
separate A13 diagnostic passed a limited subset on the exact A2-11 parent only.

Source types/out_dtype/max_num_imprecise_acc request precise FP32 dot results;
without this candidate's PTX/SASS or compiler metadata there is no evidence of
native FP8 instructions, register savings, spills or a hardware bottleneck.
The official pass/fail and whole-call timing will test usefulness directly.
