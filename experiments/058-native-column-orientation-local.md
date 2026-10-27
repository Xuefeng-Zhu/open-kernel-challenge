# P058 — descriptor orientation control, local and parked

Parent: P057 native interface, source `5542653`, result `5cbe7c6`. This local
prototype computes the same matrix equation using a different descriptor view;
it has not been compiled, uploaded, timed or promoted.

## Isolated change

Physical row-major Y[M,N] is column-major Y.T[N,M]. Compute
`Y.T = opT(colW[K,N]) @ opN(colX[K,M])` with leading dimensions K/K/N.
The output address remains `i*N+j`, summing `W[j*K+p] * X[i*K+p]`.
There is no tensor transpose, copy, repack or extra GPU kernel. All physical
validation, 256-byte alignment, current stream, BF16 input/output, FP32 compute,
private workspace and no-split policy remain unchanged.

`native_cublaslt_column_probe.py` derives the native body through nine
hash/count-checked substitutions. The original native body and P057 remain
immutable. The host build route is the same already validated `load_inline`
route; the changed native body is not itself compiled or validated on GPU.

The mapping matches [PyTorch's operand swap](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/cuBlasCommonArgs.h#L105-L108)
and [dimensions/transposes](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/cuBlasCommonArgs.h#L127-L143).
No upstream implementation was copied. These API/source references establish
layout equivalence, not a performance claim.

## Cheap local falsifiers and result

```bash
python -B -m unittest tests.test_native_cublaslt_column_probe -v
```

Eleven CPU/mock tests pass: complete integer GEMM address models on rectangular
and tail dimensions, official-dimensional address probes, all 65,536 BF16 bit
patterns as stored values (not floating-point CUDA arithmetic), frozen source
identity, exact native replacements and mocked builder arguments. Independent
review found no concrete layout/guard blocker. No real Torch/native build, GPU,
stream replay or official validation was run.

## Why parked before GPU

The saved P049 activity has more direct evidence than the initial generic API
audit: the exact P046 source hash `001ea12e...c278545e` executed QKV M128
`nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` plus a cuBLASLt split-K
reduction kernel with float-typed intermediates. Those instrumented stage times
were 14.527 + 3.327 microseconds, not performance-acceptance timings or exact
algorithm IDs. The reducer symbol supports split reduction, but does not alone
prove an exact cuBLASLt reduction enum. P057 itself captured no activity trace.
Raw evidence: `results/remote-P049-screen/timing.json`, SHA256
`6e78fffd4dfdafe53eb7b5d63102bbeffdf97d84ccab5956773944a7b60fa512`.

P057 deliberately excluded split-K and timed only two of seven eligible QKV
plans. Therefore neither poor layout nor all no-split algorithms were disproved.
Prioritize a **separate P059 FP32 split-mask-only control**, retaining P057's
original row descriptors, before spending GPU time on this orientation prototype.
The no-split restriction was a conservative initial gate, never a contest rule.
P058 is preserved for a later discriminating test, not stacked silently into P059.

If revisited, bound the orientation test to QKV M128: original row ID41, already
eligible but untimed row ID66, and first eligible column plan, plus actual P046.
Do not claim IDs41/42 are SIMT without instructions or activity evidence. Hold
precision/reduction fixed. Require original numerical/ABA/private-buffer checks
before paired graph timing and retain all samples.

## Whole-score relevance and gates

This one-shape test is an interface/mechanism falsifier, not a competitive result
on its own. Using P046's immutable official receipt (QKV M128 18.193201 us),
10%/20%/50% reductions on this shape alone model only +0.385%/+0.780%/+2.036%
whole score; all other times are fixed. Achieving 10%/20% across QKV+unpooler M>1
would model +4.07%/+8.54%, but broad extension is untested. Raw Modal times cannot
be substituted into this official score model. Promoting a complete candidate
requires wider shapes and official validation, not a faster isolated descriptor.

Current selected P046 stays 1.1841399409x. No P058 app, cost, check, failure or
performance result exists. Selection and graph/tail coverage remain unverified.

实验卡：这是不移动数据的等价布局对照，11 项本地检查通过。已有 P046 时间线
明确显示 split-K，所以先把“不拆 K”限制单独检验；本原型保留待用，不把两个
因素混在一起，也不把本地地址证明当成 GPU 正确性或加速。
