# Attention v2 — local-only review of the user's unchanged algorithm

Normalized candidate: `submission.py`, SHA256
`b1a87afb4c428502d5066688ba784f577236b6bad4fea8879c8cd98fefb9c3dc`.
The original transcribed paste is `paste.original.txt`; XML-escaped transport view
is `paste.transport.txt`. `normalization.diff` contains only five Markdown-link
repairs in four hunks. `transport-decoding.diff` separately records XML entity
transcription. No algorithm, parameter, precision or synchronization fix was applied.
The original links cause Python SyntaxError at line91; normalized Python syntax passes.

No confirmed single-invocation numerical defect in the published L=1/8, window0/2048
contract was reproduced locally. This does **not** establish Triton compilation,
CUDA correctness, graph replay safety or speed. The first change to prioritize for
concurrent production use is counter ownership, below.

## 1. High — device-global tickets mix concurrent invocations (robustness defect)

Source69–77 caches one buffer by device;262 reuses it across calls, while254–259
allocates different per-call partial buffers. Counter205–207 has no call/graph ID.
Two calls on different streams or simultaneous graph launches can therefore count
each other's CTAs and elect a merger before its own partials are ready.

Concrete CPU state-model witness, even with sequentially consistent atomics:
for published-style S=16, let A splits0–14 publish, then B split15 publish. B sees
previous ticket15 and merges B's scratch although B splits0–14 are unwritten.
After B resets, let B splits0–14 publish, then A split15 publish. A completes,
but B's invalid earlier result is never repaired; counter ends0. The model covers
S2/3/5/9/16/24/33/66/132. Actual AST-extracted `_tickets` executed on CPU returns the
same tensor object for different calls requesting2 and128 counters.
This is a definite isolation bug under overlap, not an actual CUDA race reproduction,
and not a claim that the official judge launches overlapping calls.

Minimal fix options for a separate version: give each live call/captured graph its
own owned/initialized ticket storage, or explicitly serialize all users of shared
storage with CUDA dependencies. Per-stream ownership needs to account for graphs
replayed on different streams and lifetime. Changing `relaxed` reset to `release`,
adding a fence, or adding only a barrier cannot distinguish calls and does not fix
this counterexample. A per-call zero operation may add a launch; measure its real
cost instead of hoisting input-dependent work or concealing it from timing.

## 2. Medium — intermediate BF16 fidelity/policy concern, not oracle failure

At194–195 and223–224, FP32 attention output enters RMSNorm without an intermediate
BF16 round. Official baseline129 and template88–99 materialize BF16 before this norm;
reference320–337 computes FP32 instead. TASK says activations stay BF16 wherever
SGLang keeps BF16. Numerical acceptance alone does not settle a strict literal
interpretation of that policy. For upstream-boundary fidelity, a minimal separate
variant rounds `a` through BF16 before both RMSNorm branches.

Similarly170–174 directly quantizes FP32 normalized/RoPE K to FP8. This matches the
numerical reference303–313 and template, but pinned SGLang FusedQKNorm creates K
output in qkv.dtype (BF16) before cache conversion. A strict upstream-boundary
variant would round K to BF16 before divide/FP8 cast. Because the official starter
itself omits that K boundary, classify this as a policy ambiguity, not a confirmed
in-contract numerical bug. Do not silently apply these changes to the user's test
version; they require an independently recorded variant and actual target validation.

Q BF16 at121 is explicitly permitted; the reference default is BF16. FP8 queries
are optional, so missing Q→FP8 is not a defect. K/V division by scales before E4M3,
and multiplication of K scale into logits / V scale into output, are consistent.
FP8 stored values are converted back from their rounded representation for attention.
Softmax probabilities are BF16-rounded before value dot160/187, while the oracle is
FP32; this is an unverified numerical risk, not a reproduced failure.

## 3. Medium — native numerical/compilation/capture validation remains unavailable

The Mac has Python3.12.13/PyTorch2.14.1, no CUDA and no Triton/NumPy package. Exact
attempt `TRITON_INTERPRET=1 .venv/bin/python reviews/attention-v2/submission.py` failed
at import with `ModuleNotFoundError: No module named 'triton'`. No Triton kernel or
interpreter was executed, no install was attempted and no GPU claim is made.

RoPE90–93 uses exp2/log2 and query factor121 is applied after rotation. Reference
uses a power expression and factor before rotation. Real arithmetic equivalence
is not FP32/trig/FP8 equality. CPU Torch frequency/rotation relative L2 was0.000864
at positions80000/86007, and0.012443 at1048576/1048583. The latter is extra robustness:
the published million-token test is full attention without RoPE. These are CPU
math differences, not Triton error bounds or cache/output failure measurements.
The eight representative CPU equation-model cases passed the unmodified official
output/cache validator, with output relative L2 roughly0.00051–0.00131; this cannot
prove native behavior on all16 prerequisites/12 benchmark cases or nearby shapes.

Before any performance inference, the target must compile/run the actual source,
check both FP8 cache writes and output, and exercise serial graph replay, fresh inputs,
scaled caches, DP heads, edge/coherent inputs, and any intended concurrent usage.
No GPU upload is authorized yet; the projection budget is exhausted and the separate
attention-check approval is pending. Do not send a request from this local preflight.

## Synchronization/publication: do not report an unproved missing fence

Within one isolated invocation,201–204 stores all partials then executes a CTA barrier;
206 uses GPU-scoped acquire-release RMW, and213–215 loads partials via `.cg`. The
last ticket can chain preceding publications. This is the intended publication
pattern; no definite missing acquire/release was established by source inspection.
`debug_barrier` alone is only CTA-local, but it is not the only synchronization here.
Actual compiler lowering and a target race test remain unverified.

The relaxed reset227 does not by itself prove sequential graph replay is broken:
same-stream kernel completion orders all output writes and the reset before the next
launch. The serial counter model passed seven replays for each of seven split counts.
Simultaneous calls/graphs remain unsafe due to item1, regardless of stronger atomics.

## Window, lengths and dot dimensions: published cases versus generalization

- Source180 omits the new-token lower window bound. For official L1/8 with W2048,
  every causal new token differs by at most7 positions and is inside the window;
  no published-case defect follows. Cached-token lower bound152 is correct. All140
  analytical configurations of L1/8, window0/2048, split counts and boundary lengths
  exactly match reference causal/window sets (including extra seq0 coverage).
  A generalized W2/L8 counterexample fails. Minimal hardening: additionally require
  `npos > qpos-window` in the new-token mask when WINDOWED.
- `BLOCK_L=16` at269 covers official L1/8. L17 would omit token16's cache write and
  attention contribution; L17 is not published contract support. Generalization:
  derive BLOCK_L as at least next_power_of_2(L), respecting backend dot constraints,
  or reject unsupported L explicitly. No change is required just to cover L1/8.
- Cached logits reduce over D64; cached probability/value dot over64; new-token logits
  over D64; new probability/value dot187 reduces over BLOCK_L16. Operands are BF16
  on GPU, not FP8. Current upstream NVIDIA Triton `min_dot_size` allows reduction K16
  for16-bit operands (K32 applies to8-bit operands). Thus do not assert a BF16-K16
  compile failure or pad to32 based on an FP8 restriction. The contest image's exact
  compiler version/lowering is still untested. M16/64,D64 and G8 match public TP8/DP.
- Old-cache loops stop below seq; only the designated split writes/attends new rows
  in registers. Under one invocation and the generator's disjoint page mappings,
  there is no identified cross-CTA read-before-write hazard on those new rows.

## Graph allocation and launch-count qualifications

`_tickets` uses torch.zeros on first use/growth. If warmup has populated the cache
before capture, replay does not rerun that Python allocation. If captured cold,
initialization can become part of the graph. Therefore the one-launch claim is
steady-state/context-dependent, not locally measured. Per-call torch.empty scratch
is shape-only and may use a graph pool; allocation alone is not a proven capture bug.
No host `.item()`, synchronization or input-value branch appears in kernel_fn.

Keeping only the latest per-device buffer also needs graph-lifetime ownership if a
later request exceeds4096 counters and replaces it; published calls remain below
that capacity. PyTorch documents captured tensor memory lifetime and stream dependencies.
This is a robustness/capture test requirement, not a demonstrated published-case failure.

## Reproduce and inspect

```sh
.venv/bin/python -m py_compile reviews/attention-v2/submission.py
.venv/bin/python reviews/attention-v2/preflight.py
TRITON_INTERPRET=1 .venv/bin/python reviews/attention-v2/submission.py
```

The first two pass (syntax +8/8 CPU tests); the third fails at missing Triton import.
`cpu-results.json` contains actual model cases and counterexample traces. It does
not contain benchmark timings. Candidate, original paste, diffs, reports and public
contract snapshot are committed only on a local review branch; no push/upload.

Sources: [current official task/API](https://openkernelchallenge.dev/api/v1/problems/muse_attention_decode),
[task page](https://openkernelchallenge.dev/problems/muse_attention_decode),
[Triton atomic_add](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html),
[Triton debug_barrier](https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html),
[NVIDIA PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model),
[current NVIDIA Triton backend bounds](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py),
[pinned SGLang QK norm](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py),
[PyTorch CUDA graph memory/stream guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
Official files here match the earlier public snapshot; the current upstream Triton
bounds are source evidence, not a compiler probe of the pinned contest environment.
