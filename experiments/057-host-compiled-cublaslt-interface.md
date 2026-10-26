# P057 — host compilation of the unchanged native cuBLASLt binding

Parent experiment: P056, frozen execution commit `113e05e`, terminal result
commit `fb36441`. Performance decision control remains P046 at 1.1841399409×.
This is a build/interface repair and bounded diagnostic, not a new submission.

## Evidence and single change

P056 failed in NVCC before native library load: an internal `lexical.c:22316`
assertion, exit 134, while parsing a destructor expression. The receipt records
Torch 2.14.1+cu130 and CUDA headers 13.0. It does not establish which originating
header caused that failure, nor that cuBLASLt would fail to execute correctly.
The binding has no device kernel or device code: it only calls host CUDA,
cuBLASLt and PyTorch APIs.

P057 passes the **same native source bytes** as `cpp_sources`, with
`cuda_sources=None`, `with_cuda=True`, host `-O3`, and `-lcublasLt`. It removes
irrelevant NVCC code-generation flags and uses a distinct extension name. It
keeps the compiler standard/ABI chosen by PyTorch. There is no package upgrade,
manual compiler invocation, arbitrary process, download, numerical change,
precision change or lowered validation bound.

Hypothesis: documented host C++ compilation avoids the unnecessary NVCC parser
failure and exposes the same cuBLASLt interface. Falsifier: compilation/linking
still fails, no eligible algorithm is available, or the original correctness /
graph checks fail. This compiler repair has **no predicted execution speedup**.
Its value is enabling the original P056 causal algorithm comparison.

## Primary sources and eligibility

- [PyTorch 2.14 load_inline API](https://docs.pytorch.org/docs/2.14/cpp_extension.html#torch.utils.cpp_extension.load_inline)
  documents separate C++/CUDA source strings and forcing CUDA includes/linking.
- [PyTorch v2.14.1 source creation](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/cpp_extension.py#L2230-L2283)
  creates no CUDA translation unit when CUDA sources are absent.
- [Compiler selection](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/cpp_extension.py#L3230-L3246),
  [CUDA includes](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/cpp_extension.py#L2975-L2982),
  and [link libraries](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/cpp_extension.py#L2600-L2637)
  support this route. Explicit `-lcublasLt` is retained.
- [Official competition rules](https://openkernelchallenge.dev/rules), audited
  in P055, permit the documented `load_inline` compilation exception. All other
  process, graph, fresh-input, precision and evaluator-access restrictions remain.

No upstream implementation was copied. These references inform the API design.
Independent read-only review found no material blocker in the bridge. The
installed image's actual host compile/link compatibility is not yet verified.

## Isolation and verification

`native_cublaslt_probe.py` is unchanged, SHA256
`a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc`.
Its native body remains
`0e427a632fe12c410b2f0389deadf9c3757b919999287fea5705a892aad69fc8`.
The bridge imports the exact body and contract helpers and checks this hash
before any Torch import. Ordinary import is inert.

The P057 worker preserves the complete P056 cases, math, precision, protocol,
predefined timing order, deadlines and failure reporting. An AST equivalence
test accounts only for experiment/module identities and additional source
hashes. Both the original native module and frozen P056 worker are included as
provenance, not modified at runtime.

```bash
python -B -m unittest tests.test_native_cublaslt_host_probe \
  tests.test_projection_p057_native_screen -v
```

Seven bridge, six worker and twenty controller tests pass locally (33 total,
0.070 s), plus four local payload guards. They verify exact source reuse,
mocked builder arguments, no import-time Torch/build, protocol equivalence,
provenance hashing and negative source checks. They do not compile C++, execute
CUDA, measure performance or prove graph safety.

## Native gate and scope

Reuse P056's two representative shapes, at most two metadata-selected plans per
shape, full original FP64 oracle and column/zero/cancellation/outlier tests,
fresh same-address A→B→A, poisoned outputs, private scratch, and separate-stream
independent graphs. Actual stream overlap remains unproven by this protocol;
nearby native dimensions remain a required later coverage step. Six predefined
warmups and twelve reversed cold-L2 pairs retain all samples. Compare both the
same-interface first eligible plan and actual P046, never only a weaker custom
control. Neither raw diagnostic timing nor heuristic waves are an official
score or measured hardware occupancy.

Before accepting a broad algorithm-selection mechanism, both representative
shapes must improve at least 10% over P046 with at least 10/12 pair wins and no
correctness failure. Otherwise preserve the negative/limited evidence and do
not extrapolate across all scored shapes. Two-family M>1 improvements of
10%/20% model +4.07%/+8.54% whole score, conditional on achieving them everywhere;
this repair itself does not predict those gains.

Attention currently owns the coordinated GPU slot. No P057 remote request,
compile, execution or billing has occurred. Freeze/review the bounded controller
and source commit, recheck existing auth and $0 paid/free-credit guardrails,
and wait for coordinator slot release before one new diagnostic. No P056 retry.

实验卡：P056 卡在编译器解析，不是矩阵算法失败。P057 不改数学，只把纯主机
绑定交给主机 C++ 编译器，并保留 CUDA 链接。33 项本地检查和 4 项载荷检查通过；下一步先验证
真实编译，再按原协议检查正确性与两类代表形状，不能把这次修复说成加速。


## Bounded execution preparation

Controller `projection_p057_modal.py` preserves P056's timeout, resource,
transport, cost, exclusivity and shutdown guards. The only lifecycle changes
are P057 file/app/ledger identities and explicit frozen payload dependencies.
Independent tests compare normalized lifecycle AST to the unchanged P056
controller. The 15-file allowlist has 11 source hashes plus four license/notices
files. No controller, credential, repository or unrelated file is uploaded.

The planned maximum remains one H200, 2 CPU, 8 GiB, min 0/max 1, no retry or
detached deployment, internal 240 s / child 270 s / function 300 s, startup 45 s,
watchdog 345 s and 30 s stop allowance. The previous rate floor gives a
conservative $0.55838125 free-credit reserve; actual current pricing, credit,
existing login and paid cap $0 must pass again immediately before dispatch.
This is a reserve, not incurred cost. No P057 app or external request exists.

```bash
python -B -m unittest tests.test_native_cublaslt_host_probe \
  tests.test_projection_p057_native_screen tests.test_projection_p057_modal -v
python -B projection_p057_modal.py self-test \
  --frozen-head <committed-source-sha> --control-root /path/to/control-workspace
```
