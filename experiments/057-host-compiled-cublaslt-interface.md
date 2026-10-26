# P057 — native interface passes; measured no-split tactics rejected

Parent experiment: P056, frozen execution commit `113e05e`, terminal result
commit `fb36441`. Performance decision control remains P046 at 1.1841399409×.
This is a build/interface repair and bounded diagnostic, not a new submission.

The preparation below is historical. The terminal result and limits are recorded
at the end; no official score or new selected candidate results from this probe.

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


## Actual native result — 2026-10-26 00:46:25 UTC

Source freeze `5542653a08ac5bb8ceb34f52ccb1346e3de3dd2d`; app
`ap-1P4IYSdup9e7ZBjImQj1ZM`, accepted at 00:45:15.211205 UTC.
Host compilation succeeded in 25.784488 s. The worker-internal report is
35.234701 s; the outer supervisor records 35.883874 s. These wall
durations are not an allocated-GPU billing interval.

**444 recorded original-oracle comparisons passed** (72 explicit adversarial/ABA,
12 independent-graph and 360 warmup/measured comparisons), over both representative
shapes, all four selected native configurations, P046 controls, fresh input
families, column probes, zero/cancellation/outliers, poisoned outputs, same-address
A→B→A and pre/post timing checks. Independent graphs used separate streams and
private mutable buffers, three repeats per case; actual simultaneous kernel
overlap was not instrumented. Nearby M127/M15 were explicitly skipped. This is
bounded diagnostic coverage, not full challenge correctness or a concurrency
proof. Original tolerances were unchanged.

Raw whole-call CUDA graph timings, with six warmups excluded by protocol and all
24 measured samples retained in twelve reversed pairs:

| Shape / selected plan | P046 median (µs) | Native median (µs) | Median paired P046/native | Winning pairs |
|---|---:|---:|---:|---:|
| QKV M128, algo 41 / tile 21 | 22.112 | 194.704 | 0.113495 | 0/12 |
| QKV M128, algo 42 / tile 21 | 21.952 | 1971.376 | 0.011136 | 0/12 |
| Unpooler M16, algo 66 / tile 312 | 22.688 | 22.784 | 1.004913 | 6/12 |
| Unpooler M16, algo 66 / tile 134 | 22.816 | 24.240 | 0.934772 | 0/12 |

Ratio greater than one favors native. The paired ratio is calculated per
reversed pair; it is not the quotient of the two column medians. No fixed graph
overhead is subtracted. There is no formal confidence interval or official
score here. Unpooler native0's +0.49% paired latency cut is inconclusive, with
P046 tail samples up to 50.880 µs versus native0 maximum 27.840 µs. All tails
remain in the evidence. Native1 loses to native0 in both cases as well.

No representative satisfies the predeclared >=10% latency cut and >=10/12
winning-pair gate. **Reject promotion and do not spend an official check.**
P046 remains the working candidate, official score 1.1841399409×.

The interface gate is cleared: explicit cuBLASLt plan enumeration, complete
configuration retention, build/link, BF16/FP32 arithmetic and tested graph
replay work on this image. The heuristic query returned seven no-split QKV
plans and three no-split unpooler plans; only the predeclared two per shape were
executed. All selected plans required zero workspace. Their reported waves are
heuristic estimates, not occupancy measurements. No native instruction, hardware
counter or activity trace was collected, so the severe QKV loss cannot yet be
attributed to SIMT, Tensor Core choice, memory traffic or occupancy.

This result rejects the measured descriptor-layout/no-split/selection subset,
not cuBLASLt as a whole. Candidate causes are descriptor orientation and the
no-split restriction (which excludes potentially valid FP32 split-K). Compare
those against PyTorch's actual library setup before implementing one isolated
next control; do not change both at once or sweep arbitrary tactics.

Exact app stopped at 00:45:59 UTC; an independent read at 00:46:25.461792 UTC
confirmed stopped/tasks 0 and all apps stopped/tasks 0. Measurement slot released.
Paid usage remains $0. The workspace billing snapshot still reports a zero
gross delta; billing can lag, so retain the $0.55838125 free-credit reserve.
No new credentials, package installs, official checks or ranked submissions.

Reproduction uses the committed source and guarded existing authentication:

```bash
python -B projection_p057_modal.py launch \
  --frozen-head 5542653a08ac5bb8ceb34f52ccb1346e3de3dd2d \
  --control-root /path/to/existing/control-workspace
```

The same ledger deliberately refuses a duplicate launch. Source-bound raw
receipts and every timing sample are in `results/remote-P057-native/`;
`results/router-p057-assessment.json` and `results/router-p057-paired.csv` contain
the summarized gate calculations. The three published receipts are byte-identical
to the allowlisted returned artifacts, verified against their transport hashes.

实验结论：编译入口修复有效，444 次原始正确性检查通过；但这一小组不拆 K
算法没有胜过 P046。QKV 严重退步，unpooler 基本持平或变慢。保留 P046，不跑
无价值的官方检查；下一步先隔离描述符布局与算法过滤的影响。
