# P070 — QKV M128 three-arm finalization experiment

## Completed native result — 2026-10-27 16:52 UTC

**Reject this configuration; retain P046.** After approved access restoration,
the pinned Quack SERIAL/SEPARATE comparison ran on an NVIDIA H200 (SM90).
All **240 FP64/probe-oracle checks passed**, including normal A → outlier B →
byte-identical A-return, sparse and zero inputs, before/after timing. There were
two independent weight/input cohorts, each with six excluded warmups and 24
measured rounds in balanced reversed orders. The extra library warmup/capture
calls were not individually oracle checked. No nearby shapes or concurrency
were tested.

| Cohort | P046 median µs | SEPARATE median µs | SERIAL median µs | SERIAL latency increase vs P046 | SERIAL increase vs SEPARATE |
|---|---:|---:|---:|---:|---:|
| 7001000 | 22.144 | 30.032 | 32.352 | 45.47% | 7.29% |
| 7002000 | 22.144 | 30.320 | 32.304 | 45.12% | 6.49% |

Increases use the predeclared median of adjacent reversed-order paired ratios,
not ratios of the displayed medians. Gross CUDA-event whole-call graph timing
includes Quack's per-call scratch allocation/initialization and synchronization;
no fixed-cost subtraction. All samples remain, including SEPARATE's 43.936 and
64.320 µs tails. This is neither an official score nor a claim about all Quack
configurations. The predeclared ≥5% cut versus P046 and >2% versus SEPARATE gate
fails in both cohorts. No official check was spent.

Code generation is verified against the **exact runtime cubin hashes** in both
cohorts. Both main kernels have BF16 WGMMA, FP32 accumulation, six-stage A/B
shared layouts, 168 registers, zero listed stack/local allocation, and no PTX
local-memory load/store sites. Their main shared structs are 229,376 bytes,
plus protocol storage; `cuobjdump`'s 1,024-byte static shared figure is not total
launch shared memory. SERIAL's BF16 epilogue has four stages; SEPARATE's FP32
epilogue has two, plus a 20-register reducer. SERIAL contains ordered acquire/
release synchronization and final FP32 partial addition. Therefore the test
includes epilogue/layout differences and cannot attribute the loss solely to
removing a reducer. No hardware-counter or isolated-phase causal claim follows.

Tested harness commit `6bf3c39f5878f90cb4bd1bb50532d0bffddbd524`, SHA256
`7f9a85585f51772afe2ab3c13a65600afe5cda6b8e0e636a014619ce540ee169`.
Torch 2.14.1+cu130, Triton 3.8.0, CuTe DSL 4.8.0, TVM-FFI 0.1.11, einops 0.8.2.
The inherited Torch BF16 reduced-precision-reduction flag was true; this test
changed no global precision setting. Quack's inspected matrix instructions
use BF16 operands and FP32 accumulation. Official environment compatibility of
the upstream CuTe FFI patch remains unresolved; no scoring candidate was sent.

The first CPU attempt was stopped during redundant `apt-get update`; Git was
already present. The next CPU attempt exported host wrappers only. The repair
uses NVIDIA's documented `__cubin__`, `__ptx__`, and `__mlir__` attributes with
`CUTE_DSL_KEEP=ir,ptx,cubin`, without debug code generation. Those recoverable
attempts are retained. No H200 run was duplicated.

Native app `ap-SxWfI1ObeGcdNCek1ctjdH`; measured diagnostic work wall time 5.813 s
(excludes setup). All four P070 apps independently verified **stopped, tasks 0**
at 16:52:52 UTC; inventory had no active apps. Per-app conservative reservations
were $0.15 CPU / $0.70 native, within the approved free-credit/$0-paid scope.
Actual billing was not queried, as requested; these reserves are not charges.

Reproduction (existing isolated image/profile; no credential or billing action):

```bash
python -B -m unittest tests/test_projection_p070_quack_preflight.py
python -B projection_p070_modal.py compile --output local-notes/p070-compile-device
# Inspect artifacts and populate the receipt review; never set the gate blindly.
python -B projection_p070_modal.py native --prior local-notes/p070-compile-device --output local-notes/p070-native-attempt1
```

Use new output directories for a separately authorized rerun. The controller
requires no active app, one container, a 210-second child timeout, 300-second
function limit, no retries, explicit stop and final inventory verification.

Evidence: [summary/raw cohorts](../results/projection-p070/summary.json),
[native receipt](../results/projection-p070/native/native.json),
[compiler review](../results/projection-p070/compile/code_review.json),
[reviewed compile identity](../results/projection-p070/compile/receipt.json),
[cleanup](../results/projection-p070/shutdown.json),
[recoverable attempts](../results/projection-p070/recoverable-attempts.json).
The unchanged champion remains P046 (latest ranked same-source 1.189713×).
This completes the bounded P070 experiment; do not promote or widen this losing
configuration based on the removed-reducer hypothesis.

## Historical offline preflight and initial access blocker

**Decision: CPU preflight passes; native compilation/execution is blocked.**
P046 is unchanged: `candidates/projection_pooler_column_batch2_hybrid.py`,
SHA256 `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Its ranked result remains **1.189713×**, job `497a0b5130e548c1`.
This experiment has no GPU result or speedup claim.

The hypothesis is that ordered in-kernel finalization can replace a separate
reducer profitably. Compare actual P046 with Quack **SEPARATE** and **SERIAL** at
QKV `[M,N,K]=[128,2304,8192]` only. Both Quack arms use BF16 A/B/output,
FP32 accumulation/partials, singleton batch, tile 128×128×64, split 2,
cluster 1×1×1, static persistence and no ping-pong. The only requested mode
change is finalization; this is not a tile search.

Pinned Quack: [`77a750c`](https://github.com/Dao-AILab/quack/tree/77a750c0d9548d4a6e23c0f64062d61b3ea420ff),
Apache-2.0. No upstream implementation is vendored or executed here. Its
[host path](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/gemm.py)
allocates each invocation's scratch and zeroed semaphore. The harness captures
that complete call. With split 2, SERIAL stores the first FP32 partial; the fixed
second split waits, adds, and rounds once. Release/acquire publication and a
drained final TMA store support sequential replay structurally, not as measured
GPU proof. Resident-grid progress and concurrent execution remain unverified.

**Causal limits:** SEPARATE's GEMM output is FP32 scratch, while SERIAL's is BF16.
That can change automatically chosen stages/shared layouts/resources; generated
parity must be inspected. PDL defaults on. P046 has a different vendor execution
plan, so it is the practical performance control, not a matched mainloop.
Quack's upstream CuTe FFI converter patch also needs contest-rule reconciliation
before any official candidate submission. This file is diagnostic tooling.

`projection_p070_quack_preflight.py` prepares two fresh weight/input cohorts,
same-address A→outlier B→A, sparse and zero replays before/after timing,
six excluded warmups and 24 measured rounds per cohort. All six arm orders are
balanced in reversed adjacent pairs, with cold L2 before each arm. The planned
240 original-oracle comparisons include every measured output; another 24
library-warmup/capture outputs are not individually checked. All raw samples
are retained, without fixed-overhead subtraction. Native execution requires a
reviewed source/config/dependency/object-bound compile receipt and matching
exports from the runtime plans. An export mismatch stops timing, even if it
could be nondeterministic compiler metadata.

Predeclared screen: correctness first; SERIAL must reduce raw whole-call time
at least 5% versus P046 and more than 2% versus SEPARATE in **both** cohorts.
Passing does not promote a one-shape candidate. Tail shapes, concurrency,
official whole-score performance and generated resources remain untested.

Local checks, 2026-10-27, Darwin arm64 / Python 3.12.13:

```bash
python -B -m unittest discover -s tests -p test_projection_p070_quack_preflight.py -v
python -B projection_p070_quack_preflight.py plan
```

**15 CPU/mock test methods pass**, including 650 K-partition subcases,
call-local outputs/high-level dispatch, timing order, threshold boundaries,
and rejection of incomplete evidence or changed compiler objects. Independent
source/protocol and code reviews completed. These tests do not execute Quack.
The compile command was tested only to confirm its dependency gate: exit 1,
`Missing dependencies; no installation is performed`, before compiler imports.

**Exact blocker:** this venv has macOS Torch 2.14.1 but no Triton, CuTe DSL,
TVM-FFI or einops; no complete clean checkout of the Quack pin is provisioned.
Nothing was installed. Historical remote package metadata is not a current
environment or exact-pin verification. Fake-tensor SM90 compilation is
source-supported without a GPU, but it is not available in this local setup.

Smallest next command, only once those dependencies and the pinned checkout
are available in an authorized environment:

```bash
timeout --kill-after=5s 180s python -B projection_p070_quack_preflight.py compile \
  --quack-root /path/to/pinned-quack --output /tmp/p070-compile
```

It exports SERIAL, SEPARATE and reducer objects for symbolic M/N/K/L. It does
not launch kernels or establish layout/resource parity. Inspect actual
instructions, AB/epilogue stages, shared layouts, registers/spills/shared memory
and semaphore initialization, recording the four `review` fields named by
`require_compile_review`; leave `ready_for_execution` false until supported.
Only then is the bounded native command meaningful:

```bash
timeout --kill-after=5s 180s python -B projection_p070_quack_preflight.py run \
  --quack-root /path/to/pinned-quack --compile-review /tmp/p070-compile/receipt.json \
  --output /tmp/p070-native.json
```

The external owner must enforce container lifetime and cleanup. This turn
launched no GPU/service/check, accessed no credentials or billing, and changed
no champion, environment, scorer, precision setting or other track file.

## Execution-route check, 2026-10-27 16:05 UTC

The existing Modal profile was tested with a read-only `app list`: the service
returned **token expired**. This is a current response, not an inference from
old expiry metadata. No credential was created or renewed, and no app was
created. Official `okc problems --json` succeeds: Projection has 20/20 off-board
checks and 9/10 ranked runs left at this observation; ranked runs are not authorized
for P070. `okc check --help` and `okc api` expose candidate-versus-SGLang checks,
not custom native sessions. Independent checks cannot supply the predeclared
same-run three-arm, two-cohort comparison, so no quota was spent on a substitute.

Pinned `quack/cute_dsl_utils.py` lines 31–53 directly replace CuTe's FFI argument
converter on import. This is compiler compatibility behavior, not timing
tampering, but the official loaded-module restriction has no verified exception
for it. Do not infer submission eligibility or modify the judge to resolve it.
It does not prevent an authorized native diagnostic outside the judge.

The existing local OrbStack engine starts successfully; its only cached image
is `python:3.12-slim-bookworm`, with no containers running. Local Docker cannot
provide H200 execution. No packages were installed and no compiler container
was launched while native measurement access is blocked. **Smallest required
intervention: fresh authorization to renew the Modal CLI credential.** Then use
the existing isolated H200 environment to provision the reviewed pin, compile,
inspect, and measure. Billing was not reopened. P046 is unchanged.
