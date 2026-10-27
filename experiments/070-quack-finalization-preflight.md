# P070 — QKV M128 three-arm offline preflight

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
