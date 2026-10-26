# P063 — CPU-only SM90 compilation, not another official check

Prepared 2026-10-26 after P062. P046 remains selected at 1.1841399409×.

## Experiment card and prediction

P062 established that the exact raw P043 consumer is already 28.5–41.0% slower
than P046 on target pooler shapes. The encoded branch is historically another
2.23–3.76× slower than raw. These comparisons are reference-normalized across
different official runs, with control drift; they do not identify instructions
as the cause. Existing source reconstructs BF16 directly before the dot, so
there is no missing global-decode fusion to implement.

The discriminating question is whether the **encoded branch changes generated
matrix/layout/copy behavior or resource use**, beyond adding integer decode and
address instructions. Compile identical owned kernel bodies with only
`COMPRESSED=False/True`, M16/N4096/K8192, S8/BM16/BN64/BK64, four warps, two
stages, explicit cuda:90:32 target and 16-byte pointer alignment. M16 removes
masked-M and extra-M-tile differences. Save all IR/PTX/cubin/SASS, assembler
diagnostics and static resources. No kernel is loaded or launched.

Prediction: a matrix-family, operand-conversion, copy-order or spill difference
would support a focused compiler-aware follow-up. If relevant lowering and
resources match, that rejects this compiler-regression explanation, leaving
decode/address costs unresolved. Static instructions alone establish neither
timing nor actual memory traffic/stalls/occupancy. Any later candidate must beat
P046, not just this weaker raw backend. No new optimization candidate or GPU
performance gate is implied by a successful compile.

## Official access audit

The live `okc show muse_bf16_proj --json` task SHA is
`deb9118fa75cbc03c2ebd16b283e72a0314168d40299fa7501373eb1960a2bc4`, matching
the saved task. `okc api` and `okc help status/check` document result/status,
activity profiles and the last 16,000 diagnostic log characters; no dedicated
PTX/cubin/artifact endpoint is documented. Ordinary owned-code diagnostics are
permitted. Thus a future substantive check could report bounded compiler
metadata through that documented log, subject to unchanged benchmark semantics
and the remaining rules. This is different from claiming a native artifact
download service exists. We do not inspect evaluator internals, print private
data, modify a benchmark to transport artifacts, or consume a check merely to
add logs to unchanged P043/P062.

## GPU-free route and safeguards

Triton 3.8's explicit `GPUTarget` bypasses active-device target discovery.
`CompiledKernel.asm` and compiler metadata are available before device handles
are initialized. The full AOT C-stub CLI later touches `driver.active`; this
diagnostic uses the direct compiler API instead. Register/spill evidence comes
from bundled assembler/cubin tools, not the device-loaded `n_regs` property.

Mac arm64 has no native supported Triton wheel. Installed OrbStack initially
had no running VM/socket. Its first list attempt failed while rotating a log
outside the filesystem sandbox, before startup. Normal scoped escalation then
initialized the local engine: Docker29.4/Linux arm64, no existing containers or
images. No login, license change, credential or cloud call was made. This is
not a Modal-auth workaround; all compilation stays on the user's Mac CPU.

- Official `python:3.12-slim-bookworm` image resolved to digest
  `sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70`.
- Official PyPI Triton3.8 CPython3.12 Linux ARM64 wheel, 226,474,486 bytes,
  SHA256 `a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50`.
  The wheel includes ptxas, cuobjdump and nvdisasm. No dependency for Python3.12
  is required. Package download is separate from offline execution.
- Container limit: two CPUs, 4 GiB, 240-second internal timeout, no network,
  no NVIDIA devices, dropped capabilities, no-new-privileges, read-only input
  and root filesystem, temporary package/cache storage, dedicated writable
  artifact directory. Mount only extracted kernel/helper/manifest/dependency,
  not repository, credentials, home or Docker socket. Remove container on exit.
- Extraction verifies both complete source hashes and equal function AST/body;
  the only executable module content is the owned kernel plus Triton imports.
  No fake CUDA availability, torch import, data allocation or GPU driver access.

Four local source/ABI/error-path tests passed in 0.033 s. These tests are not
compilation or numerical execution. Exact-source hashes, compiler/tool hashes,
specialization and limits will accompany the output. An official judge image
match is not claimed from a public PyPI wheel, even if its version agrees.

```sh
python -B -m unittest tests.test_projection_p063_offline_compile -v
python -B projection_p063_offline_compile.py --prepare local-notes/p063-offline
```

Primary references: [official task API](https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj),
[Triton3.8 compiler](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/compiler/compiler.py),
[target type](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/backends/compiler.py),
[NVIDIA backend](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/compiler.py),
[AOT CLI](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/tools/compile.py),
[official release files](https://pypi.org/project/triton/3.8.0/#files).
