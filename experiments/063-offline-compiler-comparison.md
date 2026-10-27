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
initialized the local engine: Docker 29.4/Linux arm64, no existing containers or
images. No login, license change, credential or cloud call was made. This is
not a Modal-auth workaround; all compilation stays on the user's Mac CPU.

- Official `python:3.12-slim-bookworm` image resolved to digest
  `sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70`.
- Official PyPI Triton 3.8 CPython 3.12 Linux ARM64 wheel, 226,474,486 bytes,
  SHA256 `a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50`.
  The wheel includes ptxas, cuobjdump and nvdisasm. No dependency for Python 3.12
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
[Triton 3.8 compiler](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/compiler/compiler.py),
[target type](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/backends/compiler.py),
[NVIDIA backend](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/compiler.py),
[AOT CLI](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/tools/compile.py),
[official release files](https://pypi.org/project/triton/3.8.0/#files).

## Attempt 1: driver discovery failed before compilation

Frozen source `6faa720` installed the verified wheel into the disposable
CPU-only container, then failed at `kernel.create_binder()`. Triton 3.8
`runtime/jit.py:694` calls `driver.active.get_current_target()` there and raised
`RuntimeError: 0 active drivers ([]). There should only be one.` No IR, cubin,
GPU execution or performance result was produced. The container exited 1 and
`docker ps --all --filter name=okc-p063-offline` returned no container.

This was our unnecessary use of a runtime binder, missed during static review,
not a failure of explicit-target compilation or a reason to fake a GPU driver.
`JITFunction.__init__` already creates params/arg_names; `ASTSource` needs no
runtime binder. Remove only that call, retain the failure, add a regression
guard against device-binder/launcher calls, and repeat the otherwise identical
offline diagnostic. No dependency, precision, kernel or resource option changes.

## Attempt 2: successful offline compilation and disassembly

Repair commit `fe2c034` compiled both variants and exited 0. Python 3.12.15,
Triton 3.8.0, bundled ptxas 12.9.86 and cuobjdump 13.1.80, Linux ARM64 on the Mac.
Both PTX files target `sm_90a`, PTX 8.8, 128 threads. Exact dependency/tool/source
and artifact hashes are in `evidence/p063/manifest.json` and `dependency.json`.
Compile+disassembly wall times were about 0.744/0.467 seconds; these are host
tool durations, **not kernel timings**. The disposable container was removed.

| Exact M16 specialization | Raw | Encoded |
|---|---:|---:|
| Registers from ptxas | 38 | 120 |
| Stack / spill stores / spill loads, bytes | 0 / 0 / 0 | 0 / 0 / 0 |
| Triton dynamic shared-memory launch request, bytes | 10,240 | 18,688 |
| cuobjdump's separately reported SHARED field, bytes | 1,024 | 1,024 |
| Static SASS instruction sites, including prologue/epilogue/padding | 224 | 944 |
| BF16 HMMA instruction sites | 8 | 8 |
| PRMT instruction sites | 0 | 162 |
| LDSM instruction sites | 8 | 0 |

Do not substitute the cubin SHARED field for the launch request or claim these
counts are executed work. The dynamic shared request is compiler metadata;
neither value establishes achieved occupancy. The zero-spill observation
specifically rejects an offline register-spill explanation for this build.

Both use `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; no matrix
instruction-family demotion is observed. Raw TTGIR lines 98–100 load the staged
weight operand into dot-op layout with kWidth 2. Encoded TTGIR lines 222–225
instead materializes the reconstructed 64×64 BF16 tile through `local_alloc`,
`memdesc_trans`, then `local_load` into dot-op kWidth 4. Corresponding encoded
SASS around offsets 0x2920–0x2d70 includes STS.128, a barrier and LDS.64. This is
a real extra decoded-operand staging/dataflow difference **in shared memory**,
not a decoded global weight buffer. The source's fused reconstruction claim
remains true, but does not imply register-direct tensor-core consumption.

The encoded output also contains scalar predicated `LDG.E.U8` sites and much
more integer/permute work. Both variants contain async copies; calling either
entirely serial would be incorrect. Static wait/copy counts do not establish
the runtime degree of overlap, transaction efficiency or stall cause. Original
IR, PTX, SASS, resource logs and all counts are retained, with no selected-window
deletion. Cubins remain locally available with published hashes; generated
text is committed. The report does not substitute P051 metadata for these files.

**Outcome:** the encoded branch changes operand layout/staging and resource
requirements materially. It does not switch away from BF16 matrix instructions
or spill registers in this offline build. This is a stronger, narrower finding
than either “the decoder is expensive” or “compression is memory-bound.” The
exact judge compiler options/revision/toolchain identity remain unverified;
we have no H200 execution of these offline cubins and no new correctness or
speed result. P046 remains selected.

## Next decision, not another official check

A compiler-only repair prototype now has a concrete target: reconstruct BF16
in the consumer's dot-operand layout, testing whether the decoded local_alloc /
transpose / reload and permute expansion disappear while geometry, numerical
bits, FP32 accumulation, split partition and reducer remain fixed. Predeclare
the gate: verify exact decoding/address models first, then require removal of
that intermediate staging with no spills and reduced register/instruction
requirements before considering a GPU correctness screen. If lowering retains
the same staging or creates a different equally costly conversion, reject the
prototype before a free check. An explicit-layout implementation may require
additional library/source study; no such candidate is claimed ready here.

Even that repair alone has no demonstrated competitive margin. Applying the
P061 best stored-byte fraction 0.7672543526 to **all** P062 raw latency with zero
decode cost models target/P046 ratios 1.00022/0.98613/1.08202 and whole score
**1.18060×**, below P046 1.18414×. This generous scenario is not a physical bound:
a new consumer architecture could improve other costs. It does show why a
decode-only speedup over P043 is insufficient. A credible later optimization
must establish a competitive raw consumer and broader coverage before a claim
of leaderboard-gap progress. P050's parked zero-bias QKV lead remains low
ceiling (~+0.968% for a hypothetical uniform 5% cut across five targets) and its
broader confirmation gate is not waived by Modal expiry.

No new official check is justified merely by these static results. The next
supported work is the compiler-only layout gate above; no quota is consumed to
print unchanged logs. Offline access is now operational, so **Modal expiration
is no longer a blocker for this compiler investigation**. Native correctness
and performance still require an actual GPU and a substantive reviewed payload.

## Reproduction and cleanup

After the preparation command above, place the wheel named in
`evidence/p063/dependency.json` in `local-notes/p063-offline/`. Its public URL,
expected size and SHA256 are recorded there; verify the digest before use.
Create `local-notes/p063-offline/artifacts/` for the writable child mount, then
use the verified image and wheel:

```sh
docker run --rm --name okc-p063-offline --platform linux/arm64 \
  --network none --cpus 2 --memory 4g --pids-limit 128 \
  --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,exec,size=2147483648 \
  --mount type=bind,src="$PWD/local-notes/p063-offline",dst=/work,readonly \
  --mount type=bind,src="$PWD/local-notes/p063-offline/artifacts",dst=/work/artifacts \
  --workdir /work --env PYTHONDONTWRITEBYTECODE=1 \
  --env TRITON_CACHE_DIR=/tmp/triton-cache \
  python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 \
  timeout 240s sh -c 'python -m pip install --disable-pip-version-check --no-cache-dir --no-index --no-deps --target /tmp/p063-deps /work/triton-3.8.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl && PYTHONPATH=/tmp/p063-deps python -B /work/projection_p063_offline_compile.py --compile /work'
python -B assess_p063.py
python -B -m unittest tests.test_p063_assessment tests.test_projection_p063_offline_compile -v
```

The final nine local source/parser/model checks passed in 0.035 s. The artifact reader
verifies generated-file hashes. Independent review confirmed the dataflow and
resource interpretation; it caught a static disassembly parser that omitted
uniform predicates and operand-free NOPs, now fixed and covered by a regression
case. Review also caught the assembler resource logs missing from the hash map.
Their hashes were added to both retained manifests without changing generated
files; the reader now rejects missing or tampered logs. The helper records these
hashes on future runs. This integrity-only postprocessing does not change the
execution source identity (`fe2c034`). No measurements were removed or recomputed
to change the outcome.

Both local attempts are terminal and their containers removed. OrbStack was
restored to its initial stopped state after verifying no containers or Linux
machines remained. Dependency and image caches remain local for reproduction.
No GPU, Modal,
credential or official-quota action occurred; paid cost is $0. The official
Projection quota was not refreshed as part of this local investigation; the
last saved P062 snapshot remains 19 checks left at 22:24 UTC. P046 source is unchanged.
