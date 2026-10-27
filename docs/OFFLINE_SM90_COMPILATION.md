# Reusing the CPU-only SM90 compiler on the Mac

This is native code generation and disassembly, not GPU execution. P063 proved
this route works without a CUDA driver. It does not establish an identical
compiler/toolchain/options configuration to the official judge.

Use Python 3.12 on Linux ARM64, the official image
`python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70`,
and the official PyPI Triton 3.8.0 wheel described by
`evidence/p063/dependency.json` (including its URL, size and SHA256). The existing
Mac cache is `local-notes/p063-offline/` in the Projection worktree. Wheel SHA256:
`a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50`.
This contains ptxas 12.9.86 and cuobjdump 13.1.80. Do not install Torch or a CUDA
driver merely for this route.

## Per-track isolation

Attention must use its own staging directory, unique container name (for example
`okc-attention-offline-<experiment>`), writable artifact child mount and temporary
compiler cache. Copy the immutable wheel or read it through a dedicated read-only
mount. Do not share Projection's writable staging/artifacts, rewrite its source,
run concurrent Git mutations, or mount a repository/home/credentials/Docker socket.
Local CPU compiles do not reserve an H200 slot. Coordinate engine shutdown:
P063 restored OrbStack to its initial stopped state; a task must not stop it while
another task has an active container. Check inventory before and after its run.

`projection_p063_offline_compile.py` demonstrates explicit `ASTSource` plus
`GPUTarget("cuda",90,32)`. Reuse the approach, not Projection-specific kernel
constants/signatures. Extract only reviewed owned kernels, include every required
helper/constexpr explicitly, freeze source hashes and alignment assumptions.
Do not call `create_binder`, `_init_handles`, driver APIs or `kernel[grid]`.
`CompiledKernel.asm` provides TTIR/TTGIR/LLIR/PTX/cubin before device loading.
Get register/spill information from the bundled ptxas report and cuobjdump;
device-backed `n_regs` properties are inappropriate here. Do not mutate the
compiler IR to hide benchmark work or substitute fake device availability.

## Bounded run template

Prepare the source/manifest/helper, verify wheel hash, and create `artifacts/`.
Mount the staging directory read-only and only its artifact child writable.

```sh
docker run --rm --name UNIQUE_OFFLINE_NAME --platform linux/arm64 \
  --network none --cpus 2 --memory 4g --pids-limit 128 \
  --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,exec,size=2147483648 \
  --mount type=bind,src=ABSOLUTE_STAGING,dst=/work,readonly \
  --mount type=bind,src=ABSOLUTE_STAGING/artifacts,dst=/work/artifacts \
  --workdir /work --env PYTHONDONTWRITEBYTECODE=1 \
  --env TRITON_CACHE_DIR=/tmp/triton-cache \
  python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 \
  timeout 240s sh -c 'python -m pip install --disable-pip-version-check --no-cache-dir --no-index --no-deps --target /tmp/deps /work/VERIFIED_WHEEL.whl && PYTHONPATH=/tmp/deps python -B /work/REVIEWED_COMPILER.py --compile /work'
```

All placeholders are explicit. No GPU flags/devices, network during execution,
cloud invocation, account or credential operation is needed. Record failures
and exit status; verify the exact container is absent after `--rm`. Keep hashes
for all evidence, including resource logs. Static instructions, registers and
shared-memory requests do not measure dynamic counts, occupancy or performance.

## Gluon sources (P065)

Use `triton.experimental.gluon._runtime.GluonASTSource` for owned `@gluon.jit`
source. It marks `Language.GLUON` and starts at TTGIR; ordinary `ASTSource`
selects the wrong pipeline. P065's reviewed helper demonstrates this and has
a regression test. Absence of TTIR is expected. Gluon also does not inherit
ordinary Triton's automatic coalescing/layout/software-pipeline passes: a shared
`num_stages` value is not evidence of equal buffering. Preserve matched controls.
P065's direct-operand layout removed the decoded shared roundtrip but failed
real performance; offline resource improvement alone is never a promotion gate.
