# M64/N8 transposed attention tile: offline mechanism probe

Local preparation only. This is not `kernel_fn`, a contest candidate, a full
attention implementation or a performance result. It does not alter selected
source `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The proposed geometry places keys/features on native M64 and the eight L1 GQA
queries on N8. One historical tile computes K64x64 times Q-transpose64x8,
FP32 max/exp2/sum over the key axis, BF16 probabilities, then V-transpose64x64
times P64x8. Encoded Q is rounded BF16→E4M3→BF16; encoded K/V widen exactly to
BF16. Both dot accumulators are FP32 and use `max_num_imprecise_acc=0`. QK
factor5.4730064863875, softmax scale0.125 and live K/V scales follow the source
contract. Raw output is FP32[8,64] after single-tile denominator/V-scale; row
maximum is in log2 score units and row sum is FP32. There is no output gate.

Both operands of each WGMMA are shared descriptors. Four logical allocations
sum to18KiB: K[64,64], Q-transpose[64,8], V-transpose[64,64], P[64,8]. The
transposed small B descriptors choose K-contiguous128-byte swizzling in the
pinned API. Real layouts/resources and native admission remain compiler gates.

Pinned wheel source inspection establishes `GluonASTSource` supports explicit
target construction without a runtime binder, shared A/B are accepted by the
Python wrapper, and N8 has no Python-side validation rejection. It does not
establish acceptance by the compiler's native verifier or the resulting MMA
shape. The exact Python API hashes are retained in the prepared manifest.

Hard gates were declared before compilation: both operations really emit
native64x8 WGMMA without hidden query padding or MMA-sync fallback, precision
boundaries remain strict, and zero spills/local traffic. Registers and shared
requests are all recorded;128registers/32KiB is reference bookkeeping, not a
tiny-probe rejection threshold. Full-producer96register parity, original
buffering, last-split KV ownership, N16 new-token tail, online state, merge,
RMSNorm/RoPE, correctness/replay and actual-best timing are separate later
gates. None is certified by this tile.

## CPU preparation

```sh
python -B -m unittest discover -s reviews/attention-v2/m64n8-transpose-probe -p 'test_*.py' -q
python -B reviews/attention-v2/m64n8-transpose-probe/offline_compile.py --prepare /tmp/attention-m64n8-preflight
```

Seven tests pass. They cover exact source/ABI/frozen preparation, no host
execution or device API, the key reduction axis and rounding boundaries,
static artifact inventory/size limits, and an independent ordinary-versus-
double-transposed matrix/index model at unit and1/64 scales. Python model
arithmetic is not a GPU reduction-order oracle or proof of numerical tolerance.

## Root-owned offline execution

`offline_compile.py --compile /work` is Linux-only and refuses NVIDIA devices.
It imports only the pinned Triton3.8 compiler and the owned kernel, constructs
`GluonASTSource` directly, and targets explicit `GPUTarget('cuda',90,32)`.
It never calls a runtime binder, active driver, device handles, GPU availability
shim, tensor allocator or launcher. A240-second alarm bounds the helper;
parent container controls provide independent resource/network limits.

The parent owns any Docker/OrbStack invocation and cleanup. Use only verified
cached wheel SHA256`a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50`
and the existing P063 image. This child did not start or stop any engine,
container, GPU, remote service, authentication flow or Git operation. Do not
modify Projection containers or mount repositories/credentials/home into the
compiler container. Input files are owned kernel/helper/manifest plus the
verified dependency; only its artifact directory is writable.

Outputs include available TTGIR/LLIR/PTX/cubin (TTIR only when exposed), full
assembler logs, cuobjdump resource/SASS output, compiler metadata and complete
hash maps. No file over8MiB is silently truncated. Compile rejection preserves
its diagnostic and stops the specific source; there is no tile/flag/stage sweep.
Raw cubins remain local unless an explicit sanitized artifact policy permits
publishing them. Offline compiler/tool identity is not official judge identity,
and host compilation duration is not kernel latency.
