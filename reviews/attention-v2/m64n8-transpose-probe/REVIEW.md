# Independent precompile source review: PASS

Reviewer `/root/offline_transpose_review` inspected the frozen kernel,
offline compiler harness, seven CPU tests and prospective experiment card.
The seven tests and an actual `--prepare` invocation to an independent `/tmp`
directory passed. No Docker/OrbStack, network, service, GPU or Git operation
was performed by this reviewer. Only this review document was written.

## Exact reviewed sources

| File | SHA256 |
|---|---|
| `kernel.py` | `118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033` |
| `offline_compile.py` | `e7a489a899d7eae222250ca546d00bfefbb3435cf4ad25c8a79c96d0a099fc37` |
| `test_offline_compile.py` | `efa9576cf2c24b48cdeb27b472224e84e1f25a11eefdabc54ef2db1eabb19067` |
| `EXPERIMENT_CARD.md` | `15dd2cdb60a0a7dd2edc784d0cf6fb602d2a52325a8ef259c9eda923c472c1f8` |

The unchanged selected source is independently bound by preparation to
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The kernel ABI is Q BF16; K/V E4M3; scalar scales, raw attention, maxima and
sums FP32. Pointer alignment is declared 16 bytes. Four warps/three stages,
target `cuda:90:32`, QK factor5.4730064863875 and softmax scale0.125 are fixed.

## Source and API findings

The source requests K[64,64] × Q-transpose[64,8], then
V-transpose[64,64] × P[64,8]. The native output layout requested by both is
64×8 with NVMMADistributedLayout version3 and instruction shape16×8×16.
The eight queries are not expanded to16/64 in the source mathematical state.
Whether native code retains that property remains a compile-artifact gate.

Q follows BF16→E4M3→BF16 before consumption; K/V widen from E4M3 to BF16.
Both MMA calls request FP32 accumulator state and max_num_imprecise_acc0.
The score is FP32, including scale and log2 conversion. Max/sum reduce axis0
of the 64×8 score/probability tensors: all64 key rows participate per query.
P rounds to BF16 before PV; output/scales/normalization remain FP32. The
mathematical index test agrees with a conventional query-major calculation,
but it is not a floating-point GPU correctness test.

K, Q-transpose, V-transpose and P have separate shared descriptors. K/Q/V
are stored before the first shared publication fence. QK is waited to zero
outstanding before scores are consumed. P is stored and fenced before the
synchronous PV call. No source overwrite or released descriptor is visible
while an asynchronous consumer remains pending. The compiler's actual alias
reuse, cross-warp reduction/barriers and shared publication still require
TTGIR/PTX/SASS inspection; source fencing alone is not a native lifetime proof.

The existing cached Triton3.8 ARM64 wheel was read as a ZIP, without installing
or importing it. Its `_runtime.py`, `_layouts.py` and Hopper module hashes match
the harness's pinned API hashes. `GluonASTSource` accepts the supplied positional
signature/constexpr/attrs arguments; its constructor selects Language.GLUON.
The shared-layout helper accepts `transposed=True` for the requested shapes.
Those API facts do not establish that the native verifier accepts N8 or every
descriptor orientation.

## Offline execution boundary

Preparation parses and hashes owned source without importing it. Compilation
requires Linux, rejects visible NVIDIA devices, verifies source/helper/API
hashes, and invokes `triton.compile` with an explicit GPUTarget. It does not
use create_binder, driver.active, launcher calls, `_init_handles` or device-loaded
`n_regs`. GPU availability is not faked. The generated module contains only
owned Gluon code/imports, with no host kernel call or tensor allocation.

The harness retains IR/PTX/cubin, assembler logs, static resource metadata and
SASS through the bundled tools; each artifact is bounded to8MiB. Tool subprocesses
have fixed timeouts and the whole helper has a240-second alarm. The parent owns
container isolation, cached-wheel verification, mount/privacy restrictions,
aggregate transport handling and cleanup; this review does not certify a
container invocation that has not occurred.

## Decision limits

There is no precompile source blocker. The frozen probe is ready for its bounded
offline compile diagnostic, not for a correctness/performance run or submission.
Native gates remain: both true64×8 operations, no hidden query padding or
mma.sync/scalar fallback, strict precision, complete cross-warp reduction and
PV publication, and zero spill/local traffic. Missing native evidence is
UNKNOWN, not PASS. Physical register/shared counts must be retained;128/32KiB
are reference values, not production admission thresholds.

This probe contains one unbuffered full historical tile. It proves nothing
yet about online carry, next-tile overlap, new-token tail, cache append,
partial merge, complete producer resources, runtime correctness or speed.
The later96-register parity/buffering/performance gates remain separate.
Selected1.233633 and A2-11 rollback are unchanged.
