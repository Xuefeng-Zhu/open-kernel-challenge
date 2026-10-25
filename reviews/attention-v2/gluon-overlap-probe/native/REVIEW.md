# Independent compile-worker and controller review

**PASS for the bounded compile-only protocol. No concrete code blocker found.**
This review permits only the separately admitted, exact frozen two-arm compile
probe. It establishes neither native synchronization, numerical correctness,
execution overlap nor performance. The parent must still perform the actual
committed-head preflight, live guardrail checks and serialized admission.

The reviewer used saved local files and CPU-only tests/helpers. No Torch,
Triton/Gluon or CUDA-driver import, service, GPU, network, credential read or Git
subprocess occurred. No kernel, worker, resource or controller source was edited.

## Matched units and compile-only behavior

The source/config pair is the exact pair covered by `../KERNEL_REVIEW.md`.
Both specializations use B32/HKV2/HQ16, L8/G8, M64/N64/D64/page1, grid(32,2),
four warps/one stage, stride86008 and QK_FACTOR5.4730064863875. The sole option
delta is OVERLAP and the sole specialized loop AST delta is QK-wait placement.
The standalone input boundary and identical zero-PV then `acc*alpha+PV` order
are explicit; no equivalence to another carry-C lowering or production FA3 is
assumed. Source-level precision, scale, causal and lifetime review is separate
from compilation acceptance.

The default worker only hashes/parses reviewed sources and imports the CPU
configuration. Native imports are inside the explicit `--compile` branch,
after the pinned Triton3.8.0/three installed-file identity gate. There is one
warmup call site in the fixed two-arm loop and module-handle initialization;
there is no subscript launch, kernel.run, remote call or benchmark. The typed
one-element pointers are deliberately undersized and must never be invoked.
No fallback install, source discovery, precision or geometry rescue exists.

Supported CUDA-driver metadata queries occur only in the native branch after
module load. They do not invoke the attention kernel. Header enums and actual
device/function limits must be available; missing or inconsistent observations
stay inconclusive. Exactly two units, serial.core and overlap.core, are reported.

## Resources and evidence classification

The resource helper rejects nonzero driver spills, local-memory bytes or
PTX/SASS local accesses, shared memory above64KiB, or an overlap-arm reduction
in CUDA-permitted maximum CTA/SM. Both arms use the same actual device/block
limits. The occupancy API receives the compiled function, actual warp-derived
block size and launch dynamic shared allocation; static function shared memory
is also reported. This is a legal maximum, not achieved CTA residency.

The old96-register cap is correctly absent for this independent M64 design.
Missing SASS, CUDA limits, occupancy or resource metadata never passes the
gate. Compilation may complete with rejected/inconclusive resources, which is
a valid mechanism result. Mnemonic counts alone leave mechanism review pending:
the exact next-tile load/conversion/shared stores, current QK dependency wait,
publication, slot ownership and prior PV drain require independent artifacts.
No resource/compile pass automatically admits invocation, correctness, timing,
official quota or promotion.

## Controller and transport

The controller uses six explicit uploaded files (kernel/config/resources/worker
and two license files), plus a local committed-controller hash. Credentials,
receipts and this review are absent from the closure-free remote payload.
Existing supported credential configuration, paid cap0, verified free-credit
floor above the fixed$.40 reserve and an exact fresh serialized-slot receipt
are checked before one exclusive request ledger is created.

Bounds are90/100/115/20/140 seconds for worker/child/function/startup/watchdog,
one H200, two CPUs,8GiB host memory, one container/invocation and retries0.
The child receives `--compile`. Stop and stopped/tasks0 verification are explicit;
uncertain terminal outcomes require reconciliation rather than another launch.
This reviewer did not exercise any of those service actions.

TTGIR/PTX/CUBIN are required for both arms; TTIR/LLIR/SASS are optional transport
formats, with absent SASS making mechanism evidence inconclusive. The fixed
twelve-name allowlist,2MiB result,8MiB raw bundle and4MiB per-file limits agree
between worker/controller. Strict base64, compressed/raw/per-file hashes and
sizes, bounded decompression/EOF framing, duplicate-key rejection and exclusive
flat output writes reject tampering, traversal, truncation, concatenation and
decompression overflow before artifact persistence.

## Independent CPU verification

All30 focused methods passed:9 source,9 worker/resource and12 controller tests.
The default worker CPU preflight was independently exercised by its author;
this reviewer additionally called the real source/config helpers.

The review also connected the actual worker artifact packager to the actual
controller decoder/completion validator using binary/text fixtures for both
six required formats and all twelve allowed files. Both round-tripped exactly.
All six real payload files plus the controller passed the parse/hash binding
helper with only Git responses mocked from current bytes; credential/network
functions were patched to fail if called. This is CPU integration evidence,
not the still-required real committed-head preflight or native compilation.

The final card correctly predeclares machine-code ownership and resource gates.
README correctly separates the worker from the separately reviewed
native/controller.py, which governs admission.
No FA3 novelty or missing-overlap claim follows while its exact backend remains
unknown; that unknown does not block this narrow primitive-viability probe.

## Reviewed SHA256

Paths are relative to this probe directory.

| Artifact | SHA256 |
|---|---|
| `kernel.py` | `3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c` |
| `config.py` | `ecaf6588893ad4ce6bc57944a9932018f511113aeb13a6ee266edcf536d701ec` |
| `test_source.py` | `143e80c007c311ff3d89e807d4b3b455c4fe4fabe24b6c73b97ee0092d4a57d2` |
| `CONFIG.md` | `e0061148b2b77cb278282c56baac27930c7e5be3d0e9348dff4ce5142ca01db1` |
| `KERNEL_REVIEW.md` | `491e1fb3789848576606fa351b2cd9c13ac4779cacdd5925944c1a86d28f6cb3` |
| `EXPERIMENT_CARD.md` | `6e1d758d4306a1e0510353e4062cf50b8ce04b17f893313c7f1a2cb408c0efa6` |
| `README.md` | `9c81382a0e15c24085e3b3096b2f7babe545bf29969b73805d8744dcab89aa39` |
| `native/worker.py` | `608cc5769883baa204c817773da80b1841eab1837c12daef7ef9a2b1518b4a39` |
| `native/resources.py` | `48adfe47cdb9ca72890b93c816772e454552cf634e5f771cd23e3a14234fbdee` |
| `native/test_worker.py` | `9f3ae1ee8125bd2c32765f14d453a45b3c005f3ffcb3ec2ae6f6c5f06dfb5aa7` |
| `native/controller.py` | `3c5cf83169e717dbc9cb1a24d5a2aadcbe1f8a44e7652ef2212186ce8546ab8a` |
| `native/test_controller.py` | `0b9ea170d578065854c3a4b3b0b6db545c90cc7ed82f26cbb8c192f1634b1722` |

Installed API identity is bound to saved inventory91c05d95…734d78 and Hopper
c427a1f7…a99dc. Complete API bodies/layout lowering remain native questions.
