# Independent local source/API review: PASS

Reviewer: `/root/transpose_compile_source/gluon_api_check`. Read-only review
and independent execution of all seven CPU tests passed (0.033 s). No source
or API-level blocker was found. This review is before compilation and does
not admit generated code or an executable attention candidate.

Kernel SHA256:
`118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033`.
Offline helper SHA256:
`e7a489a899d7eae222250ca546d00bfefbb3435cf4ad25c8a79c96d0a099fc37`.
Tests SHA256:
`efa9576cf2c24b48cdeb27b472224e84e1f25a11eefdabc54ef2db1eabb19067`.

The matrix/index contract is QK K[64,64]×Q-transpose[64,8], then per-query
FP32 max/exp2/sum over key axis0, BF16 P, and PV V-transpose[64,64]×P[64,8].
Output stores transpose the [feature64,query8] result to [query8,feature64].
Q BF16→E4M3→BF16, BF16 K/V/P and FP32 dot accumulators are explicit. Both
MMA calls preserve `max_num_imprecise_acc=0`; scale factors match the declared
source contract. There is no full producer, tail, carry, append or dispatch.

Pinned-wheel inspection confirms direct `GluonASTSource` accepts the explicit
SM90 target without runtime binder/driver discovery. Shared A/B are exposed
by the Python API; B is required shared. `NVMMASharedLayout` transposed[64,8]
chooses K-contiguous128-byte swizzling. Python layout construction has no N8
restriction; native compiler verification remains untested. The single-dep
`warpgroup_mma_wait` returns the tensor as used in this source.

The helper never allocates GPU tensors, initializes handles or launches a
kernel. Native admission must verify BOTH separate QK/PV 64x8 WGMMA sites,
actual operand/accumulator types, no hidden query padding/MMA-sync fallback,
and zero stack/spill/local traffic using all generated IR/PTX/SASS and ptxas/
cuobjdump evidence. Textual opcode inventory alone is not that review.
Registers and shared launch request are recorded; no tiny-probe resource
cutoff is presented as whole-producer parity or achieved occupancy.

Compiler provenance must include the root's independently verified cached
wheel digest and install receipt. The helper checks three API member hashes
and tool hashes, but only declares the full wheel SHA; it does not by itself
hash libtriton.so and every compiler backend. Root's hash-verified wheel
installation completes that dependency binding. No official judge binary
identity follows from matching the public version string.

The reviewer performed no Docker, remote, authentication, network, GPU or Git
action. Projection files and containers were not modified.
