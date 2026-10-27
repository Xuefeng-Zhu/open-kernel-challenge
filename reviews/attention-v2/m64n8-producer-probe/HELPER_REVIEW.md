# Independent offline-helper review: PASS

Reviewer: `/root/offline_transpose_review`. Scope is the helper, its CPU tests,
fixed compiler protocol and cached API/tool provenance. Kernel mathematics,
shared-layout legality and native buffering require the separate source/artifact
reviews. No Docker, service, GPU, network, authentication or Git operation was
performed. Only this review document was written; CPU tests/preparation used
temporary directories.

## Frozen helper and tests

| File or extraction | SHA256 |
|---|---|
| `offline_compile.py` | `f8cc6a6b72a0ba9076d2a4ece018a4ecce2140a9161ac8aa35b4eb9e4442f5fd` |
| `test_offline_compile.py` | `ac24e146e5969e6bb187a439cb2ed37e339b8bf6bb756126a1a5972d92bfdb8d` |
| Selected source | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Exact extracted control | `cfefc9b9a9b2b175945fe7e6990d7c5fe46efb6e752acf88683c0dee4bb05356` |

All16 final CPU tests independently passed. Actual preparation plus manifest
validation passed against the candidate bytes then present, SHA256
`eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480`.
That hash records a helper smoke test, not kernel source approval or a promise
that later source-review fixes retain it. Root must freeze the final reviewed
kernel hash into a fresh payload. No compiler invocation was performed.

## Control, ABI and specialization binding

The helper verifies selected bytes before extracting only `_norm_rope`, `_fused`
and LOG2E. Both function ASTs and the literal constant are preserved; host Torch,
SGlang, allocation and routing setup are excluded. Loading the extracted module
therefore does not execute the selected host implementation. The independently
recomputed extraction digest matches the hard-coded `CONTROL_SHA`.

The ABI remains21 runtime arguments plus16 explicit constexpr arguments.
Q/output/ticket pointer placeholders are BF16; historical K/V are E4M3;
partials/scales FP32; seq/page entries i32 and cache locations i64. The ticket
placeholder is BF16 because these separate-merge L1 routes pass the output
buffer and never execute the ticket branch. Four floating scalars stay dynamic;
page1 specializes to constexpr. Pointer and window/stride-token divisibility16
match saved selected producer TTGIR; odd page-table stride is not over-aligned.
The inspected historical ABI reference is
`../alpha-one-common-pv-probe/native/artifacts/baseline.producer.ttgir`, SHA256
`573720bb90b00b4b1f82e640373d19bbd12fbf528450295da87f1063933db894`,
line60. This is historical signature evidence, not judge-binary identity.

Fixed controls are D1slide S16, D16slide S9, D32slide S5 and D32full S11.
Grids are respectively(1,2,16), (16,2,9), (32,2,5), (32,2,11). All retain
L1/G8/HQ16/HKV2/D64, M16/N64 and N16 tail, separate merge, original sliding
RoPE flags and full B32 chunk8192/page lookahead. Scalar example values are
recorded, but dynamic floats are not silently converted into compile constants.

## Fixed execution protocol and bounds

`validate_manifest` rejects changed case/options/target/dependency/API/tool
declarations, changed control bytes and shortened/reordered compile lists.
The loop compiles all four fixed controls first, then candidates in order
D32full, D1slide, D16slide, D32slide. A compiler exception or the first failing
candidate resource gate returns immediately; no retry, alternate tile, changed
precision or rescue configuration exists in the helper. Root owns the promised
single invocation and disposable-container lifecycle; this review does not
claim a helper can prevent a separate external invocation.

Source files are bounded to256KiB and symlinks refused. Preparation requires
an empty destination. Generated IR/PTX/cubin/SASS/log/JSON writes are bounded
to8MiB, with oversized writes refused before partial output. The actual8MiB
boundary and8MiB+1 refusal are tested. Full compiler exception text is retained
as a hashed bounded artifact; a shortened JSON summary is explicitly flagged.
An oversized exception is explicitly marked refused with original size/hash,
not silently presented as complete. Required artifacts cannot be omitted.

The CLI has a maximum230-second alarm/deadline, checks remaining time before
units and external tools, and bounds tool subprocess timeouts. The root's
240-second container bound remains an independent outer limit. Linux and the
absence of NVIDIA device nodes are checked before compiler import.

Resource parsing treats missing/ambiguous values as unknown. The candidate
screen requires matching ptxas/cuobjdump register counts in1..96; zero stack,
spill, LOCAL and native/PTX local-access evidence; and dynamic shared request
in0..40960 bytes. Root explicitly defined the40KiB gate as **dynamic launch
shared**, and the experiment card now says so. Cubin static shared, their
conservative sum and a model including at least1024 static-reserve bytes are
recorded separately. Static capacity/residency feasibility still needs review.

`mma.sync` is counted rather than globally rejected because the preserved N16
tail intentionally uses MMA. The helper does not automatically declare the
native N8/dataflow gate passed from instruction totals; mainloop versus tail,
both QK/PV groups, precision and lifetimes remain independent artifact gates.

## Cached API and zero-driver boundary

The existing226474486-byte Triton3.8 ARM64 wheel independently hashes to
`a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50`.
All nine pinned API files and both bundled ptxas/cuobjdump tool digests match
the ZIP contents. The wheel was neither installed nor imported for this review.

Pinned `ASTSource` accepts string-keyed signatures, named constexprs and tuple
attribute keys; `GluonASTSource` uses that constructor and selects Gluon IR.
Explicit `GPUTarget('cuda',90,32)` bypasses default active-driver target lookup.
The compiled-kernel constructor retains assembly/metadata with module/function
unset; runtime initialization is lazy. The helper reads `.asm` and metadata,
and never calls a binder, `_init_handles`, launcher, kernel, driver query or
device-derived `n_regs`. It does not fake GPU availability.

No helper blocker remains for one bounded offline invocation after final
source/layout review and root payload freeze. This PASS is not a kernel,
native-resource result, numerical correctness, performance or GPU admission.
