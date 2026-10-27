# Full L1 M64N8 producer: stopped at the declared static resource gate

The four-path integration is **not admitted to GPU or official checking**.
Four selected controls compiled; the first candidate, B32 full, passed the
automatic resource screen. B1 sliding then failed the predeclared absolute
zero-stack/local rule. The helper stopped before compiling the remaining two
candidate specializations. Source, options and gates were not changed after
native counts were observed. Selected score **1.2336330588627955** and A2-11
rollback remain unchanged.

This is a static admission-policy outcome, not a measured slowdown. The
sliding controls also contain the same stack/local counts, so this rejection
does not establish an added spill cost or a failure of the matrix mapping.
No original GPU numerical oracle, replay or complete-call timing was run.

## Exact integrated change

The substantive candidate retains every selected definition and adds one
scoped producer dispatch for L1/G8/HQ16/HKV2/D64/page1, sliding B1/B16/B32 or
full B32, with more than one split. S1 and all other routes retain the selected
implementation. Allocation, grid, split/chunk policy, per-head merge and M16
partial ABI remain unchanged; unused partial rows are neutral and ignored.

Historical QK/PV place keys/features on M64 and eight queries on N8. Explicit
two-slot raw FP8 K/V gather buffers feed BF16 shared operands. FP32 online
softmax/carry, Q BF16 then E4M3 rounding, cache scales, last-split new-KV
quantization/writes and the original N16 new-token tail are preserved in source.
Tail uses the original query-major MMA family. The explicit payload/address
schedule differs from ordinary Triton's automatic pipeline, so any future
runtime effect would belong to the complete integration, not orientation alone.

Independent source/model and helper reviews passed. **32 CPU tests passed
under both the Mac default Python3.14.3 and existing Python3.12.13.** These
cover source binding, dispatch/fallback, masks/split/ownership, online mapping,
tail/rounding/merge ABI, async-slot model, compiler protocol and fail-closed
resource parsing. They are not GPU numerical or buffering evidence.

| Frozen source | SHA256 |
|---|---|
| Selected control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Extracted control JIT module | `cfefc9b9a9b2b175945fe7e6990d7c5fe46efb6e752acf88683c0dee4bb05356` |
| Integrated kernel | `eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480` |
| Standalone candidate | `96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970` |
| Offline helper | `f8cc6a6b72a0ba9076d2a4ece018a4ecce2140a9161ac8aa35b4eb9e4442f5fd` |

## Same-toolchain native evidence

Both arms used the pinned cached Triton3.8 ARM64 wheel, explicit SM90 target,
four warps/three stages, matched scalar specialization/alignment and bundled
ptxas12.9.86/cuobjdump13.1.80. Python3.12.15 ran in a CPU-only Linux container.
This is a matched compiler comparison, not official judge-binary identity.

| Unit | Registers | Stack B | Spill store/load B | Dynamic shared B | Static SHARED B |
|---|---:|---:|---:|---:|---:|
| Control D1 sliding S16 |96|32|0/0|12,544|1,024|
| Control D16 sliding S9 |96|32|0/0|12,544|1,024|
| Control D32 sliding S5 |96|32|0/0|12,544|1,024|
| Control D32 full S11 |96|0|0/0|20,480|1,024|
| Candidate D32 full S11 |74|0|0/0|34,944|1,024|
| Candidate D1 sliding S16 |75|32|0/0|34,944|1,024|
| Candidate D16/D32 sliding |uncompiled|unknown|unknown|unknown|unknown|

ptxas and cuobjdump register/stack observations agree. Candidate dynamic shared
is below the40-KiB launch-request limit, and register use is below96. Static
SHARED is recorded separately; the conservative sum is35,968 B. Static
architectural capacity does not establish actual residency or overlap.

Both compiled candidates have eight M64N8 WGMMA/HGMMA static sites in history,
with16 MMA-sync sites for the separate N16 tail. This is codegen presence,
not executed counts or numerical correctness. All three sliding controls and
the sliding candidate have96 PTX/SASS local-load sites and64 local-store sites,
despite LOCAL0 and assembler spill stores/loads0. These quantities must not be
collapsed into a claim of register spilling. The independent artifact review
traces them to inlined RoPE trigonometric range-reduction arrays, with the same
mechanism in the controls; they are not established out-of-line function calls.
Actual execution of those paths remains unmeasured.

The instruction-size audit also records candidate48 static `cp.async` sites,
all4 B. Full control has6 sites16 B and12 sites8 B; sliding control has3 sites
4 B,4 sites16 B and8 sites8 B. Different priming/refill/unrolling paths make
static totals incomparable as dynamic traffic. Narrower copy instructions
are an unmeasured issue-cost risk, not proof of a bottleneck or speed loss.

## Precompiler protocol failure, preserved separately

The first disposable container exited at `validate_manifest` before importing
Triton or invoking any compiler. Python3.14 preparation and Python3.12
validation produced different `ast.dump` function digests for identical bytes.
Every other control-contract field matched. This was reproduced locally using
the existing Python3.12 interpreter; no algorithm/resource gate had run.

A fresh payload was prepared with existing Python3.12.13. The unchanged helper,
kernel, candidate, options and gates passed independent protocol validation,
then one actual compiler session made six `triton.compile` calls and produced
the six units above, stopping
on the declared resource failure. There was no native source/configuration
retry or post-result threshold change. Both container receipts and the original
validation error are retained. See `HELPER_PROTOCOL_FIX_REVIEW.md` and
`PREPARATION_PROTOCOL.json`.

## XQA comparison and limitations

The pinned [XQA mapping audit](XQA_MAPPING_AUDIT.md) establishes this transpose
as existing SWAP_AB design. Its actual paged dispatch admits8/16/32/64/128 and
rejects1. Separate warp groups, FP8 arithmetic, per-page TMA and elected-finisher
merge differ from our strict BF16, page1 and existing merge contract. It is
source precedent, not a library port, novelty or performance evidence. Public
FA3 mechanisms/defaults still do not establish current judge dispatch geometry.

The absolute zero-stack/local admission criterion rejects a behavior already
present in selected sliding control. Its engineering applicability therefore
needs calibration against common math-library behavior **before a separately
declared future experiment**. This run retains its failed status. No full-only
rescue candidate or new gate was introduced, and no official quota was spent.
GPU correctness/replay, actual stack-path execution, achieved utilization,
latency and score remain unknown. Lower registers alone do not justify promotion.

## Evidence and closure

`evidence/compiler-manifest.json` and `evidence/execution-receipt.json` retain
complete structured observations. `evidence/native-text-artifacts.tar.gz`
contains50 hashed text entries: all generated IR/PTX/SASS/logs, both receipts,
failed preparation and exact extracted control. `evidence/ARTIFACT_INDEX.json`
lists every raw hash/size plus all six local-only cubin hashes. No raw text
window is truncated; cubins and the dependency wheel remain local. The archive
was scanned uncompressed with six secret patterns, all zero.

Both Attention-only containers exited1 and were removed; a separate exact-name
Docker query confirmed absence after the compile stopped. No GPU/Modal app,
official check, credential action, new download or host installation occurred.
Paid spending is$0. This task did not stop or reconfigure the shared engine;
a later read observed it Stopped. No global Modal-task claim is made while
existing authentication is expired. Detailed closure is in RESOURCE_CLOSURE.json.
