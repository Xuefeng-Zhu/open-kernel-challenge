# Common-PV compile mechanism and resource gates passed

The narrower multiply-only conditional restores the working control's original
K/V asynchronous pipeline and96-register producer, while suppressing historical
rescale arithmetic on the exact alpha-one path. All declared compile gates pass
for this one specialization. No candidate or merge kernel was invoked. This
admits a separately reviewed bounded native validation, not a numerical,
frequency, latency or score claim.

| Unit | Control registers/shared/spill | Candidate registers/shared/spill |
|---|---|---|
| Producer | 96 /20,480B /0 | 96 /20,480B /0 |
| Merge | 27 /1,024B /0 | 27 /1,024B /0 |

Both K and V retain two-slot E4M3 asynchronous shared buffers, four loop-carried
async tokens, six TTGIR copy/commit sites and wait-depth2. Merge SASS is byte-
identical. The target has one common PV consumer; its fast C is old carry and
slow C the unchanged alpha product, with the original matrix accumulation order.
Denominator, tail and ownership remain unchanged. No spill/local-memory access
was found in the complete artifacts. The archived failed branch's128 registers
and lost V pipeline remain a separate negative result, without alteration.

The actual native skip uses **eight predicated FMUL instructions**, not a SASS
branch around their instruction slots. An exact valid-row alpha reduction
controls the predicate; the fast case suppresses arithmetic and register write.
It is not an eager MUL followed by select. Predicated-off instruction issue
cost has not been shown to disappear. Butterfly shuffles rise34→37, total
shuffles38→41; predicate/shuffle overhead can offset any arithmetic saving.
There is no performance prediction from this compile pass.

## Identity and execution scope

Frozen commit`6df535d096aec0e76379b6795bb97b5956d0474b` binds control
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
and candidate
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.
Worker SHA256 is
`bef01dd950ce3955fb45316c5f5ee8937ea85a86c471b30c8639aeb0647e02e2`;
controller is
`1c9256c05ba907e0f5044599a584304eb1cd7c32fe67fc9c6867481e8c75caaa`.

Seven source, ten worker and twelve controller offline tests plus independent
review preceded the real frozen-head preflight and one app. Four warmup units
(two producers/two merges) use identical full B32/L1 TP8 M16/N64/D64,
8192-token chunks,11 splits, four warps and three stages. E4M3 Q/K encoding
with strict accumulation, BF16 PV and FP32 softmax/partials are unchanged.
All24 TTIR/TTGIR/LLIR/PTX/CUBIN/SASS files were transported and independently
hash/size-verified. Existing cuobjdump was bounded to10s per command; no NCU,
installation or permission/credential expansion. Warmup and driver handle loads
compile functions using typed pointer placeholders; they are not executable
fixtures or correctness evidence. False-constexpr identity remains source-only.

The fixed existing image ran H200 sm90, Torch2.14.1+cu130, CUDA13.0,
Triton3.8.0 and Python3.12.3. The worker completed in30.6593s; client wall
was47.2395s. The140s watchdog,90s worker/100s child/115s function/20s startup,
one invocation and zero retries were retained.

## Closure, accounting and next gate

App`ap-dUTLyzmcwfCtweeterml8r` stopped at22:28:11UTC. Independent
22:28:32.831668UTC reconciliation confirmed stopped/tasks0 and all listed apps
stopped/tasks0. The compile-stage slot was immediately released; when P049
approached readiness, the root explicitly confirmed the execution harness was
not ready and released the slot for Projection. The initial controller stopping/
tasks1 observation stays unchanged and is separately reconciled. Worker exit0
and compile success are distinct from the controller's initial closure exit1.

At22:36:23.368320UTC, a no-new-app/all-stopped accounting read observed rounded
workspace usage+$0.05 and detailed usage+$0.05020508, paid$0. Conservative
closed allowance$0.1583476667 fits the reviewed$0.40 free-credit reserve.
These delayed workspace deltas are not exact app invoices; no later apps are
assigned to this cutoff. Existing protected authentication and paid cap$0
remained in force without renewal or settings changes.

`native/analyze_result.py`, `ARTIFACT_REVIEW.md` and independent semantic notes
reproduce source/transport/artifact identities, resources, pipeline and carry
evidence. General runtime correctness and other specializations are untested
for this new source. The working source remains the previous validated
composition, with official1.233633059× unchanged; no official quota or ranked
entry was used.

The next local preparation is one bounded original-oracle/cache/replay stage,
a separate exact-source instrumented frequency diagnostic, and uninstrumented
paired complete-call timing only after correctness and diagnostic fidelity.
The predeclared target≥2% per fresh fixture, all six favorable balanced pairs
and unchanged control≤2% slowdown remain unchanged. Frequency is not a cycle-
saving estimate. No execution is started or reserved by this report; the parent
must coordinate a fresh serialized slot when the reviewed harness is ready.
