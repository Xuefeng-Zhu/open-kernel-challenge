# Independent common-PV compile result and receipt review

**PASS for the predeclared compile mechanism/static resource gates and for
preserving these compile results.** No numerical, frequency, timing, official
or working-selection conclusion follows automatically. A subsequent runtime
stage still needs a separately reviewed frozen harness and coordinated slot.
This reviewer used local reads/offline analysis only, with no service, GPU,
network, browser or credential action. Only the new analyzer, derived JSON and
this review were written; frozen source and native artifacts were not edited.

## Source binding, bounded transport and exact units

App`ap-dUTLyzmcwfCtweeterml8r` used preparation freeze
`6df535d096aec0e76379b6795bb97b5956d0474b`. All six receipt source/payload/
controller hashes match current bytes and exact frozen Git bytes. Control is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
common-PV candidate is
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.
Worker is`bef01dd950ce3955fb45316c5f5ee8937ea85a86c471b30c8639aeb0647e02e2`
and controller
`1c9256c05ba907e0f5044599a584304eb1cd7c32fe67fc9c6867481e8c75caaa`.

Raw native result SHA256
`de0efb07a8e46e6e261d16db249668619e7d7083c17c599478db5f2aff861e90`
matches the transport receipt. The published launch receipt is byte-identical
to its local original, including initial stopping/tasks1. The offline analyzer
verifies the bounded gzip bundle through the frozen stdlib-only controller
helper, then checks all24 decoded artifact bytes, per-file hashes/sizes, unit
maps and saved manifest. Gzip is395,655 bytes and uncompressed JSON2,647,471
bytes, below the declared bounds. No archived file is executed.

The exact four units are producer/merge for both sources with equal options:
full B32/L1 TP8,16Q/2KV,G8,D64,M16/N64/BLOCK_L16,8192-token chunks,
SPLITS11, separate merge, PAGE_LOOKAHEAD true, four warps/three stages; merge
BS16. Logical grids(32,2,11)/(32,16), runtime scalars and strides2304/86001
match the source-bound compile fixture. Warmup and module loading used typed
one-element pointers, not executable data. Recorded kernel_launches0,
correctness_tested=false, performance_tested=false and semantic_admission=false
match the reviewed no-invocation worker. Compilation is not a correctness test.

## Resource and buffering gates pass at this specialization

| Same-job unit | Registers | Shared bytes | Driver raw spill field |
|---|---:|---:|---:|
| Both producers |96|20,480|0|
| Both merges |27|1,024|0|

The fresh candidate remains at the same-job control's96 registers. Static PTX
local accesses and SASS LDL/STL inventories are zero. Both TTGIR producers have
separate two-slot E4M3 K and V shared allocations, four loop-carried async tokens,
six global-to-shared async copy sites and six commit sites. Both PTX wait-group
parameter sequences are[2,0], with two wait instructions. Direct V dataflow uses
its rotating async shared buffer and token before BF16 PV conversion, rather
than the rejected variant's synchronous global V materialization. Equal metadata
alone was not used as proof of restored buffering.

Merge SASS bytes are exactly identical, SHA256
`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`.
Other debug-bearing formats are not falsely called byte-identical. The previous
e66 variant and its96→128/V-buffer-loss rejection remain separate immutable
history. This observed restoration supports the narrow lowering hypothesis;
it does not identify a unique internal register-allocation cause or prove
identical cycle scheduling.

## Real arithmetic suppression, with issue-cycle uncertainty

The candidate TTGIR valid-row predicate compares computed alpha with1 using
unordered inequality, masks `row_ok`, reduces violations and tests nonzero.
Its multiply-only conditional returns either old carry×alpha or unchanged old
carry. The subsequent common historical BF16 PV dot uses that selected value
as C. There is no dot inside the conditional or PV-from-zero plus final carry
addition. Both complete producer PTX files retain ten BF16 MMAs, including the
same common historical sequence and unchanged last-split tail. Denominator,
row maxima, exponentials/conversions and new-KV ownership retain their original
dataflow; false-route preservation remains source-only in this first probe.

Final SASS uses eight **@P0 FMUL** instructions at0x2770,0x2790,0x27a0,0x27b0,
0x27d0,0x27e0,0x27f0 and0x2810. P0 comes from the valid-alpha reduction. The
fast condition suppresses their arithmetic/destination updates; it is not an
eager multiply whose result is discarded by selection. There is no final SASS
branch jumping around all eight instruction slots. Predicate-off instructions
may still incur issue/control cost, so neither skip frequency nor static counts
can be translated into fixed cycles saved.

Total PTX shuffle instructions rise38→41, while barriers remain18. That static
predicate overhead can offset any execution/dependency saving. Restored static
resources and logical carry order do not establish native signed-zero/subnormal/
NaN behavior, universal bitwise equivalence, achieved occupancy or performance.
The artifact/semantic reviews and direct IR/SASS checks agree within those
limits. Only original-oracle/replay checks and later uninstrumented paired
whole-call timing can answer the separate runtime questions.

## Closure and cutoff accounting

Saved independent22:28:32.831668UTC reconciliation confirms the app stopped
at22:28:11UTC, tasks0, with all listed apps stopped/tasks0. The initial controller
stopping/tasks1 receipt is preserved and separately reconciled. Worker exit0
and compiled status are distinct from the controller's initial closure exit1.
Worker30.659310397s and client47.2394720420707s are diagnostic wall times, not
kernel timings. The parent released the slot; this reviewer queried no service.

At the saved22:36:23.368320UTC accounting cutoff, the app-ID set is unchanged,
all apps are stopped/tasks0 and no new app is observed. Decimal arithmetic
reproduces rounded workspace usage delta$0.05, detailed delta$0.05020508 and
paid$0 from launch/cutoff billing objects. The recorded live-rate formula
`ceil(client_wall_seconds+5)*rate*2+0.02` gives conservative allowance
$0.1583476666666666666666666667, below reserve$0.40. Delayed workspace usage
is not an exact per-app invoice; initial zero metering was not called free
execution, and later measurement charges are excluded by the cutoff.

## Narrative and admission decision

`COMPILE_RESULT.md` accurately distinguishes this compile-stage pass from
runtime correctness, frequency, latency, score or promotion. It explicitly
explains predication and unproven issue-cycle savings. The working source/hash,
prior official1.233633059× job/interval, A2-11 rollback and Projection records
remain unchanged at this review. No official check or formal entry followed
from the compile experiment.

The analyzer reconstructs the complete actual record and returns
resource_gate_passed=true and compile_mechanism_gate_passed=true, with
onward_admission=false and execution_requires_separate_review=true. This means
there is no automatic runtime invocation or performance acceptance. A later
harness may be prepared under the parent's conditional project authority, but
must bind its own reviewed source, diagnostics, correctness gates, free-credit
bounds and fresh serial slot. This review does not execute or reserve it.

## Reviewed hashes

Paths are relative to this probe.

| Artifact | SHA256 |
|---|---|
| `COMPILE_RESULT.md` | `469b61d4ff3784014a5a2d4eb20b66b51a1812d0836bc4311388bc97d1257b70` |
| `native/ADMISSION.md` | `e08e11e089f89ed7a40eef98e7b7febae5b5db7689a3e8e0e384521504a07582` |
| `native/analyze_result.py` | `8c99653e7b1ce016f0bfd62cd47b449d85c2cf058e36d4e224abb11c3a7b0875` |
| `native/analysis.json` | `300cd51877052046b292cc68384eb8d5fab709575ed65211f1d15b0bf4d9d143` |
| `native/ARTIFACT_REVIEW.md` | `eefe9f1b48e425feebfd000483181f9d6980d6b285c9551928b07840b4dfef0a` |
| `native/SEMANTIC_ARTIFACT_NOTES.md` | `338688fd81e36807e8dd7199ce62fb58a0bb3aad58bf3f533c55e82ca044fda4` |
| `native/artifact-review-evidence.json` | `39a23820d050da53c976a4d6d7c508fa0759c218e5b48687f9a1988d2453010a` |
| `native/result.json` | `de0efb07a8e46e6e261d16db249668619e7d7083c17c599478db5f2aff861e90` |
| `native/launch-receipt.json` | `b90f1feb9a31cca3c6e8ee21061f68eb9e23e676f74082e3c4e854eed986557a` |
| `native/reconciliation.json` | `41c22824053de0cf808e97d5b0c183e3d3c3c0c9009429bdb354105d07e00772` |
| `native/COSTS.json` | `4049853db6aefc5f41d518fb4aea8aecefcde5a35d275f3db8f09f38263f6f61` |
| `native/accounting-cutoff.json` | `95d20149a7b639f77910f566709ffee27656c8e0e1ec668e7e306d6585ae4e4a` |

No blocker was found to privately preserving these reviewed compile results.
Remote export verification and any separately reviewed runtime phase remain
subsequent parent actions, not actions performed by this reviewer.
