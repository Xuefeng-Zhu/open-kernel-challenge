# Phase 1: compile-only mechanism falsifier

This phase compiles two exact sources with identical geometry and retrieves
native artifacts. It never executes `kernel_fn`, the producer or the merge,
and supplies no correctness, frequency or performance claim. The working
candidate remains the control until later independent gates pass.

| Source | SHA256 |
|---|---|
| Working composition control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Strict historical numerator candidate | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |

The only requested specialization is the scored custom full B32/L1 TP8 path:
16 Q/2 KV heads, G8, D64, BLOCK_M16, BLOCK_N64,8192-token chunks,11 splits,
four warps and three stages. Compile each producer and its unchanged BS16
merge: four units,24 artifacts covering TTIR/TTGIR/LLIR/PTX/CUBIN/SASS. A
PAGE_LOOKAHEAD-false source reconstruction is already tested locally; this
first job does not compile or claim native evidence for that other path.

## Predeclared interpretation gates

1. Exact source, target, signature, precision, launch geometry and complete
   artifacts must be present. Missing tools, timeout, compile failure or
   partial transport is a precise blocker, not a successful mechanism test.
2. TTGIR/PTX/SASS must show the exact computed-alpha predicate over valid rows
   and a real control/predicate path bypassing old-numerator rescale. If the
   candidate still executes every carry multiply, reject this mechanism.
3. On each executable branch, PV must retain the control's old-carry C operand
   and matrix accumulation order. Fast C is the existing old accumulator;
   fallback C is its unchanged alpha product. A dot from zero followed by a
   scalar addition, duplicated executed PV, changed conversions or precision
   fails the strict contract. Static mutually exclusive code duplication must
   be distinguished from executing twice.
4. Rowmax, exponentials, denominator FMA/reduction, lookahead K/V pipeline and
   last-split new-KV tail must remain unchanged in arithmetic and ownership.
   A source-level identity is insufficient if native lowering breaks carry
   order or the established lookahead buffering.
5. No new spills, local-memory accesses or increased producer register count
   are admitted to the next phase. This conservative resource falsifier avoids
   accepting a register/occupancy trade based only on a compile pass. Record
   shared-memory, barriers/shuffles, code size and async buffer changes; loss
   of established buffering or a reduced static resident-block resource bound
   also rejects this first variant. These are static resource observations,
   not achieved occupancy or stall measurements.
6. Multiplication/FTZ and signed-zero behavior must be consistent with retaining
   the same valid-row carry bits. Do not assert universal bitwise equivalence
   from finite arithmetic algebra, archived PTX or a passing compile. Native
   original-oracle and exact observed cross-arm checks remain later gates.

If a gate fails, preserve the actual evidence and reject this source variant;
do not tune a rescale threshold or silently change the candidate. If the gate
passes, prepare a separate source-bound diagnostic to count fast/slow historical
tile visits. Instrumented times are never benchmark evidence. Frequency and
predicate cost must justify proceeding to the experiment card's original-
oracle, fresh-input, uninstrumented whole-call timing gates. No automatic
official check or working-selection promotion follows from compile success.

## Execution bounds

One existing-image H200 app, two CPUs and8GiB host memory, one container and
invocation, zero retries. Bounds are90s worker,100s child,115s function,
20s startup and140s watchdog. The free-credit reservation is at most$0.40,
admitted only when twice live rates over the full watchdog plus$0.02 fit inside
it. Actual paid cap and current paid spend must both be$0. Existing credentials
are selected through supported configuration; no renewal or copying.

The allowlisted payload has only worker, two exact source files, license and
third-party notices. The controller is hash-checked locally, not uploaded.
`cuobjdump` is used only if already installed, with a10s command timeout; no
counter tool, package installation or network is introduced remotely. Gzip/
base64 transport is bounded to2MiB, decompressed JSON to8MiB, and exactly24 flat
artifact names with verified hashes/sizes. Validate all entries before exclusive
artifact writes, and reject duplicate keys, path traversal, truncated/multiple
gzip streams or decompression overflow.

No launch is authorized by this file alone. The active parent-coordinated
serialized slot, completed local review, committed payload and fresh free-
credit/$0 guardrails are necessary. After Projection P046 reached terminal,
the parent granted this single compile-only probe the serialized slot.
Stop and independently verify app stopped/tasks0 before release; preserve any
initial incomplete closure rather than rewriting it. Record costs with a
closed cutoff before later measurements.
