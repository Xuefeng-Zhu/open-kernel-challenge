# Alpha-one branch variant rejected at the compiler/resource gate

The compile-only probe completed and retrieved all24 artifacts, but the
candidate fails two predeclared admission gates. Producer registers rose from
96 to128, and generated code lost the control's V asynchronous buffering.
No candidate or merge kernel was invoked, and no frequency, correctness,
replay, timing or official-check stage followed. This is a rejected source
variant, not a measured slowdown or a performance result.

| Compiled unit | Registers | Shared bytes | Driver spill field | Result |
|---|---:|---:|---:|---|
| Control producer | 96 | 20,480 | 0 | Same-geometry control |
| Candidate producer | 128 | 20,480 | 0 | Register and buffering gates fail |
| Control merge | 27 | 1,024 | 0 | Control |
| Candidate merge | 27 | 1,024 | 0 | SASS byte-identical to control |

The +33.3% register increase alone violates the conservative no-increase gate
in`native/ADMISSION.md`, frozen before the run. Zero spills and equal total
shared bytes do not undo that failure. No achieved occupancy, CTA residency,
stall reason or latency was measured, so the resource change is not presented
as proof of a hardware bottleneck or an actual timing regression.

## What the native code establishes

The exact computed-alpha predicate ignores padded query rows and generates a
real branch bypassing historical old-numerator multiplies. Each mutually
exclusive PV branch uses the expected old-carry C operand; no threshold or
deferred maximum was introduced. The denominator and last-split tail remain
the original operations, and merge SASS is identical. This confirms the intended
skip mechanism at this specialization, not universal numerical equivalence.

The compiler also changes the loop pipeline. The candidate retains a two-buffer
K allocation but loses the control's asynchronous V shared-buffer path. Its
loop carries larger full KV-address tensors/masks and synchronous V payload
state across control flow. TTGIR async-copy/commit sites decrease from6 to3;
PTX async-copy instructions decrease from18 to6 and commits6 to3. Static wait
instruction count is unchanged, but the staged wait depth changes. These
different IR/ISA counts must not be confused with dynamic execution frequency.
The larger live state is consistent with the observed128 registers, although
no individual compiler variable is proven to account for a specific register.

Placing PV inside both source branches changes the compiler's buffering and
liveness despite preserving the source algebra. Consequently the original
question, whether saved rescale work exceeds predicate cost with unchanged
buffering/resources, was not admitted to execution. Timing this implementation
would mix a pipeline change with the intended skipped multiplies. The detailed
line/hash evidence is retained in`native/ARTIFACT_REVIEW.md`.

## Frozen sources and procedure

| Artifact | SHA256 |
|---|---|
| Working composition control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Rejected alpha-one candidate | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |
| Compile worker | `9e0377dc696518225b5091f65fe7f72ac5e96e6c2b539c399f13bf337a80a585` |
| Controller | `2ddfab210d7a589826674990a380d657a078b794ac73e175fb0fc5d27e9978bf` |
| Raw returned report | `952637501db9fa11b6a53f354824a9566e24186be5fab090531667f39fbf8fea` |

Source freeze is`99185609d7924d18bb07de950c3d689a9ec9e649`. Six source,
ten compile-worker and twelve controller offline tests passed, with independent
source/semantics/security review and real frozen-head preflight before launch.
Only the historical numerator expression changed; denominator, new-token tail,
lookahead guard, split/warp/stage geometry and precision remained source-identical.
PAGE_LOOKAHEAD-false identity is source/AST evidence only in this first probe.

One app compiled four units: control and candidate producers plus their merges,
all at full B32/L1 TP8, D64, M16/BN64,8192-token chunks,11 splits, four warps
and three stages. E4M3 Q/K encoding with`max_num_imprecise_acc=0`, BF16 PV
and FP32 softmax/partials were unchanged. Each produced TTIR, TTGIR, LLIR, PTX,
CUBIN and SASS. All files were hash/size-verified through a bounded gzip/base64
bundle before exclusive extraction. Already-installed cuobjdump was used with
a10s command limit; no NCU, installation, credential or network expansion.

Triton warmup and driver module-handle initialization compiled and loaded
functions without invoking the candidate or merge. Typed one-element pointer
placeholders supplied argument metadata; they were not executable fixtures and
are not correctness evidence. Native environment was H200 sm90, Torch2.14.1+
cu130, CUDA13.0 and Triton3.8.0 in existing image`im-cPTbGpNiX3RarKResPgX7h`.

## Closure, costs and decision

App`ap-uY5K1rBNFRR2K8rdo2F8tw` stopped at22:06:25UTC. Independent
22:06:59.192653UTC reconciliation confirmed stopped/tasks0 and all listed apps
stopped/tasks0; immediate slot release and rejection were reported before
polishing. The initial controller stopping/tasks1 receipt remains unchanged;
its exit1 indicated incomplete immediate closure, while the worker compiled
successfully with exit0. Those separate states are reconciled, not rewritten.

Worker time was19.3083s and client31.7535s. At the no-new-app cutoff
22:10:34.040391UTC, rounded workspace usage rose$0.03, detailed usage
$0.02871385, actual paid$0. Conservative closed allowance$0.1165823334 is
below the reviewed$0.40 free-credit reserve. Delayed workspace deltas are not
exact app invoices; initial unchanged metering was not called free execution.
Zero retries and90/100/115/20/140-second worker/child/function/startup/watchdog
bounds were retained. No later jobs are assigned to this cutoff.

The current working source remains the validated composition, with its official
1.233633059× score unchanged, A2-11 rollback and fusion control preserved.
No official quota was consumed and no formal entry was made. The predeclared
2%-per-fixture performance gate was never reached; it did not pass or fail on
timing data. No rescale-frequency claim follows from this run.

The only local follow-up considered is a separate lowering hypothesis based on
this actual compiler output: put only old-accumulator multiplication inside the
conditional and keep one common unconditional PV operation. It must first
restore the original K/V asynchronous pipeline and register bound in its own
compile falsifier. This rejected source is preserved unchanged; no threshold
sweep, silent repair, new candidate execution or new measurement is authorized
by this report. The follow-up card remains local until independently reviewed
and parent-coordinated.
