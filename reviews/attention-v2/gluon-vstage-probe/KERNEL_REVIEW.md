# Independent revised V-stage source review

**PASS for the local two-arm source/configuration contract.** No concrete
source-level arithmetic, masking, ownership or causal-control blocker was
found. Nine focused CPU tests passed independently without importing Triton
or the kernel. This is source readiness only: no native compile, execution,
correctness, performance or official admission follows.

Only this review was written. No old probe/source/artifact was edited; no GPU,
service, network, credential or Git action occurred. The earlier probe's failed
strict fully-serial native-control gate remains a failure.

## Distinct prospective comparison

Both arms issue current QK and call the same `_prefetch_k`, loading the same
live next-page mapping and masked **raw E4M3 K** before either dependency wait.
The helper neither casts to BF16 nor writes shared memory. Its returned K,
physical offsets and validity mask feed one common `_publish_kv` call.

That publication helper gathers raw V, widens/stores K, widens/stores V and
executes the existing shared-publication fence. `v_after_wait` waits before
this helper; `v_before_wait` waits afterward. Specializing the selector and
removing that one wait yields identical loop ASTs. Load volume, masks, full
tile stores, arithmetic, scales and buffer count match between arms. The
configuration rejects the old serial/overlap labels.

No manufactured dependency, volatile access, inline assembly, added barrier,
precision change or knob sweep forces a slower control. Compiler hoisting is
allowed. A later native gate must establish actual remaining V/conversion/
publication placement around the **corresponding** QK drain; source ordering
alone is insufficient. If the relevant native schedules become identical,
there is no causal pair to benchmark. This reframes a new hypothesis without
retroactively passing the old declared experiment.

## Arithmetic and live-input boundary

The fixed pair remains B32/HKV2/HQ16, L8/G8, M64/N64/D64/page1, stride86008,
grid(32,2), four warps/one stage, SM_SCALE0.125 and QK_FACTOR5.4730064863875.
The same14-argument standalone API consumes unscaled normalized/position-
processed BF16 Q, prepopulated E4M3 K/V including eight new tokens, live GPU
lengths/page tables/scales, and separate raw BF16 output.
The final card specifies PageTable int32 `[B,PT_STRIDE]` and HistoryLengths
int32 `[B]`, matching the official generator; the kernel widens indices after
loading. This clarification does not change the reviewed source.

Q rounds BF16→E4M3→BF16 before scaling. K/V widen inside the kernel; QK/PV
explicitly use zero imprecise accumulation. The FP32 softmax/denominator and
zero-C synchronous BF16 PV followed by `acc*alpha+PV` are unchanged between
arms; the complete loop math suffix is AST-identical to the preserved prior
probe. This is not bitwise equivalence to a different carry-C lowering or
production FA3. No normalization, cache append, output norm/gate or production
split/merge is supplied by this isolated core.

Rows map `m=j*8+g` to token/head exactly as before. The loop covers history
plus eight new keys, with `key<=history+j` causal visibility. Boundary models
cover zero history, partial64-token tiles and8191/8192. Every masked load has
zero fallback and every K/V shared store overwrites a complete64×64 tile;
the unused final speculative tile is fully zeroed.

## Ownership and remaining gates

The two BF16 K slots and two BF16 V slots remain distinct. Current QK uses
current K; publication writes only the opposite K/V slot. The previous PV was
synchronous before that slot became reusable. The current dependent score
wait precedes score use, and synchronous current PV precedes slot flip/reuse.
The CPU symbolic lifetime model covers one/multiple/partial tiles. Raw next-K
and mapping now remain live across the wait in both arms; actual register cost,
shared publication and native lifetime behavior still require compilation.

The final experiment card accurately limits this to a local component with
zero direct official-score effect and no measured recoverable-latency fraction.
It preserves the old failure and forbids a forced slow control, identical-case
benchmark or posthoc rescue. Native ISA/resource/ownership evidence and a
separately reviewed bounded harness are prerequisites to any onward numerical
phase. Original-tolerance replay/correctness and prospective uninstrumented
whole-call comparisons against the working FA3 path remain separate gates.
No B1/decode/sliding applicability, novelty, achieved occupancy or score benefit
is established by this source review.

## Reviewed SHA256

| Artifact | SHA256 |
|---|---|
| `kernel.py` | `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e` |
| `config.py` | `686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0` |
| `test_source.py` | `156508501f5d48a4f0845caccffa5fee2189b2080b902f2f0689e7382f66824b` |
| `EXPERIMENT_CARD.md` | `209082108de413aeefb114c963096519451f60908934e12ea9aac15c30fefc1d` |

The test also verifies the old kernel remains SHA256
`3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c`.
