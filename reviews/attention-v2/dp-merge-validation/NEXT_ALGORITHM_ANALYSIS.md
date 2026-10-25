# Next algorithm analysis: retire an unsupported grouped-request WGMMA idea

The immediate useful next mechanism is the parent's strict `alpha == 1`
softmax-rescale hypothesis, subject to its own source audit and frequency
evidence. The larger grouped-request decode mechanism considered below is
**deferred before implementation or GPU work**. Its arithmetic/softmax costs
are real, while the required effective-throughput gain is unmeasured. Another
compile-only demonstration of M64 WGMMA would not resolve that problem: the
earlier full-verify probe already demonstrated the instruction family and then
failed performance. No new candidate, service request, installation, GPU run,
counter retry or official check was made for this analysis.

## What the retained evidence establishes

The working composition source SHA256 is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Official job `5090b83f1d63407b` reports 1.2336330588627955x after 16 prerequisite
tests and 12 benchmark shapes. Its full-B32-decode target is 304.031893174us.
The direct fusion control remains the right attribution baseline; its separate
official target was 317.205125874us. Do not convert these different runs into a
paired algorithm experiment.

The official profile text identifies where time occurs, not why the hardware
spends it. Full B16 decode and full B16/B32 verify mainloops occupy respectively
97.7%, 94.6% and 96.9% of their instrumented kernel-time sums. The B32 custom
producer occupies 99.1% of its profile sum (297.2us producer, 2.7us merge).
Full B16 decode's instrumented producer is 297.3us even though its calibrated
benchmark whole-call time is 194.271909us; this discrepancy alone makes it
invalid to subtract profile activities from benchmark times or infer a bandwidth
shortfall from profile duration.

The separately saved A2-15 native producer specialization is M16/N64/D64,
4 warps, 3 stages, 11 static 8192-token chunks, 96 registers, 20,480 shared
bytes, driver spill field zero. TTGIR uses `nvidia_mma` version2 with
`warpsPerCTA=[1,4]` and instruction shape16x8; PTX/SASS show warp-level MMA,
not WGMMA. These are exact historical donor artifacts, not a newly disassembled
composition binary. Two earlier NCU sessions timed out; that does not establish
permission denial, a compute bottleneck, a memory bottleneck or stall fractions.

Two actual native scored-family input vectors contain 2,567,392 and 2,579,715
historical tokens. Lengths are near 80K. Without sorting or any planner, fixed
groups of eight have group-max padding ratios 1.062905859 and 1.060470633,
computed as `sum(8 * max(group_lengths)) / sum(lengths)`. The modest length
padding is favorable, but it does not cancel dense cross-request arithmetic.

## Short experiment card: grouped-request block-diagonal decode

**Hypothesis.** Eight requests each contribute eight GQA query rows to an M64
tile. A 64-column KV tile contains eight keys for each request. A strict
block-diagonal request mask removes every artificial cross-request logit before
softmax; each query still attends to every key of its own request. Already
E4M3-encoded Q/K would be widened exactly to BF16 for strict FP32-accumulating
WGMMA, with BF16 PV and unchanged FP32 softmax/partial semantics. This could
trade more tensor arithmetic for a better asynchronous implementation, but no
evidence presently shows that trade is favorable.

**Scope if ever revived.** Start with full B32/L1/TP8/D64/page1/no-RoPE only,
with other routes retaining the exact selected implementation. Full B16 is a
separate follow-on only after evidence, not a presumed transfer. The dense
block-diagonal mask is semantic request isolation, not permission to skip any
same-request key or value. New-KV ownership, fresh GPU lengths/page tables,
scales, output mapping and private partial lifetimes must remain explicit.

| Cost | Current selected producer | Grouped-request model |
|---|---|---|
| Rows/columns per dot | M16/N64, eight useful query rows | M64/N64, only one of eight request blocks is useful |
| Dense tensor arithmetic for the same valid work | Already has 2x row padding | 8x cross-request arithmetic; about 4x current dense work before length padding |
| With observed length padding | Baseline | Approximately 4.242-4.252x current tensor arithmetic, not an efficiency gain |
| Softmax/accumulator element work | M16x64 tensors each iteration | M64x64 tensors, about 4x dense element work before padding unless compiler elimination is demonstrated |
| Original 8192-token shape geometry | 32x2x11 =704 CTAs | 4x2x11 =88 CTAs; fewer than the source's 132 SM target, an immediate parallelism risk |
| Analytically motivated 2048-token chunk example | No change to current control | For capacity86001, 42 splits and 4x2x42 =336 CTAs; changes reduction cost and must have a geometry-matched control |
| FP32 partial bytes, including padded rows | 704x16x66x4 =2,973,696 | 336x64x66x4 =5,677,056, about1.91x output traffic |
| Accumulator storage per 128-thread CTA | 1024 floats, eight per thread | 4096 floats,32 per thread; additional live S/P/query state is not included |
| Shared storage example | Native donor20,480 bytes | Q64x64 BF16=8192 plus two K/V buffers=32768, at least40,960 bytes for that specific buffering design before alignment/barriers |
| Compiled registers/spills/residency | Donor96/registers, raw spill0 | Unknown; a resource estimate is not an occupancy measurement |

The 2048 example is an occupancy calculation, not an instruction to sweep chunks.
A valid experiment would compare both the unchanged original geometry and a
same-geometry raw control; otherwise improvements from subdivision could be
misattributed to WGMMA. No such source variants have been written.

**Break-even model.** Let `f` be the old time fraction attributable to its
actual dense MMA execution, `s` the softmax/FP32 scalar fraction, and let `r` be the grouped
implementation's effective tensor throughput divided by the old one. With
`k=4.25`, a deliberately simple lower model is

`Tnew/Told >= (1-f-s) + k*s + k*f/r + extra_overhead`.

It already optimistically holds memory/other time constant despite new resource
pressure, page indexing and larger reductions. Tensor throughput must exceed
4.25x merely to break even on the tensor work. To obtain a10% whole-producer
cut with `r=8`, the model requires `f >= (0.10 +3.25*s)/0.46875` before extra
overhead. At `s=0.05`, that is `f >=0.56`; at `s=0.10`, it is `f >=0.9067`.
Even `r=16` requires `f >=0.3574` when `s=0.05`. Neither `f`, `s` nor a credible
`r` is supplied by the retained activity traces. Peak tensor throughput does
not establish effective speedup on this dependency chain.

**Optimistic latency and weighted-score ceiling.** Current full-B32 latency
accounts for0.840262277 of its own decode32 load. Holding every other measured
shape fixed, `factor=(1-w*(1-new/304.031893174))^(-1/6)` gives:

| Assumed target result | Target latency | Modeled aggregate gain | Modeled score |
|---|---:|---:|---:|
| 10% target reduction | 273.628704us | +1.47354% | 1.25181118x |
| 20% target reduction | 243.225515us | +3.11393% | 1.27204754x |
| Reach the task's126.35us reference floor | 126.35us | +11.91534% | 1.38062457x |

The last row is an optimistic floor-derived score ceiling with frozen controls,
not achievable grouped-kernel latency or a forecast. It ignores additional
arithmetic, softmax, preparation, merge, index and occupancy costs. It cannot
justify implementing the candidate without a supported break-even case.

**Cheap falsifier and gates.** The present offline cost model is the cheap
falsifier: there is no measured decomposition or instruction-scheduling evidence
to support over4.25x effective tensor gain while controlling the roughly4x
softmax work. Therefore defer the idea now. Do not spend a GPU run merely to
rediscover WGMMA admission. Revival requires a concrete dependency/throughput
model from available evidence, a correct dense-mask numerical model, and a
resource plan that beats the same-geometry control. Only then would native
strict-precision correctness and a predeclared≥10% target paired benefit with
unchanged controls≤5% regression be meaningful. A lower measured gain could
be recorded but would reject this larger-architecture question, rather than
silently changing its goal.

## Why the strict alpha==1 mechanism is a better next causal question

The existing producer explicitly computes
`l_i = l_i * alpha + sum(p)` and
`acc = acc * alpha[:,None] + dot(BF16(p),V)` on every historical tile.
For M16/D64, the dense accumulator has1024 FP32 values; the valid eight-query
subset has512. When `alpha` is exactly one for **all valid query rows**, the
previous accumulator and denominator rescale are identity operations. A
uniform conditional path could remove repeated rescale instructions while
retaining the original else branch and all softmax/KV work. Its causal cost is
the all-valid-row predicate, branch, code/liveness effects and any lost pipeline
overlap. It is not free and not yet measured.

This is a narrower question than changing the main matrix algorithm. The next
evidence should first count exact all-valid-row `alpha==1` occurrences, by case,
chunk and replay, in a separate diagnostic variant. Padded rows must not make
the predicate falsely fail or pass; first-tile `-inf`/zero behavior, empty
historical chunks and the last split's new-token block need explicit semantics.
Do not use an epsilon threshold or approximate a nonunit alpha. The source
audit must address multiplication/FTZ and the actual compiler operation order,
not merely assume Python algebra proves native identity.

An instrumented frequency result is not timing evidence. Preserve a raw
uninstrumented control with the same split/warp/stage geometry and original
precision for paired whole-call timing, and keep diagnostics out of those
claims. A source/ISA audit should confirm that the conditional really removes
accumulator rescale instructions instead of evaluating both branches. Without
observed frequency and a cost bound, no target percentage is predicted here.
The parent owns that new audit and any subsequent experiment coordination.

## Local evidence and scope

Primary files read: the exact selected `fusion-lookahead-probe/candidate.py`,
its `official-check/analysis.json`, published official artifact JSON/text,
`runtime/result.json`, saved A2-15 producer TTGIR/PTX/SASS/resource summary,
the official attention `TASK.md`, and historical A2-03/E5/A2-13/next-plan
records. No third-party performance claim was substituted for Muse evidence.
The source's shape/config guards and the official12-row score definition were
used directly. The old M64 architecture-analysis file was read only to avoid
duplicating its established compiler-admission question.

This document changes no selection, source, protocol, oracle or resource state.
It recommends proceeding with the strict alpha question's evidence gates and
deferring the grouped-request idea, not a new official check of unchanged code.
