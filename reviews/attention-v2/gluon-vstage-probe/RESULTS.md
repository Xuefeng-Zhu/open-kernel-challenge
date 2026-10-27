# V-stage runtime: scheduling benefit, production replacement rejected

The original correctness/replay/cache gates passed. Moving conversion/shared
publication before QK drain improved the two matched prototype calls by
**5.0355% and3.8883%** on the two fresh fixtures, with all six reverse-order
pairs favorable. However, the faster prototype remained about **3.82× the
actual selected FA3 latency**. The predeclared actual-baseline gate failed.
This is a completed, valid experiment with a positive local scheduling result
and a **production NO-GO**, not an execution failure.

No official check, production dispatch change, promotion or ranked entry
followed. Working fusion+lookahead source
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
remains selected with its previous official1.233633059 score; A2-11 rollback
remains unchanged. Do not turn a win over this prototype control into a score
claim or another incremental scheduling branch on a much slower engine.

## Source and execution binding

| Evidence | Exact binding |
|---|---|
| Runtime freeze | `c55d14c507b9d5cc2254441531d4f0e1a18714d5` |
| Core SHA256 | `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e` |
| Worker SHA256 | `bc93f136f4943e28ce3fd6be91cf67639ee4172ea9195b715bc762ac80a13d83` |
| Controller SHA256 | `1c9e664635fc31637e5f5629fc6bf37a03f6121eef69952ef4a8ed6cf401d062` |
| Prior compile raw SHA256 | `258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240` |
| Runtime raw SHA256 | `38946bec2ddf7c749f9890fb7bae677cb4bedf11c03a07e55f0b7391b9b74da5` |
| App | `ap-Hy5O3TPhVTVKu0XyfsgJWT` |
| Image | `im-cPTbGpNiX3RarKResPgX7h` |

The existing source, installed API, precision, resource and buffer-lifetime
reviews passed before execution. Both metadata pointers match the official
int32 ABI. Native instructions already established that **both arms prefetch
raw K and V**; the tested difference is remaining conversion/shared-publication
placement, not V-load serialization. The earlier fully-serial-control failure
is preserved separately and was not retroactively passed.

## Complete-call boundary and input limits

Both prototypes capture and time the exact working fused Q/new-KV preparation,
FP8-Q-to-BF16 cast, unchanged raw Gluon attention core, and the same output
RMSNorm/sigmoid gate. The third arm calls exact working `kernel_fn`, including
FA3's unchanged automatic scheduler/splits. All preparation, new-cache writes,
cast, attention and output work remains inside each call. The FA3 arm receives
no artificial cast, changed scheduler or slowed control.

All cases are TP8 B32/L8, D64, page1, full attention with no RoPE. Two short
cases use histories `[0,1,55,56,63,64,65,129]*4` (reversed for B), at unit and
1/64 KV scales. They cover causal new-token visibility, empty history, tile
tails and live cache slot0. The third case uses nominal ctx4096, agentic
lengths/coherent values and two fresh seeds135751/135752.

The int32 page table is padded to stride86008 outside capture, preserving every
valid entry and setting unused suffixes to−1. This binds the compiled geometry
and is disclosed in the prospective card. These are **nearby ctx4096 calls
with a padded table**, not scored80K timings or a standard unpadded4096 result.
No twelve-shape score,80K competitiveness or production serving performance
can be inferred. The actual FA3 comparison applies to these same inputs only.

## Correctness before timing

All prospective gates completed before any timing:

- **30 unchanged original-oracle calls**:27 three-arm/case A1/B/A2 checks,
  one independent graph's own-A setup and two separate-stream submission checks.
- **9 exact returning-A checks**,36 whole-K/V cross-arm byte comparisons,
  18 cross-arm new-KV comparisons and9 complete prototype-output/new-KV
  byte comparisons.
- Stable captured addresses, immutable inputs and untouched historical-cache
  rows passed. Live QKV, lengths, page mappings, positions/cache lengths and
  cache locations changed at the same addresses, then returned to A.
- Four post-submission isolation checks and two extra actual replays passed.
  Distinct graph pools and disjoint writable outputs/caches were observed;
  actual hardware overlap and internal scratch addresses were not measured.
- All48 warmup/measured replays retained exact own oracle-checked bytes,
  including16 prototype-pair and32 cross-arm new-KV timing comparisons.

Prototype outputs must match each other byte-for-byte. FA3 output is checked
by the unchanged original oracle, not forced to equal different attention math.
All three arms' new K/V and entire cache bytes match. These results cover the
declared fixtures, not general serving-concurrency safety.

## Paired timing result

CUDA graphs used cold-L2 eviction outside the event interval. Four total
warmup rounds and twelve measured rounds gave six reverse-order pairs, three
per fresh fixture. No profiler or replay-cost subtraction was used. Table
medians are pooled raw event medians; acceptance uses paired per-fixture ratios.

| Complete call | Pooled raw median, µs |
|---|---:|
| Actual selected FA3 |40.0639996|
| `v_after_wait` |161.1040011|
| `v_before_wait` |153.4240022|

| Paired comparison | Seed135751 | Seed135752 | Favorable pairs |
|---|---:|---:|---:|
| Before vs after: latency reduction |5.0354684%|3.8882857%|6/6|
| Before vs actual FA3: latency ratio |3.8228765×|3.8165386×|0/6|

The prospective gates required at least2% reduction on **each** fixture and
all six pairs favorable, separately against the prototype and actual FA3.
The mechanism gate passes; actual FA3, replacement and follow-up gates fail.
One session with six pairs is not an aggregate confidence interval. The
scheduling effect includes the compiler's resulting register/instruction
allocation; no direct trace isolates active compute/IO overlap or hardware
bottleneck. Resource parity alone cannot explain the much larger FA3 gap.

The independent [result review](runtime/RESULT_REVIEW.md) and
`runtime/analysis.json` reconstruct counts, byte checks and paired arithmetic
from the unchanged raw result. The controller's initial exit1 reflected its
stopping/tasks1 observation after successful worker exit0; separate closure
resolved that transition. It is not a numerical or benchmark failure.

## Closure and cost

The app stopped after the completed screen. Independent reconciliation at
**2026-10-26 00:40:48.399511 UTC** observed stopped/tasks0 and all listed main
apps stopped/tasks0. The GPU slot was immediately released before reporting;
Projection P057 owns the next slot under parent coordination.

Worker-reported wall38.566078s, supervised child40.088719s, client49.763929s.
At00:43:26.103049UTC, no new other app IDs appeared and delayed detailed
workspace credit usage increased **$0.05258797**, paid **$0**. This is not an
exact per-app invoice. The fixed free-credit reserve was$0.58, with
160/170/185/20/210s worker/child/function/startup/watchdog bounds and zero retries.
No billing, credentials, installations or visibility settings changed.

## Next architecture question

Stop this complete-call replacement route and preserve the verified local
mechanism for reference. Adding more wait-placement tweaks to a3.82× slower
prototype does not address the actual baseline gap. No official check is
justified by this result.

Local research instead follows the source-bound B1/sliding budget: the actual
B1 decode producer has32 CTAs, followed by16 merge CTAs, with historical
profile activities7.8µs and1.7µs. GQA reuse, FP8 QK, exact window pruning,
preparation fusion and parallel merge already exist. A dedicated short-window
mapping must justify its active-head layout, duplicated KV traffic, partial
storage and critical path before a single actual-best-baseline falsifier.
Those profile durations do not prove a launch or occupancy bottleneck and
cannot be subtracted from calibrated whole-call time. This remains one piece
of a broader B1/sliding/full coverage plan, not a forecast of catching the leader.
