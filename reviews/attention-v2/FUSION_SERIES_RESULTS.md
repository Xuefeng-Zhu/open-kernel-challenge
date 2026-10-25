# Bounded attention prologue-fusion results

Selected `mixed_fa3_fused_full_small.py`, source `f01799cea791006f109f09bd064f953eff6b584b`.
Official off-board job `cf979895a3f24685`: **1.15576716x**, CI [1.138430654999424, 1.1731036698725028].
Starting A2-04: 1.07706589x; selected point change +7.307%.
Separate jobs are not directly paired trials of our candidates. Each is paired against SGLang.

| Experiment | Job | Official score | Target normalized latency cut |
|---|---|---:|---|
| [A2-05](A2-05-RESULT.md) | 50465a4c440f4b75 | 1.11579874x | verify 1/sliding: 28.18% |
| [A2-06](A2-06-RESULT.md) | 05b52e7834064b14 | 1.13811173x | verify 16/sliding: 23.46%, verify 32/sliding: 19.49% |
| [A2-07](A2-07-RESULT.md) | ba9e02c1d6584cc4 | 1.15053324x | verify 1/full: 15.77% |
| [A2-08](A2-08-RESULT.md) | cf979895a3f24685 | 1.15576716x | decode 1/full: 12.54% |

## Method and exact limits

Each passing run contains16 prerequisite cases and12 timed shapes. Native sample counts
per shape are recorded in the accompanying JSON; they count timing sample entries,
not a proven total number of graph replays. Each run has11shapes with60samples
and fullverify32 with56samples:716entries per side. All service-returned entries
are retained without filtering by us. Raw/calibrated times and ratios stay separate.
Score uses six loads, each46*sliding+16*full, then geometric mean(SGLang/ours).
Profiles are separate instrumented runs with PDL overlap, not additive walltime.
No register/spill/shared-memory/counter measurements are exposed for these kernels.

A2-05 new path: B1/L8/sliding/page1/scale1. A2-06 adds B16/B32 sliding
plus three native prerequisites (mixed lengths, scale1/64, soft inputs), all page1.
A2-07 adds fullB1/L8, including the1,048,576-context/page64 prerequisite.
A2-08 targets fullB1/L1 with the supplied actual FA3 scheduler, page1/scale1.
For a failed experiment, its full intended coverage must not be inferred as passing.
Explicit same-address A-B-A CUDA graphs, untouched-old-cache stress and concurrent
invocations were not separately tested for these new fused sources. CPU models do
check these invariants but do not execute Triton/FA3. E2 fallback shared-ticket
concurrency limitations remain. Historical Modal replay tests use different sources.

Initial local A2-05 preparation incorrectly attributed explicit saturation to the
reference. The extra clamp was removed before upload; official direct-cast semantics
and finite CPU oracle checks were restored. Original preparation/history is retained.

## Budget and reproduction

Attention cycle: 8/10 accepted, 2 remain; old six-cycle is separate.
Platform quota at 2026-10-25T04:48:34.196775+00:00: {'scored': {'per_day': 10, 'used': 0, 'left': 10}, 'checks': {'per_day': 20, 'used': 10, 'left': 10}}.
Projection remains at five authorized checks; no projection request in this series.
No paid compute, ranked entry, credentials, or Modal launch. Conservative historical
Modal accounting remains$1.708917914/$2; previous shutdown evidence is retained.

```sh
.venv/bin/python reviews/attention-v2/summarize_fusion_series.py
```

Per-experiment cards list source hashes, exact commands and CPU/native evidence.
Original source commit IDs map to reviewed private-export commits in
`analysis/PUSH_EXPORT_MAP.json`. Model configuration is not exposed to tools.
