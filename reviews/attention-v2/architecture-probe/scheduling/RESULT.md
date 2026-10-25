# Dynamic persistent scheduling: fixed chunks show no recoverable model imbalance

Local analysis of recorded H200 inputs, 2026-10-25. No GPU job, official check,
browser operation, CUDA prototype, credential action or candidate edit was made.
The independent worktree starts at `f67c58987cab6416080ef0dcb8f345f9700a95c7`.
Selected A2-11 and the A2-13 source remain unchanged.

**Decision:** do not implement a persistent scheduler that merely reorders the
existing 8,192-token chunks. For the two actual recorded B32 inputs, an
idealized FIFO model already reaches the lower bound imposed by the number of
full chunks. A persistent queue with the same chunks has zero recoverable
imbalance in this model, before assignment overhead. This is not proof about
real hardware scheduling, CTA service times, or an alternative partitioner.

## Actual inputs, rather than inferred CUDA RNG

The primary vectors are the recorded `actual_lengths` from the previous H200
A2-13/A2-15 paired run, `a215/paired-result.json`, scored-family full B32/L1,
D64, page1, scale1, 80K agentic/coherent. Seeds were 135701 and 135702. Both full
32-element vectors and all model results are preserved in `results.json`.
They are historical real GPU inputs, not a new measurement or assumed official
timing repetitions. The four supplementary CPU examples invoke the exact saved
official `_lengths` function with benchmark seeds 604/606/610/612. CPU and CUDA
RNG streams can differ; these examples are expressly not substituted for the
recorded H200 vectors.

| Actual H200 seed | Range | Mean length | Full 128-tile CTAs | Partial-history CTAs | Empty-history CTAs |
|---|---:|---:|---:|---:|---:|
| 135701 | 74,201–85,893 | 79,712.0625 | 594 | 64 | 46 |
| 135702 | 74,083–85,967 | 80,255.40625 | 604 | 64 | 36 |

The shape-only page-table capacity is 86,001 tokens, so the current grid has
`32 × 2 × ceil(86001/8192) = 704` CTAs. The two vectors perform 79,740 and
80,284 historical 64-token tiles across both KV heads. Their small ±7.5%
length spread does not resemble a highly ragged workload with arbitrary zeros.

Every empty-history CTA is split10, **the last split**, and must still perform
Q normalization/E4M3 conversion, attend to and write the live new token, and
write its partials. There are no fully removable empty CTAs in these fixtures.
The existing loop already skips their historical KV loads. Dropping these
CTAs from a taskmap based only on historical length would break semantics.

## What the scheduling calculation does and does not establish

The model uses x-fastest CUDA grid order (batch, KV head, split), identical
independent service slots, and one unit per historical 64-token tile. It is not
an observed CTA trace. For an assumed one resident CTA per SM, 132 slots:

| Seed | FIFO makespan, tile units | Full-task indivisibility lower bound | Divisible total-work lower bound | Optimistic divisible-work gap |
|---|---:|---:|---:|---:|
| 135701 | 640 | `ceil(594/132) × 128 = 640` | 604.576 | 5.535% |
| 135702 | 640 | `ceil(604/132) × 128 = 640` | 608.697 | 4.891% |

A last-split new-token operation is charged one illustrative unit in these
numbers. Actual prologue, assignment/descriptor construction, page-table
gather, memory contention, softmax/partial traffic, merge and synchronization
costs are not measured by this model. The divisible-work gap is a mathematical
ceiling in these assumptions, not a latency forecast or expected gain.

To expose the uncertainty, `analyze.py` also considers 1–5 identical resident
CTA slots per SM and prologues of 0/1/4/16 tile equivalents, without fitting any
parameters. All 40 cases (two vectors × five slot assumptions × four prologues)
reach the full-chunk pigeonhole bound under both FIFO and longest-job-first
scheduling: reordering the same chunks has zero modeled benefit. At zero
prologue, arbitrary repartition's total-work gap varies from approximately
4.9–5.5% for 1 or 5 slots/SM, 20.7–21.3% for 2 or 3, and 40.6–41.0% for 4.
That extreme sensitivity is a reason to obtain an actual residency/service-time
trace before interpreting a modeled repartition benefit. Resident CTAs share
the SM; real per-CTA time is not constant as assumed here.

The parent experiment targets full verify, which currently routes to FA3. The
above input vectors and resource shape belong to the custom B32 **decode**
path. Reusing their vectors for L8 is only a hypothetical distribution scenario;
this report makes no L8 scheduling or performance claim from them.

## Fine tasks must aggregate before emitting partials

For current M16/D64, each producer CTA writes
`16 × (64 + 2) × 4 = 4,224` bytes of FP32 output/max/sum partials. Across 704
CTAs that is 2,973,696 bytes (2.836 MiB), before reading them for the fixed
capacity merge. The empty-history owners still emit these partials.

Naively emitting one equal-sized partial per historical 64-token tile would
produce 79,740/80,284 records: 321.218/323.410 MiB, roughly 113.3×/114.0× this
fixed partial output, excluding new-token work and descriptors. A full 8,192
chunk has 128 tiles, the maximum expansion per full chunk. A viable fine-grained
planner would have to aggregate contiguous buckets into bounded partials,
preserve each request/head's online-softmax state, assign unique new-token
ownership and account for gather/merge overhead. None of those costs is free
or validated here. No such CUDA implementation is justified by the present
fixed-chunk evidence alone.

## Existing FA3 already does length-derived preparation

See `fa3-audit.md` and its hashed evidence. Actual A2-13 official full-verify
B16/B32 profiles include `prepare_varlen_num_blocks_kernel<1,true>` at 3.1 us
and `<2,true>` at 3.6 us. Mainloop/combine activity times are respectively
189.8/3.7 us and 373.9/3.7 us. Activity durations cannot be directly subtracted
from complete-call latency. Mainloop template names are truncated, so they do
not establish the installed persistent scheduler type.

The Mac environment lacks FA3 implementation/source; historical package metadata
does not identify the H200 FA3 binary. Exact installed wrapper/shared-object
hashes and full specialization names remain missing. Existing upstream FA3 has
persistent scheduling mechanisms, so novelty cannot be claimed from a proposed
dynamic queue. The task's HPC-Ops benchmark evidence warns that assignment is
excluded by default unless `--include-taskmap`; those H20/D128/page64 results
are not a measured Muse/H200/D64/page1 gain.

Any future comparison must consume live GPU lengths and page tables, build
assignment/descriptors inside **every timed call**, and test same-address
A→B→A graph replay with changing contents. Cross-layer or cross-call taskmap
reuse would invalidate this proposed experiment even though the baseline's
provided decode metadata is part of the documented input contract.

## Reproducibility, limits and next evidence

Run `analyze.py` using the original workspace `.venv/bin/python`. It asserts
frozen source hashes, preserves the two actual vectors, verifies non-removable
empty owners and exact fixed-task bounds, and emits supplementary official
CPU length-function examples. It passed locally. This validates arithmetic and
source provenance, not GPU correctness or timing of an unbuilt scheduler.

Frozen A2-13 SHA256:
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Recorded paired result SHA256:
`92b5ba197689c3e17ec3c86cefdefdb1edd4bfe0222171b0bd63bd67f000e867`.
Official reference SHA256:
`386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b`.
All new local deliverable hashes are in `manifest.json`.

The next useful scheduler evidence is an authorized bounded trace of actual
CTA residency/service-time distribution plus the current FA3 wrapper/shared
object identities and full scheduler specializations, collected alongside the
already-prioritized compile probe. Only if that trace shows worthwhile loss
should a live per-call planner prototype be proposed. Its acceptance criterion
must be complete-call correctness/replay and paired latency including assign,
descriptor, gather and merge; a mainloop-only benchmark is insufficient.

The initial manual-budget prerequisite was superseded by the user's direct
reply and then resolved through a separately authorized read-only Modal billing
tab: current free balance$28.74, paid spend cap$0 at17:41:26UTC. The independent
compile-only probe was accepted as app `ap-9WHeLVDCItf6dndN7LpuIn`; no scheduler
CUDA prototype was built or launched. Scheduling analysis itself incurs no GPU
cost. See the parent result for the final compile/shutdown/cost outcome.
