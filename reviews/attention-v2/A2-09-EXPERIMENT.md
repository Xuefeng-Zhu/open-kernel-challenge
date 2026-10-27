# A2-09: isolate fallback completion counters

Parent: A2-08 `mixed_fa3_fused_full_small.py`, source f01799cea791006f109f09bd064f953eff6b584b, official job cf979895a3f24685 (1.1557671624x).
Candidate: `mixed_fa3_local_tickets.py`, SHA256 3b5d5d711a16e309cf512fbde2ff1a06c85b3e6fd52a73712cf66a5ffa10ea12.

Problem: fallback invocations use device-global completion counters while their partials belong to individual calls. Interleaved calls with independent input/cache buffers can elect a merger before that invocation finishes publishing. This is an upstream concurrency defect; published sequential judge passes remain valid. It does not imply a defect in acquire/release atomics.

Single change: allocate the same zeroed counter capacity independently each call, inside the timed function. Remove the global dictionary. Preserve all GPU functions, math, precision, cache writes, splits, dispatch and other adapters exactly (AST comparison passes). Keep the original minimum 4096-counter allocation so this experiment isolates lifetime/ownership, not size tuning.

Affected scored paths: sliding decode B1/B16/B32 and full decode B32. Nearby E2 fallback paths also change ownership. Eight other scored FA3 paths remain identical.

Hypothesis: fresh per-call storage removes cross-call aliasing. The extra initialization is expected to cost time; this is a correctness repair, not a predicted speed win. Report any normalized path regression above 5% and the six weighted-load aggregate. Do not automatically promote based on an overlapping score interval. Keep original measured best for explicit sequential-only comparisons.

Local evidence: five ownership/protocol/AST tests, including 96 two-call completion interleavings, pass; 12 admission tests pass. Commands/environment in A2-09-cpu-preflight.json. No GPU simulation claim.

Native plan: exactly one official off-board check of this committed file. Official output/cache tests and ordinary graph replay establish only their documented coverage. Dedicated two-stream/two-independent-graph testing, explicit A/B/A on current paths, untouched-cache rows, actual allocator lifetime behavior and overlapping graph stress remain unverified. No Modal launch authorized in this phase.

Graph boundary: independent captures use private pools by default; explicitly shared pools must not replay concurrently. Replaying the same graph/output/cache concurrently is not made safe by this patch. See https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management . Counter memory-order semantics: https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html .
