# Proposed native diagnostic — not authorized or executed

Purpose: determine whether the exact A2-15 page-lookahead source creates a
different useful dependency schedule, and obtain the missing A2-13/A2-14
resource comparison. This is one bounded diagnostic, not another tuning sweep.

Frozen source set:

| Version | File under reviews/attention-v2 | SHA256 |
|---|---|---|
| A2-13 | mixed_fa3_fp8_qk.py | e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e |
| A2-14 | mixed_fa3_deferred_denom.py | bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d |
| A2-15 | mixed_fa3_page_lookahead.py | fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c |

Use a separately approved H200 environment with the pinned official dependency
image and existing valid authentication. No account, token renewal, spending,
upload or launch is authorized by this document. The worker's Torch/Triton,
driver, GPU and image versions must be captured, rather than inferred from
public current documentation. Native instrumentation has not been implemented
or tested here; its exact API must be matched to that runtime before launch.

1. Allowlist only the three sources, a reviewed diagnostic harness, necessary
   official public input/reference files and notices. No repository, auth
   configuration, private logs or unrelated candidate transfer. Preserve input
   snapshots because these kernels write cache.
2. Specialize the same full B32/L1, D64/G8, page1/scale1 shape and official
   8,192-token chunks. Do not change precision, geometry or launch stages.
   Capture the compiled producer and merge objects through the runtime's
   supported compiled-kernel interfaces. Inspect native metadata and emitted
   IR/PTX/SASS where available; do not use restricted profiler counters.
3. Record registers, spills, shared memory, actual tensor instructions and
   compile configuration for each exact source. Distinguish unavailable fields
   from zero. Preserve bounded artifact hashes and a small result summary;
   avoid repeating the prior oversized-result BlobCreate transfer failure.
4. Compare the load/use dependency around the loop: does A2-15 issue the next
   page load before current QK/PV without waiting for its result until the end?
   Ignore mere register renaming or artifact hash differences. If the schedule
   is equivalent, or the earlier load immediately stalls current work, the
   intended overlap is not demonstrated. A2-14 resource changes may explain
   possibilities but do not retroactively prove a hardware cause by themselves.
5. Gate timing on correctness. Use the official oracle for full B32 with fresh
   cache, page1/page64, scales1/1⁄64, scattered/zero pages, empty/partial/8,192
   boundary lengths, plus fixed-address graph A→B→A with changing actual
   lengths/mappings. Check new rows and untouched cache rows. Then compare the
   12 scored paths if time/budget permits; record every skipped case explicitly.
6. Only if the native schedule and correctness support the mechanism, measure
   whole-call CUDA graphs against A2-13 with predefined warmup, cold-L2,
   alternating paired order and all raw samples. Compile/setup and profile
   activity times stay separate from steady-state whole-call timing. This
   diagnostic is not an official score. At least a 10% normalized target cut
   is the useful threshold, with no unexplained >5% control regression.

Use one ephemeral container, no detached deployment or schedule, no retries,
and an explicit startup/runtime deadline plus cost cap agreed before launch.
No dollar amount or token validity is assumed here. Stop and verify termination
even on compile/test failure. If the approved runtime cannot expose compiled
metadata or fit the required cases within its bounds, return that exact limit
without extending resources. No OKC check is reserved by this proposal.

If only the normal OKC service is available, it can validate correctness and
whole-call timing after a separate exact-payload approval, but its current
receipt schema has not exposed the required compiler resources. State that
limitation; do not call a timing result proof of overlap or spilling.
