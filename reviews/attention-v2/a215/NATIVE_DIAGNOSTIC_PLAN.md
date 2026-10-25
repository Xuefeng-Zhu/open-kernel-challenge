# A2-15 native diagnostic — approved scope, blocked before upload

Status at 2026-10-25 08:31:05 UTC: the supported read-only `modal app list
--json` failed with exit 1, `Token has expired`. No upload, app creation or
GPU execution occurred. Do not renew credentials or retry through another
authentication route. See NATIVE_ATTEMPT.md and native-attempt.json.

The latest exact approval narrows the original proposal to A2-13, A2-15 and
one reviewed diagnostic program only. It permits one bounded H200 diagnostic,
at most $0.25 incremental free credits and $0 paid spending, using existing
authentication only. A2-14 and separate official fixture/reference files are
excluded. No OKC check or GitHub push is authorized. The full allowance is
unused. Historical conservative cumulative Modal accounting remains
$1.787819802; current credits, paid cap and live apps could not be reverified.

Purpose: determine whether the exact A2-15 page-lookahead source creates a
different useful dependency schedule versus A2-13. The missing A2-14 resource
comparison remains deferred and outside this approval.

Frozen source set:

| Version | File under reviews/attention-v2 | SHA256 |
|---|---|---|
| A2-13 | mixed_fa3_fp8_qk.py | e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e |
| A2-15 | mixed_fa3_page_lookahead.py | fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c |

Use the approved bounded H200 environment with the pinned dependency image
only after existing valid authentication and the credit/cap prerequisites can
be verified. Credential creation or renewal is not authorized. The worker's Torch/Triton,
driver, GPU and image versions must be captured, rather than inferred from
public current documentation. Native instrumentation has not been implemented
or tested here; its exact API must be matched to that runtime before launch.

1. Allowlist exactly the two sources above and one reviewed diagnostic
   program. Do not reuse the old Modal wrappers: they package a larger file
   set. The older replay probe also lacks QK_DOT, SEPARATE_MERGE and
   PAGE_LOOKAHEAD compile arguments. A narrow program must match the current
   signatures, include only necessary reviewed synthetic input/oracle logic
   with attribution, and preserve input snapshots because kernels write cache.
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
   intended overlap is not demonstrated. Resource differences do not by
   themselves prove a hardware cause.
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
and an explicit startup/runtime deadline conservatively fitting the approved
$0.25 incremental free-credit ceiling, including overhead, with paid cap $0.
No token validity or current credit balance is assumed. Stop and verify termination
even on compile/test failure. If the approved runtime cannot expose compiled
metadata or fit the required cases within its bounds, return that exact limit
without extending resources. No OKC check is reserved by this proposal.

If only the normal OKC service is available, it can validate correctness and
whole-call timing after a separate exact-payload approval, but its current
receipt schema has not exposed the required compiler resources. State that
limitation; do not call a timing result proof of overlap or spilling.
