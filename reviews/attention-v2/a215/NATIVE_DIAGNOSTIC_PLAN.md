# A2-15 native diagnostic — exact compile-only attempt prepared

Historical attempt at 2026-10-25 08:31:05 UTC: the supported read-only `modal app list
--json` failed with exit 1, `Token has expired`. No upload, app creation or
GPU execution occurred. See NATIVE_ATTEMPT.md and native-attempt.json.
A separately approved credential renewal subsequently completed; the current
verified credential expires 2026-10-26 03:39:46 UTC. It is reused without any
further credential action. The current attempt has not yet launched.

The latest exact approval narrows the original proposal to A2-13, A2-15 and
one reviewed diagnostic program only. It permits one bounded H200 diagnostic,
at most $0.25 incremental free credits and $0 paid spending, using existing
authentication only. A2-14 and separate official fixture/reference files are
excluded. No OKC check or GitHub push is authorized by this Modal diagnostic
scope; the later official capability audit/private delivery is recorded
separately in OFFICIAL_CAPABILITY_AUDIT.md. The full diagnostic allowance is
unused. The stricter cumulative stage cap leaves $0.212180198 from
$2 minus historical conservative accounting of $1.787819802. Current Mac UI
shows $28.80 credits, $0 net charges, $0 spend cap and no live apps. No
settings changed. The proposed conservative reserve is $0.196609280.

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
public current documentation. native_diagnostic.py now implements the narrow
compile gate; local checks and independent signature/transport review precede
launch. Runtime compiler compatibility remains unverified until execution.

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
5. This particular bounded attempt performs compilation and disassembly only.
   Tiny allocator-aligned CUDA tensors provide pointer types/alignment; kernels
   must NEVER launch with this undersized storage. Logical full B32/L1 scored
   agentic metadata is stride_tok2304, stride_pt86001, SPLITS11, BS16, chunk8192.
   Real input contents, correctness, cache writes, ABA and timing are not tested.
   Any later timing requires full correctness with proper fresh allocations.
6. Preserve exact generated TTIR/TTGIR/LLVM/PTX/cubin and available SASS, plus
   hashes and metadata. Small <=16KiB yielded chunks avoid a single giant
   result; all bytes and final completion marker are verified locally. Missing
   or partial artifacts remain incomplete evidence, never a native pass.

Use one ephemeral container, no detached deployment or schedule, no retries,
and an explicit startup/runtime deadline conservatively fitting the approved
$0.212180198 stricter remaining free-credit ceiling, including overhead,
with paid cap $0. Worker hard limit60s, function75s/startup20s, app watchdog110s
plus15s stop allowance; fixed H200/2CPU/8GiB, max1, no idle capacity. Modal
1.6.2 generators forbid an explicit retry policy (even0), so retries=None
means no application retry policy; only one invocation is made.
No token validity or current credit balance is assumed. Stop and verify termination
even on compile/test failure. If the approved runtime cannot expose compiled
metadata or fit the required cases within its bounds, return that exact limit
without extending resources. No OKC check is reserved by this proposal.

If only the normal OKC service is available, it can validate correctness and
whole-call timing after a separate exact-payload approval, but its current
receipt schema has not exposed the required compiler resources. State that
limitation; do not call a timing result proof of overlap or spilling.
