# Independent compact-window source review

**LOCAL SOURCE PASS; native eligibility, numerical correctness, replay and
performance remain untested.** No concrete source-level blocker was found for
one bounded original-oracle validation. This is not a promotion recommendation.
The selected production source remains unchanged.

Reviewed candidate SHA256:
`c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594`.
Baseline SHA256:
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Prospective card SHA256:
`3dfe8928c3449ff4717f8d88ee21d07a6a96467c4c504f62dfab2a29ef72fc28`.
Source manifest SHA256:
`34e5e6fd9f293e429d17855edcbd9f4feaa2b665b66ccfb913efa6329073c759`.
The earlier candidate `5fc39af...` was superseded only by the root-requested
bounded import-time metadata diagnostic. This review binds the final c430
source, not the superseded hash.

The reviewer independently read the Attention TASK/AGENTS, complete new
producer/adapter/guard, preserved baseline functions, original generator,
saved installed public wrapper, prospective experiment card and CPU tests.
Seventeen CPU tests independently passed: six source/dispatch/metadata tests
and eleven geometry/byte/address/ABA model tests. They do not import or execute
the candidate, Torch, Triton or FA3.

## Source and semantic checks

The whole-module AST reconstructs the exact baseline after removing the new
helpers/entrypoint/metadata call and restoring the old entrypoint's name.
Original normalization, RoPE, Q/K/V rounding, scale placement, original-cache
append, FA3 helpers, output norm/gate, imports and license are retained.
The new guard depends only on structural configuration: L8, sliding2048,
RoPE500000, page1, HQ16/HKV2/D64, with no batch whitelist. Page64, decode, full,
DP heads and other configurations reach the unchanged original dispatch.

Let H be the historical length, L=8, W=2048, T=H+L, C=min(T,W-1+L), and
S=T-C. The copy retains absolute positions S..T-1. In compact coordinates,
FA3's bottom-right causal endpoint for query j is C-L+j. Keeping left
window W-1 admits compact indices
`max(0,C-L+j-(W-1))..C-L+j`. Translating by S yields exactly
`max(0,H+j-(W-1))..H+j`. Thus every required key is retained, future new keys
remain masked, and query j>0 drops the obsolete prefix. Disabling the window
after compaction would be incorrect; the candidate retains `(2047,0)`.
Global-position RoPE is computed by unchanged preparation before copying.

The byte-copy kernel consumes live GPU `cache_seqlens`, the same total-length
field used by baseline FA3, rather than a host-derived or cached length.
Its page1 lookup is `page_table[b,start+row]`, masked to retained valid rows.
Physical page0 is allowed; negative/suffix page IDs cannot become active
source accesses. Source addresses use int64 page offsets and the contiguous
128-byte TP8 K/V-row layout provided by the original generator. Destination
addresses partition `[B,2055,2,64]` exactly: every capacity row is written each
call, including zero padding; the last 32-token tile has seven capacity rows.
Only tile0 per request writes its compact int32 length, so there is one writer.
Separate K/V buffers preserve encoded UINT8 bytes without requantization.

Original cache writes remain solely in the unchanged preparation kernel.
The subsequent gather only reads those caches; separate stream-ordered kernel
launches provide the preparation-to-gather and gather-to-FA3 dependencies.
No extra stream, completion counter or cross-CTA producer/consumer assumption
is introduced. No cross-call scratch, input-derived map or output is cached.
Scratch and length allocations belong to the invocation/captured graph, and
the gather rewrites them from current lengths/page tables on replay. CPU ABA
models establish the intended address/content model, not native replay safety
or observed hardware concurrency.

The new FA3 call differs from the original sliding adapter in exactly four
keyword expressions: K scratch, V scratch, `page_table=None`, and compact
cache lengths. Q, cumulative Q lengths, max query length, descales, softmax
factor, causal/window arguments, scheduler metadata and gate remain the same
source expressions. Under the contract sliding metadata is None, so no old
schedule is incorrectly reused. Requested automatic split/PackGQA policy is
unchanged; its resolved choices may change with representation and capacity.

## API and diagnostics boundaries

The saved wrapper in the unsplit runtime receipt has full-file SHA256
`610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`.
Its public interface supports absent page tables/contiguous cache storage,
int32 per-request cache lengths, cumulative Q lengths and default per-request
cache batch indices 0..B-1. The scratch has exactly B cache batches, contiguous
last dimensions, and the original varlen Q request boundaries. This is API
surface evidence only. The saved wrapper is not proof of the current judge's
binary or that its SM90 FP8 contiguous/varlen/local specialization is available,
capture-safe or fast. Unsupported native dispatch must be an honest rejection.
Neither TMA, an opcode dtype, a resolved split count nor PackGQA is established.

The import-time metadata helper reads only scalar Torch/CUDA version strings
and the registered vendor operator schema, without invoking the operator or
accessing inputs/tensors, files, harness state or CUDA functions. Version
strings are bounded to128 UTF8 bytes, schema to4096 bytes; oversized schema is
replaced by a constant rather than presented as a complete truncated schema.
Failures print a constant without exception contents. The fake-operator CPU
test verifies no native fwd invocation and the bounds/failure behavior.
The TASK explicitly allows diagnostic print calls; no explicit saved-rule
blocker applies to these scalar diagnostics. Actual server static admission
remains untested. A rejection does not authorize checker weakening or evasion.
An observed operator schema may still expose only inputs/unnamed returns; it
must not be misreported as resolved packing, split counts or internal P dtype.

## Evidence limits and decision gates

The prospective card correctly treats additional scratch, full-capacity stores,
2080 packing CTAs at B32 and an extra launch as costs. Three logical traversals
are not measured HBM traffic or a strict latency bound. Existing baseline
GQA, fusion, window pruning and async mechanisms are acknowledged. Potential
specialization/scheduler changes are coupled representation consequences,
not proof that the baseline lacked a feature.

One original official check can falsify native eligibility and numerical
correctness when the normal Modal path is unavailable. It compares the new
candidate against its own SGLang reference, not a same-session selected3da arm.
Historical normalized comparisons remain descriptive; current evaluator/code/
image compatibility and unchanged-path drift must be checked before attributing
gains. Passing source tests, compiling, losing a combine or seeing a better
point score cannot promote this candidate automatically. The selected1.233633
candidate and rollback remain unchanged until the declared later assessment.

Forced native slot0, same-address metadata mutation, cache-byte preservation,
independent graph pools and serving-concurrency behavior are not established
by these CPU models. Preserve those coverage limits even if the official
16+12 suite passes; random official fixtures do not guarantee a forced slot0
case. Any original-oracle failure, unsupported native path or failed prospective
performance gate must stop this source without tolerance/threshold rescue.

Reviewed CPU-test SHA256 values:

| File | SHA256 |
|---|---|
| `test_source.py` | `c287be8ba250ba6c0aa57f38db4c138f6fb4cc52771c4c75350456993a2ccc04` |
| `test_geometry.py` | `1b3a681e42568172dae87865f693801df05108c70526d5d9d6fb3d7fc278edb8` |

The reviewer performed no authentication, network/service request, upload,
GPU/official execution, Git operation or candidate edit. Only this review
document was written.
