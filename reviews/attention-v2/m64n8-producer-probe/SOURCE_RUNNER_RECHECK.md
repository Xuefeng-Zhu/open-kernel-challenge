# Frozen-source recheck for one original-oracle official check

Reviewer: `/root/compact_review`. **No known source-level correctness or
contract blocker found.** This supports the separately authorized bounded
runner experiment, not numerical correctness, performance or promotion.
Only this report was written; no source edit, compiler, container, service,
GPU, authentication, network, official or Git action was performed here.

| Frozen input | SHA256 |
|---|---|
| Selected baseline | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Integrated `candidate.py` | `96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970` |
| `kernel.py` | `eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480` |
| `build_candidate.py` | `de1da53203a166d43d04c425ce81c4af855ca6d20453a406507b26d46bc5b93f` |
| `test_contract.py` | `e872828b1604a393062bebf4b4daea28b20a50f9cbab9ed2a45462740203a236` |

All16 independent pure-stdlib contract/model tests passed again. They cover
source roundtrip, guard/fallback, split/window boundaries, page-map mutation,
tail ownership, quantization boundaries, transposed online math and merge ABI.
These are mathematical/indexing models, not hardware precision or race tests.

The two raw FP8 K/V slots issue one committed group per real tile, prime at
most two groups, wait1 while another tile exists and wait0 on the final tile.
The completion barrier precedes raw reads, and the following CTA barrier
separates those reads into distinct BF16 scratch from refill of that raw slot
for tile+2. Branches containing waits/barriers depend on per-CTA scalar
sequence/split values, not lane predicates. QK completion is waited before
softmax, P publication has a barrier/shared fence, and PV requests synchronous
completion. The next iteration's CTA barrier precedes scratch reuse; final
wait0/barrier precedes tail use. No source-level unread-slot overwrite or
nonuniform-barrier defect was identified. Complete generated/runtime lifetime
correctness remains unproven.

History K×Q.T and V.T×P use the matching distributed layout. Key-axis reductions
and maxima/denominator/accumulator carry are FP32; P is BF16, both WGMMA calls
request strict accumulation. The carry and query-major N16 tail convert layouts
explicitly. Only the last-index split writes/attends new KV, including when its
history is empty. Q BF16→E4M3 rounding, direct normalized-FP32/ks→E4M3 K,
BF16-input/FP32/vs→E4M3 V and existing scale placement remain source-matched.
Eight valid partial rows retain the selected M16 ABI; neutral ignored rows8–15
are legal because the unchanged merge never consumes them.

Assembly recovers byte-exact selected source after removing the owned append
and restoring its single launch replacement. Allocation/grid/split policy,
merge and outer FA3 dispatch are unchanged. Only the four declared L1/G8/page1
families with SPLITS>1 select this producer; S1 retains the original path.
The unannotated `page` parameter is not a known blocker: saved selected page1
code specializes it, and pinned Triton3.8 GluonJITFunction inherits the ordinary
JIT binder (`_runtime.py` SHA739a9c208c3c73ff5d4ec2d38a4051c20140e4f149c442a05cc37010443d5684).
No specialization override was added. Actual judge compiler/launch identity
still requires runner evidence.

The historical absolute zero-stack/local gate **remains failed** in RESULTS;
this review does not retroactively pass it. The common sliding-control inlined
RoPE stack/local arrays are not a demonstrated added spill or correctness
defect. Reassessing that admission policy before a first official run is
separate from the preserved experiment outcome. D16/S9 and D32/S5 candidate
specializations remain locally uncompiled. Native reduction rounding, replay,
empty/tail hazards and runtime compilation remain unknown; one runner check
may fail on them. TASK checks every timed call against its original reference,
but a pass would not establish targeted corner/concurrency coverage or justify
formal promotion. Selected1.233633 and A2-11 remain preserved.
