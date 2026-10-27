# Page1 physical-locality fixture: CPU plan and invariants

This is a diagnostic input fixture, not a kernel candidate, official upload or
performance result. The actual selected kernel remains exact3da. It changes
physical KV placement only while preserving logical attention inputs.

Two fresh fixed seeds are135831 and135832. Runtime cases are B1/L1/sliding2048,
B16/L8/full and B32/L1/full, each TP8 HQ16/HKV2/D64, page1, ctx80000, unit scales.
Use the original generator per seed; QKV, positions, all lengths, cu_seqlens_q,
scales, config and contract-provided scheduler_metadata remain unchanged.

## Mapping and complete-byte preservation

`build_plan(page_table_a, cache_seqlens, cache_loc_a, q_len, capacity_pages)`
accepts CPU Python integer sequences and returns `Page1LocalityPlan`.
The original table has valid prefix length cache_seqlens[b], which includes
all L current new tokens, and a -1 suffix. The B table maps valid logical j
to physical page b*PTwidth+j. Its shape and -1 suffix stay the same.

The plan constructs full-capacity inverse permutations, not only a live-page
mapping. `K_B[new] = K_A[new_to_old[new]]`, and similarly for V. Remaining
unused source pages are assigned to remaining unused destination pages in
ascending order. Thus all encoded bytes of both complete caches, including
unused pages, survive. `old_to_new` is the exact inverse, and new cache_loc
is that inverse applied to the original cache_loc. It equals the canonical
page of each request's last L logical tokens. Original KV capacity is retained.

The table and input location containers are copied into tuples. Compact
`array('q')` maps are separately allocated; callers must treat their contents
as read-only. There are no Torch/vendor imports or device calls. The plan is
bounded to4,000,000 physical pages, sufficient for the fixed B32/80K cases.
`permute_rows_bytes` is an opaque-byte validation helper capped at16MiB;
it must not simulate/copy the full B32 caches on CPU. Large-runtime adaptation
uses the same row-index maps with GPU index_select or equivalent.

## Runtime adapter responsibilities

The fixture plan does not prove runtime pointer invariants. The adapter must:

1. Prepare A and B fixture bytes before capture. Permute whole physical K/V
   rows without numeric conversion or changing FP8 dtype/capacity/strides.
   Preserve page-table dtype/shape/stride and cache_loc int64 dtype/shape.
2. Capture actual selected kernel_fn once against the actual input tensors.
   Restore A/B at those SAME tensor addresses with copy operations outside
   the timed graph; do not replace tensors or recapture for the comparison.
3. Restore original A or permuted B cache bytes before each call because the
   call appends new KV in place. Copy both page_table and cache_loc for each
   fixture. Preserve all other input bytes, including supplied scheduler
   metadata; physical page placement does not change its logical schedule.
4. Prove logical K/V equality, new-write equivalence, untouched-cache bytes,
   original oracle checks and fixed output/input addresses before timing.
   Apply inverse permutation for whole-cache comparisons as needed.
5. Keep observed locality timing separate from kernel/algorithm performance.
   A useful A/B result is evidence about sensitivity of this exact baseline
   to the physical mapping; it is not a production repack speedup, free
   preparation optimization or evidence that an upstream tile is installed.

## CPU validation

Eleven independent CPU tests pass. They check full-capacity bijection/inverse,
all opaque K/V bytes, logical rows, new KV writes for small models of the three
case families, physical slot0, unused pages and malformed-input refusal.
Full/sliding causal key sets are checked for every current query through
history80000 with address-only large models. Tests also restore A/B/A bytes
in the same bytearray objects and reject duplicate live pages across requests,
missing/OOB live pages, non--1 suffixes, ragged tables, inconsistent new-token
cache_loc and insufficient canonical capacity. Both unit and1/64 scale labels
are covered as opaque bytes; no FP8 arithmetic is simulated.

Run from repository root:

    python3 -B -m unittest discover -s reviews/attention-v2/locality-probe -p 'test_fixture.py' -q

No authentication, service, GPU, vendor import, network or Git operation was
performed by fixture preparation. Native correctness, graph-address invariants,
physical-locality effects and performance remain unobserved.
