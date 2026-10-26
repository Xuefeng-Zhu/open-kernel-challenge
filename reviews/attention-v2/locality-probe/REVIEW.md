# Independent locality diagnostic review: local PASS

The final fixture/worker source and CPU-only preflight pass local review.
No remaining source-level blocker was found for the declared diagnostic.
**No native execution, authentication, service, GPU, Docker or Git operation
was performed by this review.** Native eligibility, actual address invariants,
original-oracle results and locality sensitivity remain unobserved. This is
external diagnostic code, never an official candidate or a production change.

## Exact reviewed files

| Artifact | SHA256 |
|---|---|
| `fixture.py` | `203240b0f4de073b4c6222c62daeeaff20ce0b3de32b519f8d9f59c5528e0e28` |
| `test_fixture.py` | `24f11ea7c9f412104af10d0f069e24f5426176179e9fa22a51264204b46de821` |
| `worker.py` | `c353b5e8682e586526d56273ada331588d518b744c78c7993ee0fd22d59c91f7` |
| `test_worker.py` | `2d7c637ec1b226a6ef0618d083e398ffd37964475f873ff6cd044c105debeec0` |

The worker also source-gates selected3da, original reference386f7f... and
checks9a046c... against their complete expected hashes, refusing changed or
symlinked source files before execution. The default command independently
returned `CPU_preflight_only`, zero vendor imports/kernel launches/service
calls and those four verified source hashes. Twenty-five CPU tests passed
independently:11 fixture tests and14 worker/protocol/parser tests. Neither test
success nor source binding is an execution result from the remote image.

## Fixture invariants

The plan maps every valid page1 logical token j of request b to canonical KV
row `b*page_table_width+j`. Original table shape and every unused −1 suffix
are retained. It rejects live duplicate/cross-request aliases, invalid page
IDs, inconsistent new-token cache locations, malformed lengths and insufficient
capacity. These restrictions match the declared original-generator fixtures;
they are not a general shared-prefix serving contract.

`new_to_old` and `old_to_new` are inverse permutations across the **entire
physical KV-row capacity**, not just live prefixes. Unused source rows fill
unused destination rows bijectively. Thus `B[new]=A[new_to_old[new]]` preserves
all encoded bytes and `B[old_to_new[old]]=A[old]` recovers the full old cache,
including unused rows. K and V are independently permuted through UINT8 views,
without FP8 arithmetic or scale changes. Original new-token cache locations
map through the same inverse; the candidate's unchanged append consequently
writes equivalent logical rows in both layouts.

The runtime explicitly records actual attended/history row movement, page-table
hashes and adjacency counts. It requires moved attended historical rows,
fully contiguous canonical adjacency and strictly fewer contiguous adjacent
pairs in the original layout. Identity input, changes only outside the attended
window, or a merely shifted already-contiguous layout cannot satisfy the
declared contrast. For the sole sliding case L1, the inspected key interval is
the actual last2048 tokens; full cases cover the full causal union. This helper
must not be generalized to sliding L8 without revisiting its interval formula.
These observations concern KV tensor row addresses; they do not establish
physical DRAM-bank/TLB/cache behavior or actual HBM transaction counts.

## Runtime and replay design

The three fixed cases are B1/L1/sliding, B16/L8/full and B32/L1/full; all are
TP8/D64/page1/80K/unit scales. Two fresh seeds are135831 and135832. Per seed,
QKV, positions, all lengths, cumulative Q lengths, scales, config and supplied
decode scheduler metadata are byte-identical across layouts. Only whole-cache
placement, page-table values and cache locations change. This keeps the page
table present and preserves source, capacity, strides and logical schedule.

One actual selected graph is captured per case against one persistent `work`
dictionary, reused for both layouts and both seeds. `restore` copies every
input tensor into those same addresses outside the timed interval and rejects
shape/dtype/stride or structural-metadata changes. Captured input and output
pointer/shape/stride/dtype/device records are checked after oracle replays and
every warmup/measured/activity replay. Graph pool handles are recorded; no
simultaneous execution or hardware-overlap claim is made.

Each layout is checked by the unchanged original oracle. Ownership checking
uses that layout's own physical new-token locations, temporarily masking those
rows only outside timing, then proves all historical/unused bytes untouched
and restores the new rows. Full post-call B caches are inverse-permuted and
compared byte-for-byte with A, including equivalent new KV. New-row snapshots
use opaque UINT8 selection, avoiding unnecessary FP8 indexing requirements.
Per-layout oracle-checked outputs/new rows are checked on later replays;
cross-layout exact output equality is observed separately, not assumed from
the byte permutation or substituted for the original numerical oracle.

## Counts, timing and activity

There are six case/seed fixtures and two layouts each:12 original-oracle
calls,48 checked warmup replays and144 measured replays. Thus warmup plus
timing is192 replays, and oracle+warmup+timing is204. Optional activity adds
six checked replays, for210 total. Nine initial function warmups and three
capture function calls are explicitly excluded. Attempted, completed and
successfully checked replay/oracle counts are separate on failure; a partial
run cannot masquerade as the planned successful count.

Each fixture has six reverse-order pairs, with arm order balanced inside
each pair and alternating across pairs. Pair ratios are summarized without
mixing fixtures, layouts or seeds. Input restoration and256MiB L2 eviction
occur outside events. Between `begin.record()` and `end.record()` there is
only the unchanged captured graph replay; bookkeeping lies outside the event
interval. Raw complete-call timing has no replay-cost subtraction. It measures
this controlled fixture sensitivity, not an official score or a practical
repacking optimization whose preparation is free.

One optional CPU/CUDA activity session replays both layouts for each case
using the final seed, separately from timing. Fenced ranges require real
nonempty full kernel names and preserve bounded raw trace bytes. Symbol/node/
launch comparisons use multisets so overlapping event start order cannot
create false identity differences. Missing or invalid launch fields return
**UNKNOWN**, never a parity pass from `None == None`. Downstream reports must
inspect `native_identity_fully_observed`; a completed diagnostic with missing
metadata is not proof of observed geometry/resource equality. Explicitly
different observed identities stop the worker.

The optional trace is bounded at8MiB raw and2MiB encoded transport, with256KiB
combined native symbols and at most64 kernels/range. Oversize evidence fails
rather than being silently truncated. These fields are static/instrumented
observations, not SASS, instruction dtypes, resolved PackGQA/split counts,
achieved occupancy, CTA residency, hardware counters or a bottleneck diagnosis.

## Fixes verified and remaining gates

Review caught the draft's missing-metadata false parity and missing explicit
attended-address contrast. The final source fixes both with regression tests.
Additional hardening includes scalar byte-view flattening, output-state checks,
opaque-byte new-row selection, separate attempt/completion counters and empty
native-symbol rejection. No performance gate was relaxed and no source
algorithm was changed to obtain a favorable result.

The160s in-worker alarm/deadline and bounded result handling are useful guards,
but a future admitted remote run still requires an independently reviewed hard
child/function/watchdog timeout, authenticated free-credit/$0-paid admission
and explicit app closure. This review did not review or admit such a controller.
Existing credential expiry remains a service blocker; it is not permission to
renew credentials, use another profile or route around a denial. No native
result, resource closure or cost receipt is implied by this local PASS.

Only this review document was written; fixture, worker, candidate, selected
records and other-track files were not edited by the reviewer.
