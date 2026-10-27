# Independent DP multi-chunk source and protocol audit

The proposed two-case correctness diagnostic targets an actual gap in the
working composition's admitted dispatch. It does not change the algorithm,
precision, tolerance or selected candidate. Source, protocol and final worker
review pass. Controller review and live source-bound admission remain separate
requirements before launch. This audit used local reads, CPU protocol tests
and an integer address/partition check only, with no service, GPU, credential
or network action.

## Exact source and route

| Artifact | SHA256 |
|---|---|
| Working composition, `../fusion-lookahead-probe/candidate.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Fusion control, `../fusion-lookahead-probe/baseline.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Original oracle, `../official/reference.py` | `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b` |
| Original checks, `../official/checks.py` | `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055` |
| Clone/restore/capture helpers, `../modal_worker.py` | `9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e` |

The composition and fusion hashes were recomputed from current bytes. The
worker must independently check all source/support hashes again before
import/execution. The unchanged candidate was officially accepted in
`5090b83f1d63407b`; this diagnostic is not another official score.

The candidate entrypoint at lines585–602 sends full B32/L1 to `_e2_kernel_fn`.
The fused-FA3 route at lines587–589 explicitly excludes `(32,1)`. Inside
`_e2_kernel_fn`, line316 enables lookahead for B32/L1/full/no-RoPE without
restricting head counts, pages or scales. Official `HEADS` maps DP to128 query
and16 KV heads (`reference.py:18`). Thus DP B32/L1 exercises true lookahead,
not a FA3 fallback or an unreachable proposed configuration.

For both cases `G=128/16=8`, `BLOCK_M=max(16,next_power_of_2(8))=16`, `D=64`,
`BLOCK_N=64`, `CHUNK_SIZE=8192`, `SEPARATE_MERGE=True`, four warps and three
stages. Source lines303–306 derive split count only from fixed page-table
capacity. GPU length values remain live inputs at `_fused:96` and determine
historical loop bounds within that fixed launch.

## Chosen two-case boundary coverage

The original historical next-plan proposed two SPLITS2 cases. This audit
recommended replacing its page1 case with one SPLITS3 case, and the parent
adopted that change before implementation/compute. It adds non-power-of-two
merge padding within two total cases, not a split parameter sweep.

| Case | Page/scale/distribution | Capacity and launch consequences |
|---|---|---|
| DP ctx8193 | page64, scale1/64, soft |129 columns, capacity8256, SPLITS2, merge BS2 |
| DP ctx16385 | page1, scale1, coherent |16386 columns, capacity16386, SPLITS3, merge BS4 with one masked lane |

Both use B32/L1, DP128Q/16KV and D64. For the first, lengths are
`[0,1,63,64,65,8191,8192,8193]×4`; for the second they are
`[0,1,63,64,65,127,128,129,8191,8192,8193,16320,16321,16383,16384,16385]×2`.
Fixture B reverses each vector and uses a fresh seed. These cover zero history,
page/tile endpoints, the8192 boundary, a one-token last historical chunk,
the16384 boundary and a one-token third historical chunk. Maximum new token
positions still fit the declared pool because capacity includes `ctx+q_len`.

The source producer grids are `(32,16,2)` and `(32,16,3)`, respectively1024
and1536 logical CTAs; merge grids are `(32,128)`,4096 logical CTAs. These are
static launch dimensions, not achieved-residency or observed CTA traces.
Private FP32 partial allocations total4,325,376 and6,488,064 bytes, respectively,
from `n=32×16×SPLITS×16` and `(64+2)×4` bytes per partial row. No occupancy or
performance conclusion follows from those sizes.

A local integer check enumerated every history token of all32 lengths in each
chosen case. Split intervals cover `[0,seq)` exactly once; every valid initial
or lookahead page index lies within the live prefix; the new token index is
within `ceil((seq+1)/page)` live pages and fixed capacity. The merge predicate
keeps exactly the actual split indices and excludes the fourth BS lane in the
SPLITS3 case. This is an address/partition check, not GPU or numerical proof.

## Empty splits, final partial and new KV ownership

The producer computes historical interval
`[split×8192,min((split+1)×8192,seq))` at lines122–124. Intervals whose start
is beyond `seq` execute no historical iterations; their initialized partial
is `m=-inf,l=0,acc=0`. They still write their own allocated partials.

Only `split==SPLITS-1` executes new-token norm/quantization, cache stores and
new-token attention at lines177–202. That final capacity split must remain
present even when its historical interval is empty. This is especially
important for seq0 and seq8192 in SPLITS2 and for seq0/8192/16384 in SPLITS3.
An optimization that simply deletes empty history CTAs would lose new KV.
This experiment changes no such behavior.

Producer partials use the actual SPLITS allocation stride at lines213–217.
The L1 merge at lines250–278 uses query head's valid `g`, masks split loads
with `offs_s<SPLITS`, and gives padded lanes `m=-inf,l=0,o=0`. Empty historical
partials receive zero softmax weight when combined with the finite final
new-token partial. The SPLITS3/BS4 case directly exercises that masked lane
and final partial rather than assuming a power-of-two reduction is sufficient.

## Fixture and replay requirements

The old TP8 boundary harness at `fusion-lookahead-probe/runtime/worker.py:219`
hardcodes64 in its unused-page suffix calculation. Copying that expression
would be incorrect for the new page1 case. The new harness must use
`used_pages=ceil((seq+L)/config.page_size)` for every request, compute valid
`cache_loc` before masking unused suffixes, and verify its exact agreement
with `reference.new_slots`. It must ensure all history/new positions remain
mapped, unused table entries are negative, physical live row0 is exercised,
and independent request new slots do not alias.

Generate each fresh fixture through the unchanged official generator, then
apply the declared length/page remap consistently to `seq_lens`, `positions`,
`cache_seqlens`, `cache_loc` and serving scheduler metadata. Metadata generation
belongs to fixture setup outside graph execution. The custom path may ignore
supplied FA3 metadata under the official contract. Coherent values initially
come from the generator; later boundary length changes create a valid nearby
fixture, but should not be described as an identical scored coherent recipe.

The existing clone/restore helper copies tensors into the same allocations.
Capture one graph per arm/runner, restore A→B→A into fixed addresses and retain
each graph. Both lengths and page tables must demonstrably change. Do not
recapture B or reuse expected outputs as a computation substitute. The original
oracle remains `check_implementation` with its unchanged output/new-KV bounds;
also check immutable inputs, untouched cache rows, stable addresses, exact
return-A bytes, and cross-arm entire K/V cache bytes after every replay label.

For two cases/two arms/three ABA labels, expect12 original oracle calls,
four exact return-A checks and12 K/V cache-byte comparisons. The independent
second candidate graph first establishes its own A expectation with one
additional original oracle call. A separate-stream pair then adds two original
oracle calls, for15 total. Two post-submission replays reuse both graphs'
own established expectations: primary A→B, then independent B→A, with the
inactive graph checked without replay each time. These four exact isolation
checks are reported separately, not counted as extra original oracle calls.

## Observable ownership and concurrency limits

The custom entrypoint allocates a fresh output and `part_o`, `part_m`, `part_l`
inside each invocation at lines307–312. Those tensors become captured graph
allocations. L1 uses separate merge and aliases the unused ticket argument to
output at line317; the producer does not enter its ticket/atomic branch when
`SEPARATE_MERGE=True`. No new global counter is introduced.

Independent default graph captures with both graphs alive should have private
allocation pools. Record pool handles if the installed supported API exposes
them, and verify distinct output ranges plus disjoint writable caches. Output
and input addresses should remain stable within each graph. Keep saved output
bytes from one graph while replaying the other and verify they are untouched;
then check the inverse. This supplies observable output isolation and repeated
scratch-use correctness without modifying kernel definitions or allocators.

Internal scratch tensors are not returned. Source allocation proof, distinct
graph pools if observable, and successful independent replays support scratch
ownership, but do not constitute direct scratch pointer inspection. Do not
monkeypatch `torch.empty` or the candidate to expose them. If pool handles are
unavailable, record that limitation rather than inventing identifiers.

Submit independent captures on separate streams, join them safely and run the
original oracle for each output/cache. Successful submission does not establish
actual overlapping GPU intervals. This bounded correctness diagnostic does
not add a profiler, NCU, counters or an overlap experiment. Same captured graph
overlap and callers explicitly sharing graph pools remain outside its claim.

## Admission and interpretation

The final harness review must bind exact source/support/worker/controller bytes,
check the page-specific remap and all counts, and retain strict deadlines,
one serialized app, zero retries and independent stopped/tasks0 reconciliation.
Existing free credits and live paid cap$0 are admission requirements. No new
install, credential or official check is required by this source audit.

A pass would close this targeted DP multi-chunk plus padded-merge coverage gap
for the exact selected candidate. It would not produce a new official score,
prove universal serving/concurrency safety, or justify spending a check on the
same source. A failure should preserve partial receipts and isolate the exact
contract before any candidate change; there is no preauthorized relaxed oracle
or automatic retry in this protocol.

## Final worker implementation check

The reviewed `runtime/worker.py` SHA256 is
`4507a19cdaea450415b6d7a40315e9e0021741919f5a33842fed290e0f2af59d`.
Its seven CPU test methods passed independently. The source gate reproduced
the exact two candidate/control and three support hashes; shape geometry and
the post-submission isolation schedule were also evaluated independently.

The worker implements the declared two cases and both32-length vectors,
page-specific `valid_pages`, live row0, unique new slots, negative unused
suffixes and freshly rebuilt supplied metadata. It keeps the original oracle
call and tolerances, uses the unchanged complete entrypoints, and performs no
benchmark, profiler, loaded-module mutation or algorithm override.

Final code contains the own-version setup oracle for the independent graph
before its B submission. This resolves a pre-launch harness issue: a nonoracle
check must never request an expectation that that graph has not established
with its own oracle. No expected bytes are transferred between runners. The
final isolation schedule is primary→B/check-independent-B, followed by
independent→A/check-primary-B. Both graphs therefore reuse captured allocations
and mutate input state against their own established expected bytes.

The final count assertion is15 original oracle calls, including one independent
setup call and two independent-submission calls; four ABA return-A checks;
12 cross-arm K/V byte comparisons; four post-submission exact isolation checks;
and two additional candidate replays. Controller expected counts were checked
to agree at the time of this audit, but the controller has its own review.

Pool handles are read only when the installed graph API exposes a callable
`pool`; absence is explicitly reported, not treated as proof of separation.
When both handles are available, equality fails closed. Distinct output and
writable cache pointers are checked while both default captures remain alive.
No direct internal scratch-pointer or actual hardware-overlap claim appears.

The worker retains a160-second alarm/deadline, bounded400KB atomic partial
JSON, source checks before GPU imports and a failed/incomplete status on an
exception. Its CPU failure-path test confirms an exact source-gate failure
returns a bounded partial report before any GPU import. No concrete remaining
worker/source correctness blocker was found for this bounded diagnostic.
