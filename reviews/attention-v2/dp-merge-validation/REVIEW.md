# Independent pre-GPU DP harness review

**PASS for the reviewed local protocol and implementation.** No concrete
source, correctness, accounting-control or payload blocker remains. The parent
must freeze these exact worker/controller bytes in Git and pass the controller's
live admission checks before its one granted native invocation. This review is
not evidence that a GPU job has started, passed, stopped or incurred any cost.

This reviewer performed local source reads and offline tests only. No service,
network, browser, credential, installation, billing or GPU action was performed.
Only this review document was written.

## Exact scope and numerical contracts

The working candidate remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
and fusion control remains
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
The worker independently verifies those bytes, the existing graph helpers and
unchanged official reference/checks before importing GPU packages. It does not
patch candidate functions, allocators, tolerance constants or FA3 arguments.
Both complete `kernel_fn` entrypoints execute through the original capture
helper and unmodified `check_implementation` oracle.

The two fixed cases are DP128 query/16 KV heads, B32/L1/full/D64/no-RoPE:
page64, ctx8193, scale1/64, soft; and page1, ctx16385, scale1, coherent-derived
nearby inputs. Shape-derived split counts are two and three, with merge BS2
and BS4 respectively. The latter exercises a masked fourth reduction lane.
Actual configuration/table shapes are checked at runtime; exact source binds
these shape-derived launches and private FP32 partial allocations. Internal
scratch pointers are not exposed or directly inspected.

Fresh seeds135721/135722 and declared boundary vectors exercise zero history,
page/tile tails,8192/16384 boundaries and one-token final history chunks. The
new-token row remains owned by the final capacity split, including empty-history
splits. The page-specific suffix mask is correctly `(seq+L+page-1)//page`,
including the page1 case; the old hardcoded page64 divisor is not copied.
New slots, valid prefixes, negative unused suffixes, physical live row0 and
nonaliasing request new slots are checked. Changed lengths, positions, cache
lengths, new slots and actual supplied scheduler metadata are rebuilt consistently
outside capture. The generator's coherent values precede the boundary remap;
these are valid nearby fixtures, not an identical scored coherent recipe.

## Replay, output isolation and precompute fix

Each runner retains one graph and restores fresh tensors into the same allocated
addresses. A/B lengths, QKV and page tables must actually differ. The oracle,
immutable-input/untouched-cache checks, finite output, exact return-A bytes,
fixed input/output addresses and whole-cache cross-arm K/V equality are enforced.
The unchanged oracle itself also rejects nonfinite new K/V values.

Review found an early local harness defect: the second independent graph had
no own oracle-checked A expectation before a nonoracle A replay. This was
corrected **before compute**, without transferring expected bytes between graphs
or suppressing the check. The final worker first calls that graph's own A oracle,
then establishes its own B expectation in the separate-stream submission.
The post-submission isolation schedule now changes primary A→B and independent
B→A; each active replay compares with its own oracle-checked bytes, and each
inactive graph is checked without replay. A focused CPU expectation-state test
covers initialization and ordering.

Final expected counts match the controller exactly:15 original oracle calls
(12 ABA +1 independent setup +2 after concurrent submissions), four exact ABA
return-A checks,12 K/V byte comparisons, two concurrent oracle calls, one setup
oracle, four additional isolation checks and two additional candidate replays.
The four isolation checks are not counted as oracle calls.

Independent candidate captures own distinct live outputs and writable caches.
The supported graph-pool handle is recorded when available, and observed handles
must differ. Missing handles are explicitly qualified, without fabricated
scratch addresses or claims of shared storage. The source's fresh partial
allocation plus default-pool captures and successful reuse can support ownership
for these cases. They do not prove universal serving concurrency, direct scratch
pointer separation or same-graph overlapping replay. Separate stream submission
is explicitly not a measurement of actual hardware overlap. No profiler,
counter, timing loop, benchmark gate or new official score is introduced.

## Admission and closure controls

The controller uploads exactly eight allowlisted files: the new worker,
unchanged control/candidate, existing helper/reference/checks, license and
third-party notices. The controller is also verified against the frozen Git
object but is not uploaded. The remote entry is closure-free and contains no
local configuration, slot/billing receipts or credential payload. The fixed
existing image is `im-cPTbGpNiX3RarKResPgX7h`; no software installation is requested.

Explicit launch requires an exact committed HEAD, a fresh parent serialization
receipt, fresh paid-cap0/free-credit evidence, supported CLI actual-paid0,
all apps stopped/tasks0 and current resource rates. The existing protected
credential is selected only via supported config/profile paths; there is no
renewal or environment expansion. The fixed$0.58 reserve must exceed twice the
full watchdog's live resource rate plus$0.02. No removed artificial stage cap
is reintroduced, and no payment/billing setting is changed.

Bounds are160s worker,170s child,185s function,20s startup and210s watchdog;
one H200,2 CPUs,8GiB host memory, maximum one container, one invocation and
zero retries. An exclusively created local ledger/result prevents duplicate
launch. Worker partial JSON is atomic and bounded400KB; controller transport
is bounded500KB and exact SHA256. Claimed success additionally requires the
frozen worker/source/support hashes, exact15/4/12/2/1/4/2 counts, correctness-only
protocol and no timings or official admission.

The app is stopped in normal and exceptional exits. The controller preserves
initial shutdown observations and independently reads app state/billing; a
stopping/nonzero-task state is not closed and needs parent reconciliation.
Actual post-run closure and delayed workspace cost accounting remain future
observations, not conclusions of this local preflight review.

## Offline verification and reviewed hashes

The existing local Python independently ran **7 worker tests and8 controller
tests, all passing**. These cover source tamper rejection, exact cases/geometry,
page-specific extent/new-token boundaries, independent expectation initialization,
no algorithm/tolerance/performance mutation, atomic bounded partial failure,
paid0/free-credit/rate/slot guards, stopped-state interpretation, transport
corruption/overflow, closure-free credential-free payload and mocked committed
preflight. They do not establish native numerical correctness or performance.

| Artifact relative to this validation | SHA256 |
|---|---|
| `PLAN.md` | `80c54d61920e88afdb58d4b62bb523fe267e6e68b4ba8eba52c442d17316cd53` |
| `SOURCE_PROTOCOL_AUDIT.md` | `f3d56ad4b152b6da46d6534b2476dd19943c2a85180773259d31b090e5fd3868` |
| `runtime/worker.py` | `4507a19cdaea450415b6d7a40315e9e0021741919f5a33842fed290e0f2af59d` |
| `runtime/test_worker.py` | `251afb67be2a3c5f2ad904e32b4573f7799f05522950ccc0c52279f7f9b5d7c1` |
| `runtime/controller.py` | `0299612f835e68b26ece31f5d5d031c959953988ce4d998d95941ddd9d778f17` |
| `runtime/test_controller.py` | `a78756b5a94f5447c41ecff45f750bc8d8e03f77210ae72975718b0337a7b29b` |
| `runtime/CONTROLLER.md` | `97b4e979e0c8ad6f416a3bab8f66ce048e288b9a1cb244ce95abdbb2e6da3bae` |

No material pre-GPU blocker remains for this exact reviewed protocol. Freeze,
live service admission, one native execution, stopped/tasks0 reconciliation and
private export are the parent's subsequent actions, not actions taken here.
