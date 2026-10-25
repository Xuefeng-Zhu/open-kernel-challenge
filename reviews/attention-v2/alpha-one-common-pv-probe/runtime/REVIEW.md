# Independent complete-runtime harness review

**PASS for the reviewed local source/protocol and bounded experiment harness.**
No concrete worker, diagnostic, original-oracle, timing-separation or controller
contract blocker was found. This is a pre-execution review, not a numerical,
frequency or performance result. The parent must still freeze these exact
bytes, perform actual offline preflight, and satisfy the live serialized-slot
and free-credit/paid-$0 admission checks. No additional user confirmation is
introduced by this review. Only this report was written by this reviewer;
all tests were CPU-only, with no GPU, service, credential or network action.

## Exact source and reviewed files

| Artifact | SHA256 |
|---|---|
| Working control `../baseline.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Common-PV candidate `../candidate.py` | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |
| `worker.py` | `c3e9193d8c7dbb8cd68c06ea363b2f77cc8b4d64657be6a20ad832ac3e1739bb` |
| `test_worker.py` | `48cc3db91fc2d5023901c695611a5383f8847af526ebf9e87b4fc67107cbb4ca` |
| `controller.py` | `ab4b88f430ab2934f4cd0e75ce5bfa1b16fd1f6c1ca3801631a3f0b77ef7f0e3` |
| `test_controller.py` | `2c1f957f5d454376a5d3a18ed6f170e938a827d5452d23aa1ddf5764d711bd07` |
| `diagnostic.py` | `ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60` |
| `test_diagnostic.py` | `70cb800eb8cf08833170fc157d741a56822a10b4d91e05d7eca7b7f08416cdb0` |
| `PROTOCOL.md` | `7d76b2ba37d069b53de8dad30c7e29b52dd65cb675e451365e49d33cd01d0548` |

The runtime source gate independently reproduced the two algorithm hashes,
diagnostic hash and unchanged support hashes: original reference
`386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b`,
original checks
`9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055`
and clone/restore/capture helpers
`9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e`.

Independently run CPU tests passed: seven candidate-source tests, twelve
runtime-worker tests, five diagnostic-source/count-model tests and ten final
controller tests. The candidate-source tests reverse the sole common-PV
historical branch to the entire control AST. The diagnostic tests reverse all
five instrumentation patches to every candidate byte and the complete AST.
No candidate precision, denominator, new-KV tail, geometry, dispatch or merge
change is hidden in the harness. No imported algorithm or oracle is patched.

## Six cases and original correctness contract

The six declared cases are the unmodified scored B32/L1 TP8 agentic/coherent
target, three explicit first/middle/last history-maximum adversarial cases,
one DP128Q/16KV multi-chunk boundary case, and the unchanged full-B16/L8
agentic/coherent fusion control. Seeds are the new135731/135732 fixtures.
Boundary vectors include zero history, tile/page boundaries,8191/8192/8193
and reversed request order. Page suffix masking uses the actual page size;
new slots are recomputed consistently, live physical row0 is exercised, and
serving-supplied decode scheduler metadata is rebuilt outside graph execution.

Forced cases write Q and new-K views of the fused QKV input before capture.
Every valid Q head has dimension0=1 and all other dimensions0; new K is0.
Historical K is0 except raw representable E4M3 dimension0=32 at a declared
valid leader row per request/KV head. The worker checks actual leader positions,
nonzero component counts, positive reference/encoded Q and zero new K. Thus
both reference and candidate see a positive leader score above background0.
The independent running-max model correctly counts a first slow tile in every
nonempty split, plus a second slow tile only when its leader appears after
that split's first tile. A leader at the next chunk's first tile is not counted
twice. V, new V and gate remain fresh. These are adversarial fixtures, not
untouched scored distributions; ordinary target and DP fixtures retain normal
nonzero cache-write observability.

Both unchanged complete entrypoints execute in captured graphs. The original
`reference.check_implementation` remains the oracle with its original bounds.
Collecting output/new-KV as uint8 and comparing uint8 views preserves signed
zero and other byte distinctions; floating equality alone is not used as the
exact-output proof. Finite output, immutable inputs, untouched cache rows and
stable input/output addresses remain mandatory.

Six cases × two arms × A→B→A give36 oracle calls, twelve exact ABA return-A
checks,36 whole-cache K/V byte comparisons and18 cross-arm output/new-KV byte
comparisons. Every graph is captured once; input values, including live lengths
and page mappings, change inside its existing allocations rather than by
recapturing fixture B.

The middle-maximum independent candidate graph establishes its own A oracle
expectation before submitting primary A/independent B on separate streams.
Their two additional oracle checks make39 total. Primary→B and independent→A
then reuse each graph's own established expected bytes, with both active and
inactive retained state checked after each replay: four isolation checks and
two extra replays. Expected outputs are not transferred between graphs or
used to compute the candidate output.

Distinct live output/cache pointers are checked, as are distinct default graph
pool handles when that supported API is available. Missing handles are recorded
as an observation limit. Internal scratch pointers and actual hardware overlap
remain unobserved; successful separate-stream submissions do not certify
overlap, same-graph simultaneous replay or callers sharing allocation pools.

## Diagnostic faithfulness and frequency admission

The separate module adds three scalar int32 counters per CTA and three owned
stores after the history loop. It reuses the exact already computed
all-valid-query-row alpha-one predicate. Counter values do not feed floating
attention expressions, addresses, new-token ownership, dispatch or merge.
The row mapping `(b*HKV+kvh)*SPLITS+split` is injective. Historical visited
tiles are independently checked for every CTA, with `fast+slow=visited`,
nonnegative integer triples, empty-split zeros and forced transition counts.
The new-token tail is deliberately excluded from these counts.

Five cases × two versions give ten diagnostic calls and ten uninstrumented
own-oracle-expectation reference replays. Each diagnostic gets a newly allocated
contiguous CUDA int32 `[B*HKV*SPLITS,3]` tensor reset to−1 before replay. Every
row must be overwritten, so an unwritten empty CTA cannot masquerade as zero.
The diagnostic output/new-KV and entire post-store K/V bytes must match the
exact uninstrumented candidate state on that same fixture. Only the exact
designated `alpha_stats` output is omitted from that runner's immutable-input
view; no other input or uninstrumented runner receives an exemption.

Any diagnostic count, byte, input, cache or address failure stops the timing
branch. Statistics describe the faithful diagnostic's computed predicate;
they are not uninstrumented hardware counters or cycles saved. Instrumentation
can alter code generation even when observed outputs match. In particular,
the accepted compile artifacts contain predicated FMUL slots, so their issue
cost cannot be inferred from fast counts.

After all ten calls pass, each ordinary scored fixture must have a fast count
greater than0. No tuned percentage threshold is added. If either fixture has
zero observed fast tiles, the worker returns completed negative evidence with
no timings, no timing replay counts and benefit false. That is a valid negative
result, not an execution error or permission to tune the diagnostic.

## Uninstrumented paired timings and fixed benefit test

Only the original two uninstrumented entrypoints are timed, and only after
all correctness and frequency gates. Diagnostic capture, statistics, fixture
creation, metadata, input/cache restore, validation and256MiB L2 eviction stay
outside each CUDA-event interval. Complete graph device work stays inside.
There is no empty-graph subtraction or comparison of absolute Modal times
against calibrated official values.

Each of target/control has four warmups and twelve measured two-arm samples.
Adjacent observations reverse order with the same fixture. Six pair geometric
means provide exactly three pairs per fresh fixture. All64 arm replays must
match their own original-oracle-checked bytes and immutable/untouched state;
32 further cross-arm output/new-KV comparisons are mandatory.

The fixed decision requires target median paired latency reduction≥2% in
each fixture, every one of six target pairs favorable, and control slowdown≤2%
in each fixture. It does not accept a pooled gain that hides a failed seed or
control. A completed failed benefit test remains a valid negative outcome.
No instrumented time or counter-to-cycle model enters the decision. Even a
pass still requires the declared fresh-input confirmation and review before
promotion or a complete official check; none is automatic in this controller.

## Controller, closure and remaining admission

The final controller pins exact committed worker/source/support/diagnostic
bytes, parses Python in default offline preflight, and uploads only nine
allowlisted source/license files. Credentials and local admission/billing
receipts are excluded. The protected existing supported config is selected
without credential renewal or broader access. It checks live paid0, verified
free-credit floor, resource rates and no active apps before one invocation.

Bounds remain one H200/2CPU/8GiB-host container, zero retries,160s worker,
170s child,185s function,20s startup and210s watchdog with at most$0.58 free
reserve and actual paid authority$0. Raw JSON is bounded and transport-hashed,
and saved before claimed-success validation. Full timing completion and the
explicit zero-opportunity negative each require their exact count contracts.
The final ledger distinguishes planned timing from actual timing: reservation
starts `performance_measurement=False`; terminal actual state follows whether
timing records exist, with `timing_complete` retained separately.

Stop requests and final independent stopped/tasks0 verification remain
required even for a negative or failed worker. Initial stopping/tasks1 or an
uncertain action must be preserved and reconciled, never rewritten or retried.
Slot release and cutoff-qualified cost accounting precede later measurements.
`CONTROLLER_REVIEW.md` records the separate integration review; this report
does not claim live admission, app closure or a runtime result already exists.

No concrete local harness blocker remains on these exact bytes. Native
numerical behavior, observed diagnostic frequency, benefit, broader serving
coverage and actual concurrency overlap remain questions for the bounded run
and its independent result review, not properties proved by these CPU tests.
