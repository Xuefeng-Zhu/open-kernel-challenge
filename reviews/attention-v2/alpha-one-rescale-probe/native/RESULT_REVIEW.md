# Independent alpha compile result, resource rejection and receipt review

**PASS for preserving the completed negative experiment; NO onward admission
for candidate e66f502.** The exact source compiled and all artifacts were
retrieved, but producer registers96→128 violate the resource gate fixed before
the run. Generated V buffering also differs. No numerical, frequency or latency
experiment followed. This reviewer used local artifacts and offline analysis
only, with no service, GPU, browser, credential or network action. Only the new
analyzer, derived JSON and this review were written; frozen files were not edited.

## Source/transport identity and complete artifacts

App`ap-uY5K1rBNFRR2K8rdo2F8tw` used freeze
`99185609d7924d18bb07de950c3d689a9ec9e649`. All six receipt source/payload/
controller files match current bytes and exact frozen Git bytes. Control remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
rejected candidate remains
`e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764`.
Worker is`9e0377dc696518225b5091f65fe7f72ac5e96e6c2b539c399f13bf337a80a585`
and controller
`2ddfab210d7a589826674990a380d657a078b794ac73e175fb0fc5d27e9978bf`.

Raw response SHA256
`952637501db9fa11b6a53f354824a9566e24186be5fab090531667f39fbf8fea`
matches the launch receipt. The published receipt is byte-identical to its
retained local original, including initial stopping/tasks1. `analyze_result.py`
uses the frozen, stdlib-only controller decoder without calling launch/auth,
verifies bounded gzip framing/hashes and checks every decoded artifact against
both the per-file catalog and its saved bytes. The exact24-name set, complete
unit artifact maps and saved manifest all match. Gzip is393,593 bytes and its
uncompressed JSON body2,649,727 bytes, within the frozen limits.

Four units compile with identical options across arms: producer L1/G8/HQ16/
HKV2/D64, BLOCK_M16/BLOCK_N64/BLOCK_L16, CHUNK8192,SPLITS11, separate merge,
PAGE_LOOKAHEAD true, four warps/three stages; merge BS16. Logical grids are
(32,2,11) and(32,16), with qkv/table strides2304/86001. Runtime scalar and typed
one-element pointer alignment records match the reviewed compile specialization.
These undersized pointers were compiler metadata, not execution fixtures.
Zero candidate/merge launches, correctness_tested=false, performance_tested=false
and semantic_admission=false agree with the frozen warmup-only implementation.
The driver loaded module handles; no candidate or merge invocation occurred.

## Resource gate: compilation succeeds, admission fails

| Same-job unit | Registers | Shared bytes | Driver raw spill field |
|---|---:|---:|---:|
| Control producer |96|20,480|0|
| Candidate producer |128|20,480|0|
| Both merges |27|1,024|0|

The32-register/33.3% increase independently fails the predeclared no-increase
rule in frozen`ADMISSION.md`. Equal shared bytes and raw spill0 do not waive
that rule. PTX local accesses and SASS LDL/STL textual inventories are zero.
Merge SASS bytes are exactly identical, SHA256
`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`;
debug-bearing merge CUBIN/PTX/IR bytes are not falsely called identical.

Independent textual reconstruction confirms two2×64×64 E4M3 shared allocations
in the control versus one in the candidate, TTGIR async-copy sites6→3 and
commit sites6→3. The loop wait-group **parameter** changes2→1, with final wait0
in both; each PTX file still contains two wait-group instructions. Counts from
different compiler levels are correctly kept separate from executed frequency.

Direct TTGIR inspection at candidate332–337 shows current V coming from
`tt.load`, with FP8-dot and BF16-blocked materializations, rather than the
control's two-slot V async memdesc and V copy tokens. The baseline and candidate
loop-carried signatures differ, including full KV-address/mask tensors in the
candidate. These support the reported buffering/liveness change. They do not
provide a per-variable register-allocation map or prove a unique cause for the
32-register difference. Registers, static buffers and instruction inventories
are not measured latency, achieved occupancy, DRAM throughput or stall reasons.

## Mechanism evidence and numerical limits

The detailed artifact review's central observations were checked against the
fresh generated files: candidate TTGIR294–302 forms unordered-not-equal on the
computed alpha, masks with valid rows, reduces violations and tests zero. Its
TTGIR338–354 fast dot uses old carry`%arg23` as C; fallback first multiplies
that carry by alpha and uses the product as C. Denominator dataflow remains
outside that conditional, and the source new-KV tail remains unchanged.

The candidate PTX has a real branch to a slow block with eight carry multiplies
before BF16 PV MMAs, and an exclusive fast block without those carry multiplies.
Both join the common denominator path. This supports real skipped work at this
specialization; it is not merely a comparison of whole-file MUL counts. Static
PV duplication must not be called twice-executed PV. The lower-level SASS/CFG
and precision notes are preserved in`ARTIFACT_REVIEW.md` and the separate
`SEMANTIC_ARTIFACT_NOTES.md`; neither presents this as universal bitwise proof.

Finite scalar identity and retained logical C placement are insufficient to
certify all rounding, signed-zero/subnormal/NaN behavior. The compiled source
has not run the original oracle or replay/cache checks. The register and
buffering gate failure stops this exact variant before frequency counting,
correctness execution, timing or official checking. The predeclared2%-per-
fixture performance threshold was never tested. This is a failed compile-stage
admission, not a measured slowdown or evidence against every alpha-one lowering.

## Closure, accounting and working selection

Saved22:06:59.192653UTC reconciliation confirms this app stopped at22:06:25UTC,
tasks0, and all listed apps stopped/tasks0. The initial stopping receipt is
preserved separately. Worker exit0/status compiled is distinct from controller
exit1 caused by incomplete immediate closure. No receipt is rewritten to merge
those states. Worker elapsed19.308325546s and client31.753475792s are diagnostic
wall times, not kernel latency results.

The saved22:10:34.040391UTC accounting cutoff has the same app-ID set, all
stopped/tasks0 and no new apps. Decimal arithmetic reproduces rounded workspace
usage delta$0.03, detailed delta$0.02871385 and actual paid$0 from the initial
launch/cutoff billing objects. The retained live-rate formula
`ceil(client_wall_seconds+5)*rate*2+0.02` gives conservative allowance
$0.1165823333333333333333333334, below reserve$0.40. Delayed workspace deltas
are not exact per-app invoices; initial zero metering was not called free
execution, and later jobs are excluded by the cutoff.

`RESULTS.md` accurately records the rejection and its limits. The selected
working source/hash and official score1.233633059× with its job/interval remain unchanged;
A2-11 rollback and Projection objects match the pre-probe freeze. No official
quota was consumed or ranked entry made by this compile-only source. A proposed
common-PV lowering is a separate local hypothesis: it has no mechanism,
correctness or performance result in this experiment, and does not silently
replace the preserved rejected source.

## Reviewed hashes

Paths are relative to this probe.

| Artifact | SHA256 |
|---|---|
| `RESULTS.md` | `474e153b70dd6915cada04cbd0eeefbdaa67cfbd7c5fabaeb614c684841e7e27` |
| `native/ADMISSION.md` | `cc98e237028eeb5b4fa3ae6f548f8996f1a0042f206a03d82b3e3cbb42ed5b27` |
| `native/analyze_result.py` | `929f1d2d043b402e3e73a8e16b5c516126be2e9a00bcb36dcc14f0f7d59add3c` |
| `native/analysis.json` | `725824c4941411a6c5db7c9a6434347322f442e871b164b010211598fd0e6ffd` |
| `native/ARTIFACT_REVIEW.md` | `c122b0422d980e0315b9f9b25f4c84f2a6ecffe79e76ec21c89529370afb810a` |
| `native/SEMANTIC_ARTIFACT_NOTES.md` | `96b8b3dbc0ca16da51db7771a4bb15291517e2e293f00119eccd1d02bcf7c8e0` |
| `native/artifact-review-evidence.json` | `afeac7514d5ca47c289d5b4a894ed98c9625d6f9250d524b81c318f39b5c489b` |
| `native/result.json` | `952637501db9fa11b6a53f354824a9566e24186be5fab090531667f39fbf8fea` |
| `native/launch-receipt.json` | `5361b469cb6eb162370fcfaf04d0dbabf367b05bba2e6b61923d75b428eb5652` |
| `native/reconciliation.json` | `c3098eeb019ce619931c567f618b1cb15c00e77ea38ea845de6ecc77292d1d18` |
| `native/COSTS.json` | `2072ece90de0167f2d2a092d5f8015476c88669db642f39e87eb08c368c198b2` |
| `native/accounting-cutoff.json` | `1e2171b6f638566cf4f31aea347a083625470a86e38dcfe0dbe445360f493fab` |

The analyzer reproduces these actual saved artifacts and explicitly returns
resource_gate_passed=false, onward_admission=false and candidate_rejected=true.
It does not execute the GPU oracle or independently measure runtime hardware
behavior. No blocker was found to publishing the reviewed negative evidence;
private remote verification remains a subsequent parent action.
