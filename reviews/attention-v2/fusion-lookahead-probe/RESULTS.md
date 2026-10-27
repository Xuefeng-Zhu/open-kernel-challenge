# Fusion plus A2-15 composition: validated modest improvement

The exact composition passed the native oracle/replay/cache checks and one
complete official nonranking check. Native paired full-B32-decode latency fell
4.8226% relative to the validated preparation-fusion control. Official job
`5090b83f1d63407b` passed all16 prerequisites and12 benchmark shapes, with an
aggregate1.2336330588627955× and95% interval
[1.2151285629798536,1.2521375547457374]. The aggregate point estimate is0.6591%
above the historical fusion job; this is not a paired aggregate significance
result. The composition is now recorded as the working attention candidate;
original A2-11 is preserved as rollback. No ranked entry was made. This
engineering selection does not require claiming statistically proven aggregate
superiority.

| Evidence | Result | Scope |
|---|---|---|
| Native target, fusion→composition | 313.552→297.392µs raw medians;4.8226% balanced paired cut;6/6 favorable pairs | One session, two fresh fixtures |
| Native unchanged full-B16-verify control | 0.1532% balanced paired slowdown | Largest pair slowdown0.6382%, below5% gate |
| Official composition | 1.233633059×; all16 prerequisites/12 shapes pass | One complete nonranking check |
| Historical official fusion | 1.225555436×; CI[1.207172105,1.250062193] | Primary control job`2f5f1fb54b48486d` |
| Official changed full-B32-decode path | 304.031893µs vs same-run reference383.583994µs;1.261657090× | Reference-normalized cross-run latency cut vs fusion3.8251% |
| Official11 unchanged paths | Normalized reductions−1.8323%..+3.0215% | Largest apparent regression1.8323% |

The native≥4% replication gate was declared before compute and passed. The
official3.8251% normalized cross-run observation is a separate comparison; it
must not be described as an official≥4% paired reduction. The old A2-15 run's
10% target remains unmet. Its historical5.189% native gain was against A2-13,
whereas this experiment uses the already fused source as control.

## Exact isolated source

Independent branch/worktree `attention-fusion-lookahead` starts from verified
fusion commit`05c9cc08948c42fed9e09aae1cd2b41edf966c74`. Source freeze is
`c9062363c665e37535d3370c1b6bfb54e04fc7c8`. Only the exact A2-15 `_fused` and
`_e2_kernel_fn` definitions replace their fusion counterparts. Lookahead is
enabled only for full B32/L1 without RoPE. All fusion dispatch, FA3 helpers and
arguments, automatic splits, original tolerances and quantization, E4M3 Q/K
encoding with`max_num_imprecise_acc=0`, BF16 PV, FP32 softmax/partials,
8192-token chunks, split/warp/
stage choices, last-split new-KV ownership and merge remain unchanged.

| Source or harness | SHA256 |
|---|---|
| Fusion control`baseline.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| A2-15 donor | `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c` |
| Composition`candidate.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Native worker | `03c1261d8cb18b9c2ea266556463d087ee83d2da8cc121e1be80ff04fdc3cb15` |
| Native controller | `0b78f5a5147030f67701dcaf1298ba5077b947e133c6485478931e938014cbf1` |
| Original selected A2-11 | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |

The source audit checks the donor's exact saved ancestry and25 native artifact
hashes, the two-function transplantation and complete fusion AST reconstruction
when lookahead is disabled. Twenty-six local source/model/worker/controller
test methods passed before admission. The frozen preparation manifest is
retained unchanged; `result-manifest.json` covers subsequent published evidence.

## Correctness and measurement

Native fresh seeds135711/135712 differ from the donor's135701/135702. Five
whole-entrypoint cases cover the scored full-B32-decode target, scaled page64
tile/chunk/empty-history boundaries, unchanged B4 fallback, small B32 DP-head
single-chunk true-lookahead scope, and unchanged full-B16-verify control. All32
original-oracle checks, ten exact return-A checks,30 whole-cache cross-arm byte
comparisons and64 checked warmup/timing replays passed. Independent graphs with
disjoint writable caches also passed after submission on separate streams;
actual hardware overlap was not asserted. Inputs, untouched cache rows and
fixed addresses were verified without relaxing tolerances.

Unprofiled complete-call CUDA-graph timing used four warmups and twelve measured
two-arm observations per family, six reverse-order pairs, with pairwise fixture
version matching and256MiB L2 eviction outside timed regions. The median of
balanced pair ratios is primary; pooled raw medians are descriptive. Native
environment was H200 sm90, Torch2.14.1+cu130, CUDA13.0, Triton3.8.0 and
Python3.12.3 in existing image`im-cPTbGpNiX3RarKResPgX7h`.

The official evaluator matches fusion's complete evaluator dictionary, including
code/files/image digests. Both official runs report H200, driver580.159.03,
SM clock1980 and memory clock3201. Nodes and all prerequisite/benchmark seeds
differ. All twelve path rows and the six-load weighted score reconstruction
are retained in`official-check/analysis.json` and`OFFICIAL_REVIEW.md`; profile
events are not extra benchmark samples. The six loads each use46 sliding and16
full layers, followed by a geometric mean of load speedups.

The official aggregate difference is small relative to the reported interval
widths. Marginal confidence intervals do not test the cross-run difference, and
unchanged paths drifted in both directions. Reference normalization supports a
descriptive comparison but cannot attribute the full0.6591% aggregate point
change to lookahead. Native and official evidence support retaining this as a
modest validated candidate, not a demonstrated award-level architecture advance.
The donor changes page-load scheduling and generated K/V buffering together;
neither this test nor historical timing isolates their causal contributions.

## Resources, accounting and closure

Exactly one native Modal app`ap-EY8FNKYqqFMGgZAatYZXOQ` ran, serially after
Projection's slot release. Existing credentials/image were used without new
installs, credentials or billing changes. Live admission verified free credits
and paid cap$0, reserving$0.58. Reviewed160s worker/170s child/185s function,
20s startup/210s watchdog bounds and zero retries remained in force.

The worker completed in54.297s; client wall was66.630s. The app stopped at
21:15:59UTC. Independent21:16:21.481125UTC evidence confirms stopped/tasks0 and
all listed apps stopped/tasks0. The initial controller stopping/tasks1 receipt
is preserved unchanged; its incomplete shutdown observation is reconciled by
the later terminal read, not rewritten. Native rounded workspace usage rose
$0.05, detailed usage$0.05481666, actual paid$0. Conservative closed allowance
was$0.207944, below the$0.58 reserve. These delayed workspace deltas are not an
exact per-app invoice, and no later jobs' usage is assigned here.

The one free official check was submitted once, accepted, and observed terminal
done/passed at21:27:58.453455UTC; the parent received immediate slot release
before report polishing. Actual post-admission attention quota was18/20 checks
used,2 remaining,0/10 ranked entries. No additional Modal run, official check,
counter attempt or formal entry occurred. Remaining quota alone does not justify
another experiment.

## Receipts, review and decision

Native raw result SHA256 is
`5e6447008a1f55baff65939aa2af75521235891fad976434ac378b4dbc55395d`.
Official raw response SHA256 is
`0095903e817e17aa8b252c4310a1b6a140aee1cc3690b82a8b04c40f05283cbc`;
the reformatted published response SHA256 is
`a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb`.
No redactions were needed in the official response. Its service content ID is
an opaque payload identifier, not asserted equal to raw local source SHA256.
Exact committed local candidate bytes were verified immediately before the sole
accepted CLI upload. Raw CLI profile artifacts are copied byte-for-byte with
their own hash map; whitespace is retained.

`runtime/analyze_result.py` and`official-check/analyze.py` reproduce source and
receipt checks, case coverage and timing arithmetic. Independent native and
official reviews found no correctness, receipt or arithmetic blocker. Sanitized
source/results are suitable for the existing private repository under the
standing reviewed-push rule. The original checkout and A2-11 rollback source
remain unchanged.

## Working promotion and rollback

Use the composition as the best validated working candidate, fusion as the
direct experimental control, and A2-11 as a preserved fallback. Leaving A2-11
as the working performance baseline solely because official intervals overlap
would discard direct paired evidence. The recommendation uses the following
chain of evidence rather than treating three historical scores as paired data.

| Candidate | Official score and95% interval | Relevant evidence |
|---|---|---|
| A2-11 | 1.187678756× [1.169863575,1.210934250] | Preserved known fallback; job`cfa8a8670e394f80` |
| Preparation fusion | 1.225555436× [1.207172105,1.250062193] | Native three changed targets improved3.7837%/4.1058%/2.3596%, each6/6 pairs; exact preparation bytes and42 oracle checks; official16/12 pass |
| Fusion plus lookahead | 1.233633059× [1.215128563,1.252137555] | Adds4.8226% native paired target cut vs fusion,0.1532% unchanged-control slowdown;32 native oracles and official16/12 pass |

The composition's aggregate point estimate exceeds A2-11 by3.8693%. With each
path normalized by its own official reference, the composition's full-B32-
decode latency is15.8144% lower than A2-11. Full-B16-decode, full-B16-verify
and full-B32-verify are respectively3.2121%,3.6685% and1.9599% lower. Across
all twelve paths, the largest normalized regression versus A2-11 is1.8818%
(sliding verify B1); its source route was not changed by this composition.
`official-check/analysis.json` retains every row and historical receipt.

Fusion is the right control for attributing the incremental composition change.
The native4.8226% target improvement is paired within a session, reproduced the
direction of historical A2-15 on fresh inputs, and preserved the fusion control.
The complete official check independently accepts correctness and shows the
same target direction without a material unchanged-path regression. This is
sufficient to choose the next working baseline while acknowledging that the
small incremental aggregate gain is not precisely established. Fusion remains
a useful simpler fallback if later serving validation finds a lookahead hazard.

Remaining risks are limited coverage of arbitrary serving states, one new
native session, cross-run official inputs/calibration, and the coupled address-
scheduling/buffering mechanism. Keeping exact rollback sources and validating
any real replay gap addresses those risks better than retaining a slower
working baseline indefinitely. After the final source/replay audit found no
concrete blocker, the parent authorized this working promotion. It is recorded
in `BEST_VALIDATED_MANIFEST.json`, `CONTINUE.md`, `CURRENT_AUTHORIZATION.md` and
this probe's `WORKING_SELECTION.json`. Projection records, original source
files and ranked submission state are unchanged.

To roll back, set the attention working source path/hash to
`reviews/attention-v2/mixed_fa3_decode_parallel_merge.py` /
`c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`,
using the preserved `attention_rollback_A2_11` record. Do not overwrite either
candidate. Fusion's exact control path/hash is also preserved in the selection
record. Selection applies to project records in this isolated branch; no
serving deployment or formal contest entry is implied.

The next useful evidence, if a promotion decision needs greater precision, is
a separately coordinated paired complete-six-load comparison of composition
against fusion. It would address aggregate uncertainty rather than repeat this
official check or sweep split knobs. No such run is started or budget reserved
by this report; a distinct candidate hypothesis is required for another check.
The immediate local robustness plan is narrower: `NEXT_VALIDATION.md` identifies
DP multi-chunk true-lookahead/merge as uncovered. The existing native ABA test
already mutates live length/page tables at fixed addresses within captured
graphs, so it is not proposed again. The DP plan preserves exact candidate
bytes and original oracles; a duplicate official check cannot add its custom
case. No credible new performance mechanism or extra official check is claimed.
