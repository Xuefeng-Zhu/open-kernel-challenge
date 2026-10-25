# Independent DP native result and receipt review

**PASS.** The exact selected source passed the two targeted DP multi-chunk
cases and independent graph reuse/isolation checks. No result, source-binding,
receipt, accounting or selection-record blocker was found. This reviewer used
saved local files and offline analysis only; no service, GPU, browser, credential
or network action was performed. Only the new offline analyzer, its derived
JSON and this review were written; frozen files were not edited.

## Source binding and raw evidence

Native app `ap-gXUKhAM11JBfvrA3h5rip9` ran source freeze
`8d38550d061af14df3257afe0d438dbf0fc82f5f`. All nine receipt payload/controller
hashes were checked against both current files and frozen Git bytes, including
worker `4507a19cdaea450415b6d7a40315e9e0021741919f5a33842fed290e0f2af59d`
and controller
`0299612f835e68b26ece31f5d5d031c959953988ce4d998d95941ddd9d778f17`.
The candidate remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
with exact fusion control
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
Original oracle/check/helper hashes match their reviewed values. No candidate,
precision, dispatch, tolerance or FA3 argument was changed by this experiment.

The raw native result SHA256 is
`3640d3df3c6d31deff277c98d04e1bdc4d685a45d1d5ee35333932301b7cae47`;
it matches the controller transport receipt. Published `launch-receipt.json`
is byte-identical to the retained local original ledger, including initial
stopping/tasks1 rather than a rewritten successful closure. Worker exit0,
status passed and correctness_complete are independently consistent with the
records. The source-bound worker wall23.42475984s is internal elapsed execution,
not a measured kernel latency or performance result.

## Case geometry and correctness reconstruction

Both fixed cases are full B32/L1, DP128 query/16 KV heads, D64, no-RoPE, with
fresh seeds135721/135722. Exact case fields and both declared reversed length
vectors match the raw report. Actual supplied scheduler-metadata shapes remain
fixed for graph replay. Source-derived geometry recomputes as follows:

| Case | Table columns / capacity tokens | Splits / merge BS | Private FP32 partial bytes |
|---|---|---|---:|
| page64, ctx8193, scale1/64, soft |129 /8256 |2 /2 |4,325,376 |
| page1, ctx16385, scale1, coherent-derived boundary fixture |16386 /16386 |3 /4 |6,488,064 |

The third split with BS4 covers the padded fourth reduction lane. Empty history,
page/tile/8192/16384 boundaries, last historical partials, negative unused page
suffixes and live physical page0 are present. The exact-source producer retains
new-KV ownership in the final capacity split even when its history is empty.
The geometry is source/shape-derived evidence, not an achieved-CTA trace or
occupancy measurement. The coherent fixture is correctly described as generated
then boundary-remapped, not identical to the untouched scored distribution.

`runtime/analyze_result.py` independently recomputes the full record coverage:
**15 original-oracle calls =12 ABA +1 own-A independent setup +2 after separate-
stream submissions**; four exact ABA returning-A comparisons;12 whole-cache
K/V cross-arm byte comparisons; four post-submission isolation checks and two
extra candidate replays. Every oracle record passes finite-output, immutable-
input/untouched-cache and stable-address checks under the unmodified oracle.
Every nonoracle isolation check exactly reproduces its graph's own previously
oracle-checked bytes. The initial expectation issue was repaired before compute
by adding the setup oracle; no expected bytes are transferred between graphs.
The reported counters match this independently reconstructed count exactly.

## Observable ownership and limits

The installed graph-pool handles were observed: first-case control `[1,0]`,
candidate `[2,0]`, independent candidate `[3,0]`; second-case control/candidate
`[4,0]`/`[5,0]`. The analyzer additionally computes full live output/K/V address
intervals from observed base pointers and exact byte extents, and verifies that
those intervals do not overlap across each simultaneous pair. Both captures
retain stable input/output addresses.

The independent stream pair uses different fresh fixture versions. Following
submission, primary A→B and independent B→A replays pass, and the inactive
output/new-KV/cache state remains unchanged each time. These observations
support captured workspace reuse and output/cache isolation for these cases.
The exact source also allocates fresh FP32 partials per call, with decode's
counter-free separate merge. Internal scratch addresses were not returned or
directly inspected. Pool separation and stream submission do not establish
actual simultaneous hardware execution; the raw report explicitly says overlap
was not measured. Same-graph concurrent replay and universal serving safety
remain outside this bounded result.

## Closure and accounting

Saved independent reconciliation at21:52:08.731188UTC confirms this app stopped
at21:51:32UTC, tasks0, and all listed apps stopped/tasks0. It is distinct from
the initial controller stopping observation. The parent reported immediate slot
release; this reviewer made no new live state query.

The further saved cutoff at21:53:27.435772UTC has the same app-ID set, all
stopped/tasks0 and no new apps. Billing remains actual paid$0. Decimal arithmetic
reproduces rounded workspace delta$0.03 and detailed delta$0.03730389. Initial
and final amounts match the launch and cutoff billing objects. Using the
recorded live rate and `ceil(client_wall_seconds+5)*rate*2+0.02` reconstructs
conservative allowance$0.1244133333333333333333333334, below reserve$0.58.
The report properly treats delayed workspace usage as a cutoff-bound observation,
not an exact per-app invoice or evidence that initial zero delta meant free
execution. No later job is attributed to this stage.

## Report and working records

`RESULTS.md` accurately separates correctness from performance and retains all
coverage/concurrency limitations. The working candidate source/hash, official
score1.2336330588627955, job and interval are unchanged. The root manifest adds
this DP validation with the exact app/freeze/counts and raw result hash; its
Projection record and A2-11 rollback object are identical to the preparation
freeze. New continuation/authorization headers describe the same bounded result.
The older fusion experiment selection snapshot remains historical rather than
being rewritten. No extra official check or formal contest entry followed.

The next conditional-alpha-rescale algorithm is explicitly prospective and
separate. This review does not establish its mathematical, generated-code or
performance feasibility, and the current DP result does not admit another run.

## Reviewed hashes

Paths are relative to this validation unless prefixed `repo:`.

| Artifact | SHA256 |
|---|---|
| `RESULTS.md` | `b97836e523f740fb4b29d46869caf8797d039441f0580e86fd03df0ee4badd34` |
| `runtime/analyze_result.py` | `67609894c94afde186c0cd02d4e9b59106fb498852eca91479fbdb3ef92863d4` |
| `runtime/analysis.json` | `d8f1f41470ee3dd5eaafd3f6ee3922b3bbb9bcd92c02ded9c03a20b5bf845cd8` |
| `runtime/result.json` | `3640d3df3c6d31deff277c98d04e1bdc4d685a45d1d5ee35333932301b7cae47` |
| `runtime/launch-receipt.json` | `1839f535637795253f25923b74748f989714e0dada44b8e63c986f9a16e9f7d9` |
| `runtime/reconciliation.json` | `0a0cf28eebae9a2f89157d57bd704ea543c2ac3954366009d492c60e18340f92` |
| `runtime/COSTS.json` | `aabf0ce997556c7b6435fb18b125f964e49c3dd5691535f2560b2c46f33564ba` |
| `runtime/accounting-cutoff.json` | `ec5a39c22726e88fa68d0b2cbf036d3ab42b4c410facf37e715d571d96999192` |
| `repo:BEST_VALIDATED_MANIFEST.json` | `919f6ed459f6f00c13fab861b3ca9cf09d95dcb13a123b07bcc45cdb745975f9` |
| `repo:CONTINUE.md` | `8b4ee43d230bce65b27f9036f179f80c766fae2e99f8507bef5209168d69b66d` |
| `repo:CURRENT_AUTHORIZATION.md` | `3974a74fd74d8a52fc8804ef60073648865d831e4e183feba78910bf2c26953a` |

The offline analyzer verifies these actual artifacts; it does not reexecute the
GPU oracle or independently measure hardware events. No publishing blocker was
found in the reviewed result set. Private export verification remains a later
parent action, not an action performed by this reviewer.
