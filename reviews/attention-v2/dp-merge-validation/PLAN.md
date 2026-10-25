# DP multi-chunk merge: bounded correctness validation

This experiment validates an actual uncovered portion of the working
fusion-plus-lookahead candidate's admitted dispatch scope. It keeps the
candidate and fusion control byte-identical. It measures correctness rather
than latency and does not consume another official check or change selection.

The parent granted one serialized native validation after Projection P045
became terminal. Existing project authority covers this free-credit experiment,
local implementation/review and sanitized private export. Actual paid authority
remains $0. No credential renewal, installation, counter collection, formal
entry or billing/visibility change is permitted or needed.

## Gap and fixed cases

The current true-lookahead guard admits full B32/L1 without RoPE for both TP8
and DP heads. Existing DP native coverage uses ctx65 and one split; the official
B32 scored case uses TP8 and no prerequisite uses B32. Existing A→B→A tests
already replace live lengths/page tables in retained captured graphs at fixed
addresses. Repeating that generic replay question would not close the DP
multi-chunk/merge gap.

Two cases cover the gap without a sweep. DP means 128 query and 16 KV heads,
D64, batch32, query length1, full attention, no RoPE.

| Case | Capacity | Page, scale, distribution | Distinct coverage |
|---|---:|---|---|
| DP two chunks | ctx8193 | page64, scale1/64, soft | SPLITS2, mergeBS2; empty history, page/tile/8192-chunk boundaries |
| DP three chunks | ctx16385 | page1, scale1, coherent | SPLITS3, mergeBS4; non-power-of-two split reduction and masked merge padding |

The second case extends the preliminary two-ctx8193 plan for a concrete reason:
source audit shows the merge pads split count to a power of two. SPLITS3 tests
that branch while retaining two total cases. This is not a parameter/performance
sweep, and differences between cases are not interpreted as causal speedups.

Use fresh seeds135721/135722. The first history vector is
`[0,1,63,64,65,8191,8192,8193]×4`; the second is
`[0,1,63,64,65,127,128,129,8191,8192,8193,16320,16321,16383,16384,16385]×2`.
Fixture B reverses the vector and uses fresh values/mappings. Retain physical
page0 as a live new-token slot and negative unused suffixes. Mask suffixes with
`ceil((history+q_len)/page_size)`, not the previous page64-specific divisor.
Refresh actual supplied FA3 scheduler metadata outside capture, even though
the custom path may ignore it. Cache shape bounds, not live lengths, fix splits.

## Source and correctness gates

| Artifact | SHA256 |
|---|---|
| Working candidate | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Fusion control | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Existing graph/restore helpers | `9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e` |
| Original reference | `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b` |
| Original checks | `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055` |

Both complete entrypoints must pass the unmodified original oracle on A→B→A
for both cases: twelve oracle calls, four exact returning-A comparisons and
twelve whole-cache K/V cross-arm byte comparisons. Validate immutable inputs,
untouched cache rows, finite output/new-KV, exact fixed input/output addresses,
changed A/B lengths and mappings, legal new-KV ownership and actual shape-derived
SPLITS2/3 plus separate-merge configuration. No tolerances or algorithm change.

On one declared case, create two independent candidate graph captures with
disjoint writable cache and output ranges. First establish the second graph's
own A expectation with one original-oracle replay. Submit the graphs on
separate streams with different fresh fixture versions, then check both against
the original oracle: two concurrent oracle calls, fifteen total including the
setup. The second graph now owns both A and B expectations without transferring
expected bytes across graphs. Use the retained graph
outputs before and after two further isolated replays to check four independent
output/cache-isolation states, changing the first graph to B and the second to
A. This tests reuse of captured per-call workspace
without accessing internal scratch pointers or patching the candidate. Record
pool identities only if the installed documented API exposes them; missing pool
IDs do not imply shared scratch or failure. Source audit separately verifies
fresh per-call FP32 partial allocation and decode's counter-free merge route.

Independent stream submission is not proof of simultaneous hardware execution.
No profiler is introduced to prove overlap, and same-graph overlapping replay
is outside this test. These are bounded serving robustness observations, not
universal concurrency certification or a new official score.

## Admission, bounds and stop criteria

Freeze the independently reviewed new worker/controller and exact allowlisted
source/support files in Git before launch. The default controller command is
offline preflight only. Launch once using the existing protected credential
through supported configuration and fixed existing image
`im-cPTbGpNiX3RarKResPgX7h`, H200, two CPUs and8GiB host memory, one container,
one invocation, zero retries. Worker160s, child170s, function185s, startup20s
and watchdog210s bounds are retained from the reviewed successful harness.

Fresh read-only billing and supported CLI state/rates must establish paid cap
$0, actual paid$0, sufficient remaining free credits and no active Modal app.
Reserve at most$0.58; admit only if doubled current resource rates over the full
watchdog plus$0.02 fit within it. The old artificial stage ceiling is removed.
Credentials and local billing/slot receipts never enter the payload or export.

Stop on any source/hash, guardrail, case/shape, oracle/cache/replay/isolation
failure, access denial, timeout or unexpected app. Retrieve failed/partial
results unchanged; check uncertain outcomes before any further action. No
automatic retry or substitute source. Stop the app and independently verify
stopped/tasks0 before releasing the serialized slot. Preserve initial stopping
observations and reconcile them separately. Record delayed workspace credit
delta with its cutoff and avoid assigning later jobs' usage to this stage.

Only correctness evidence can follow from this experiment. A failure should
identify the exact state/branch before proposing a fix. A pass closes this
specific DP multi-chunk gap and strengthens the working selection; it does not
justify a duplicate official check. A separate local analysis will seek the
next whole-score-relevant algorithm mechanism from actual profiles and source.
