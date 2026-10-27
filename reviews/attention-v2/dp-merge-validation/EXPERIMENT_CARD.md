# DP merge robustness experiment card

**Evidence.** Working source `3da22a…0a7cd8` admits B32/L1/full/no-RoPE DP
heads. Its only saved true-lookahead DP native case has ctx65/SPLITS1; official
scored B32 is TP8 and prerequisites contain no B32. Source-derived multi-chunk
partial/merge behavior is therefore a real native coverage gap. Fixed-address
live metadata A→B→A replay is already tested and is not a newly claimed gap.

**Single question.** Does the unchanged candidate preserve the original output,
new-KV, untouched-row and graph-isolation contracts when DP has two or three
chunks, including a padded non-power-of-two merge reduction?

**Isolated change.** Only a new correctness harness and unseen fixtures. No
candidate algorithm, dispatch, precision, split policy or official tolerance
change. Two declared cases: page64/scaled/ctx8193/SPLITS2 and
page1/unit/coherent/ctx16385/SPLITS3, fresh seeds135721/135722.

**Optimistic latency and weighted-score upper bound.** Zero direct gain: this
does not change timed candidate work. It can raise deployment confidence or
find a precise defect. No performance success claim follows from a pass.

**Costs and resource risks.** One existing-image H200 invocation, no profiling,
zero retries,160/170/185/20/210-second worker/child/function/startup/watchdog
bounds, at most$0.58 free-credit reservation and actual paid$0. DP increases
cache/oracle work relative to TP8. Source-derived FP32 partial storage is
4,325,376 bytes for SPLITS2 and6,488,064 for SPLITS3; GPU workspace and fresh
cache fixtures are not reused across independent graph captures. Compile/oracle
time and memory pressure remain possible failure modes, not measured latency.

**Control design.** Exact selected composition and exact fusion control share
the same two fixture versions and geometry within each case. Both use the
original oracle independently. Entire K/V bytes are compared across arms; output
bytes are compared to each graph's own previously oracle-checked state. No
cross-case timing comparison or transfer of expected bytes between graphs.

**Cheap falsifier.** Local source/hash/split/address proofs, page-size-aware
mask checks and focused mocked worker/controller tests run before compute.
Independent review caught a missing own-A expectation in the second graph;
the frozen protocol adds one explicit original-oracle setup replay before any
GPU run rather than relaxing the isolation gate.

**Predeclared correctness and acceptance.** Exactly15 original oracle calls
(12 A/B/A +1 own-A setup +2 concurrent-submission checks), four exact return-A,
twelve K/V byte comparisons and four post-submission isolation checks after two
additional candidate replays must pass. Stable addresses, legal/masked cache
accesses, immutable inputs, unchanged cache rows, finite output and exact
candidate/support hashes are mandatory. Both independent graphs mutate their
retained state. Actual hardware overlap and direct internal scratch-pointer
observation are explicitly not claimed.

**Decision.** Any failure rejects this coverage claim and stops the branch for
diagnosis. A pass closes only this bounded gap and retains the current working
choice; it does not consume an official check or prove universal concurrency.
Stop and independently verify app stopped/tasks0, release the serialized slot
and record a cutoff-qualified cost before the next measurement. No posthoc
threshold, unchanged-source official rerun or deduplication bypass.
