# Independent native-result review

The native gate passes and the paired whole-call evidence supports one complete
official **check**, after resource closure and parent slot authorization. It
does not yet justify adopting the candidate or claiming an official score.

The reviewed invocation is app `ap-I6PNHOGgL6uRL1AjjG2yoV`, frozen commit
`0280972637e600e5f82a45f35778aed3f72886b1`. The nine recorded payload/controller
hashes match committed Git bytes. The result transport hash, reported worker
hash, both source hashes and official helper/reference/check hashes agree with
the frozen payload.

| Evidence | SHA256 |
|---|---|
| `result.json` | `576d8e09e41b344d6c262c5bbdeb830e5800c6e3a1e19ed6f1aa8cb266e06868` |
| Candidate | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Unchanged A2-13 control | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| Executed worker | `b4b47a4729fb06b85995c115cb91492bc6d552f5a809653359e38a67cd35f08f` |

The result reports complete execution in 70.488964435 seconds on NVIDIA H200,
SM90, Torch2.14.1+cu130, CUDA13.0, Triton3.8.0 and Python3.12.3. This is a
native GPU result, distinct from the preceding CPU host models.

All twelve preparation groups—two live input versions for each of six
predeclared cases—pass. Their 36 Q/K/V comparisons have zero unequal bytes,
finite queries and intact immutable inputs/untouched cache rows. All six cases
have both complete entrypoints captured separately. All 36 A→B→A official
oracle checks pass, including stable addresses and intact non-new cache rows;
all twelve return-A checks reproduce output and new-cache bytes exactly.
All six additional independent-cache, separate-stream official oracle checks
pass. This demonstrates safe concurrent submissions for these cases, not
measured hardware overlap. Decode uses real supplied metadata of shape49 for
both input versions; verify metadata remains None. Every A/B length vector
differs. All preparation/replay phases precede every timing phase.

Each target retains four warmup and twelve measured two-arm samples, with
balanced adjacent reverse-order pairs sharing one input version. All 96
warmup/measured replay checks pass against the same-input oracle-validated
bytes. Independent recomputation agrees with the retained summaries:

| Full-attention target | Raw median control → candidate (µs) | Reduction from median paired ratio | Positive balanced pairs |
|---|---:|---:|---:|
| B16 decode, L1 | 264.224 → 255.824 | 3.7837% | 6/6 |
| B16 verify, L8 | 211.200 → 202.704 | 4.1058% | 6/6 |
| B32 verify, L8 | 397.152 → 388.448 | 2.3596% | 6/6 |

These are complete captured-call device timings with preparation, cache stores,
FA3 and output gate included. Restores, L2 eviction, validation and profiler
work are outside the timing interval; no profiler was used. Separate empty
graph calibrations were retained, and raw paired timing remains primary.

Decode B16 has markedly different times between the two fixtures: control
medians are approximately322.608 and207.648µs. Its pooled264.224µs median is
not representative of either individual fixture. The paired ratios preserve
equivalent input versions and support the reported relative reduction; do not
transfer the pooled absolute time into official scoring. All eighteen balanced
pair ratios favor the candidate. One session with repeated pairs supplies no
independent-session confidence interval, and the effect remains modest.

The analyzer's ratio/reduction arithmetic, sample order, version balancing and
gate checks agree with the raw result. The independent review additionally
checks exact case/preparation-version coverage and support-source hashes.
Those checks should remain explicit in any published standalone analyzer.

Coverage includes page1/scale1 scored families and page64/scale1/64 boundary
families for all three changed routes. It is not their full cross-product or
the complete sixteen-prerequisite/twelve-scored official suite. A complete
official check can now answer that coverage and calibration question. An
official check is separate from formal contest entry or candidate adoption.

The worker exited0 and reported complete/pass. The initial controller ledger
recorded an app still stopping with one task, causing its closure-sensitive
exit status to fail. That observation is not a kernel or correctness failure
and should be preserved. Root owns independent terminal app/tasks-zero
reconciliation and final cost accounting; this review neither controls
resources nor infers closure from the initial snapshot.

No GPU, network, browser, relaunch or official action was taken during this
independent result review. Selected A2-11 remains unchanged.
