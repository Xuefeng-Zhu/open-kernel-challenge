# A2-03 — official FA3 for full decode batch16 only

Parent is correct A2-02 `mixed_fa3_all_verify.py`, source790d425/SHA25b35cc7,
job3e85261cb8ed4899, score1.0124269677×. Its total gain over A2-01 remains
inconclusive; two shape-local wins are supported. Preserve both parent versions.
Candidate `mixed_fa3_decode16.py` adds one branch for batch16/q_len1/window0/
rope_theta0. Existing full FA3 adapter is unchanged; every other11scored shape
retains its parent dispatch/helpers. No precision change beyond the already
explicit official-backend behavior on this newly routed path.

Problem: full decode16 remains257.392µs vs same-receipt SGLang200.384µs.
Hypothesis: existing full FA3 pipeline reduces normalized latency at least20%.
It must consume supplied full-decode scheduler_metadata, cache_seqlens and
cu_seqlens_q; no host tensor-value inspection, schedule regeneration/caching,
window truncation or sparse attention. Full has no RoPE; causal full-page FA3,
FP8 normalized Q/cache with scales, row0-live writeback and BF16 output before
RMSNorm/gate remain the official contract. All past keys/values remain attended.

20%-cut and matching-reference score scenarios are in A2-03-cpu-preflight.json;
they are arithmetic with other times frozen, not achieved predictions. Reject
target goal if native correctness fails or target cut<20%; inspect controls>5%
without automatically treating cross-run shifts as a source effect. Aggregate
CI must support any claim of total improvement over parent or SGLang.

Four local CPU stand-in tests pass:

```
.venv/bin/python reviews/attention-v2/test_mixed_fa3_decode16.py
```

Coverage: all helper ASTs unchanged;180nearby dispatch configurations;
ctx0/64/2048 and coherent attention; page64/nonunit scale; live cache0 and untouched
rows; same-address A→B→A with changed actual lengths/content. An opaque int32
scheduler sentinel is forwarded by identity and immutable, including changed
sentinel contents during ABA. This tests our arguments only, NOT native FA3
scheduler interpretation. CPU stand-ins do not execute native kernels/graphs.
New native target page64/nonunit/explicit ABA/untouched-row/concurrency coverage
is unverified unless the official receipt provides it. Scored full decode16
uses actual scheduler metadata and agentic spread; prerequisites do not enter
exactly this newly routed batch16/L1 full path.

After source commit and free shared GPU slot:

```
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-03 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-03 --cycle2
```

Before request: attention8checks left; projection5after terminal P040. No Modal,
paid compute, new credentials/terms or ranked entry. Automatic reviewed private
pushes permitted, with GPU-unverified label until actual terminal receipt.
Every accepted/uncertain request counts, never blindly repeats.
