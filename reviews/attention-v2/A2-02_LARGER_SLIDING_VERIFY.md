# A2-02 — same sliding FA3 path for verify B16/B32

Parent `mixed_fa3_verify1.py`, source0309241/SHA2c7bd6a0,
official6e761bdd1d9747ad, score1.0002799616×. Candidate
`mixed_fa3_all_verify.py` adds only16/32 to the existing B1 sliding verify
batch-membership condition. Every helper body, full path and decode dispatch
remains AST-identical. Other10scored shapes are unchanged controls.

Hypothesis: the already-correct official sliding FA3 adapter reduces each new
target's reference-normalized latency at least15%. Parent C/S medians are
37.247/29.328µs (B16),44.883/34.416µs (B32). These measurements justify the
backend extension; no new tile/math/precision tuning is included.
The same official BF16 norm-before-RoPE, BF16 RoPE output→FP8 Q, FP8 scaled KV
writeback, supplied positions/lengths/page table, causal window=(2047,0), and
BF16 attention output before RMSNorm/gate now apply to these two paths. Relative
to their E2 fallback this has explicit allowed rounding differences and needs
native oracle validation. No third-party competitor source is executed.

Holding other parent times fixed,15%cuts on both targets model1.0155625×,
matching same-receipt SGLang models1.0231544×. Scenarios, not results. Such narrow
aggregate changes may have overlapping CIs; do not claim a resolved total gain
from target wins alone. Reject target goal if either cut<15% or correctness fails;
investigate any>5%control regression without assuming source causality.

## Local verification before native check

```
.venv/bin/python reviews/attention-v2/test_mixed_fa3_all_verify.py
.venv/bin/python reviews/attention-v2/test_mixed_fa3_sliding.py
```

Four new+five parent CPU tests pass. The CPU stand-in helper now supports batch
loops, preserving exact before-RoPE BF16 rounding. Its initial table-only test
fixture omitted `batch` after generalization and raised one KeyError; fixed by
adding batch1 to the fixture, without candidate changes. First new suite4passed,
then final combined9passed. This is recorded, not a GPU/candidate failure.

AST/control dispatch tests cover144nearby configurations. CPU output/new-cache
oracle coversB16page1/unit andB32page64/1/64scales, live cache0, immutable inputs,
untouched cache rows, fresh same-address A→B→A with changing actual lengths and
contents. E6 boundary tests remain passing. CPU table extent8192 is a testing
bound; candidate keeps the official2^21config-only table, reused from existing
B1sliding cache. No input-derived cache. CPU stand-ins are not native FA3/CUDA.

Official prerequisites do not enter these exact B16/32sliding verify paths;
scored nativepage1/unit correctness and fresh-input graph timing are required.
Explicit target nativepage64/nonunit/ABA/untouched-row/concurrency coverage remains
unverified unless receipt supplies it. E2 decode ticket limitation remains.

Record full hash/tests/models in A2-02-cpu-preflight.json. Commit exact source
before dispatch, then:

```
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-02 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-02 --cycle2
```

New attention9checks left before request. Projection P039 is terminal before
native dispatch; one shared GPU slot, no concurrent measurements. No Modal,
paid compute, new credentials/terms or ranked entry. Automatic reviewed private
pushes are now explicitly authorized; source is labeled GPU-unverified until
the receipt exists. Failures count; never duplicate uncertain POST.


## Accepted — native outcome pending

Job3e85261cb8ed4899 accepted once2026-10-25T02:50:05.688335Z.
Exact source790d4256fe311086eb9f899ae88b0915ce0eb5ab,
SHA25b35cc748262774a85c3a8ed584ed02b37dbb429b967d437c9372e7afe9037d.
New attention2/10used,8left. Projection P039 was terminal first; projection6left.
No native correctness/performance claimed until this job's terminal receipt.
Monitor same ID, never resubmit an uncertain request. Reviewed source/acceptance
may auto-push with GPU-pending label under explicit standing authorization.
