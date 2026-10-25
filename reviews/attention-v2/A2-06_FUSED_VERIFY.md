# A2-06 — extend the proven prologue to sliding verify batches

Parent A2-05 source535d602, job50465a4c440f4b75,1.1157987383x. The one new
mechanism passed native B1/L8 and reduced normalized target latency28.178%.
This experiment removes only the batch==1 restriction. Every helper's AST,
Triton launch geometry, arithmetic/precision and FA3 arguments remain unchanged.
All sliding L8 batches now use that same prologue; all other paths stay intact.

Scored targets are B16/B32 sliding verify. Native parent times29.246/34.575us;
profiles still show the separate prologue calls that fusion removed at B1.
Larger batch resources may offset saved launches, so the predeclared goal is at
least10% normalized latency reduction on EACH target, with no material regression
in ten unchanged controls. Aggregate10% scenario is saved in A2-06_MODELS.json;
it may be inside the score CI. Do not claim that all total changes are causal.
Reject performance promotion if targets miss their goal; retain full evidence.

The generic dispatch also exercises native prerequisite B5mixed, B3nonunit-scale,
B4soft sliding verify through the fused path. Official cases use page1 there;
page64 and explicit same-address A-B-A GPU stress remain unverified. Complete
CPU-entrypoint and component tests cover page64/scales/live0/window edge/batches/
DPheads/changed-length replay. Six methods pass, but all launches are CPU stubs.
Original B1 candidate and native receipts remain immutable. No third-party code
execution, harness tricks, precision lowering, Modal or ranked entry.

```
.venv/bin/python -B reviews/attention-v2/test_fused_verify_dispatch.py
.venv/bin/python -B reviews/attention-v2/test_fused_prepare_batch_models.py
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-06 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-06 --cycle2
.venv/bin/python reviews/attention-v2/assess_cycle2.py A2-06
```

Source SHA and full CPU logs: A2-06-cpu-preflight.json. One request after exact
source commit, serialized behind completed A2-05; failures count. Leave one final
reserve, obey real daily limit. Reviewed automatic private push is authorized.
