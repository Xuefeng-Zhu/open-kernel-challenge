# Coordinator checkpoint — 2026-10-24 08:44 UTC

Authorization: up to **10 NEW** meaningful off-board dense-projection experiments,
$0 spend, prior baseline excluded; no scored/final entry, new credentials or material
agreement. **5 admitted, 3 done, 2 queued, 5 allowances remaining, 0 uncertain**.

| ID | Job | State | Official speedup vs SGLang |
|---|---|---|---|
| R01 control | 59abe5825690427c | done; correctness passes | 1.0012943× [0.9862749,1.0163137], parity |
| R02 wide GEMM | 89fe00c5034a4198 | done; correctness passes | 1.0476946× [1.0319792,1.0634100] |
| R03 dispatch | 718e0730e1bd4c30 | done; correctness passes | 1.0267804× [1.0113787,1.0421821] |
| R04 Triton router | e3bc1481851f4a23 | queued | unmeasured |
| R05 wide layout | 1631810d19af4af4 | queued | unmeasured |

Each completed run passes 9/9 prerequisites and 30/30 scored-shape checks. Current
best completed source is `candidates/router_fused_wide.py` (R02); normalized ratio to
our original 0.9076064× starter is 1.1543491×, a cross-run reference-normalized comparison.
Original `submission.py` and all pending source files remain unchanged.

Account quota: **6/20 checks used, 14 left; 0/10 scored used** (includes prior baseline).
This is distinct from the new authorization's 5/10 usage. No paid resource or charge.
No tool/approval blockage; bounded status reads return normally. The only pending
prerequisite for the two admitted jobs is the organizer's shared H200 queue.

See `CONTINUE.md`, `experiments/remote-budget.json`, `REMOTE_RESULTS.md` and `BEST.md`.
All admitted requests must be followed to terminal; do not replace or resubmit queued IDs.
