# Selected Projection source

Use [P046](projection_pooler_column_batch2_hybrid.py), SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Latest ranked same-source result is 1.1897129769×; [BEST.md](../BEST.md) contains
receipts, rollback and qualifications. Other candidates below are historical.
P070 Quack SERIAL/SEPARATE configurations were rejected after native testing.

## Historical candidate catalog

Latest source projection_qkv_m128_singlepass.py (035/M36) passed fullCUDA and
nearby/layout correctness but target~2.5xslower; rejected/noofficialcheck. Best
projection_full_composition.py unchanged.112localCPUtests pass,all11apps stopped.
See ../QKV_M128_SINGLEPASS_RESULTS.md and STATUS.md for currentevidence/budgets.

# Current cycle-two candidates

Best official source projection_full_composition.py (c34c497b), job48b1357016134e52,
1.1486499523x [1.1314202031,1.1658797016],9+30gatespass. Baseline preserved.
Follow-up: projection_qkv_layout.py rejected after CUDA pass/m256regression;
projection_unpooler_layout.py rejected after CUDA+nearby pass/no aggregate gain;
projection_pooler_grouped.py and projection_pooler_narrow.py both CUDA-pass but
severely slower; rejected. Allapps stopped.
104localCPUtests pass; CPU branch models do not execute Triton. No push/new scored
entry. See ../STATUS.md and experiment025–030 records for current scope/budgets.

The following first-cycle catalog is historical; its57test/budget/R10-best statements
refer to that completed stage, superseded above and by current explicit ledgers.

# Isolated candidates and actual validation

The original `../submission.py` remains unchanged: baseline H200 check passed at
0.9076064× SGLang. Current source/result commits, SHA256s, configurations and CPU
reports are in `../experiments/index.json`; all actual H200 results are in
`../REMOTE_RESULTS.md`. The latest combined Mac regression suite passes 57/57 tests.
CPU branches explicitly model BF16 products with FP32 accumulation and do not test
CUDA `out_dtype`, Triton compilation or graph capture.

| Candidate | Remote evidence / status |
|---|---|
| `router_split_after8.py` | R01 passed, 1.0012943×: faithful control, parity |
| `router_split_after1.py` | R03 passed, 1.0267804×; m8 router improved |
| `router_fused_wide.py` | R02 passed, 1.0476946×; strongest measured large-M router component |
| `router_triton_split16.py` | R04 passed, 1.0459769×; native low-M router compilation/correctness verified |
| `router_fused_view.py` | R05 passed 1.0509508×; larger-router regression versus R02, overlapping score intervals |
| `router_transpose_view.py` | Local-only two-product layout control; no GPU validation |
| `pooler_triton_split8.py` | R06 passes correctness but regresses low-M pooler; do not compose |
| `router_hybrid_library.py` | Local-only composition of R03 small-M and R02 large-M paths |
| `router_hybrid_native.py` | R07 passes 1.0991970×; native small-M + wide large-M composition |
| `router_triton_split32.py` | R08 passes 1.0617021× with R01 large-M control; m1/8 router 7.30/6.87µs |
| `qkv_triton_split8.py` | CPU-only; deprioritized after related R06 pooler regression |
| `pooler_simt_row.py` | R09 passes correctness; whole workload 1.0076197× (parity), target m1 pooler 18.30µs vs R01 21.66µs |
| `router_pooler_hybrid.py` | R10 passes, best observed 1.1207320×; no allowance remains |

Every local candidate passed 39 full-size CPU contract cases and focused boundary,
probe/negative tests recorded separately. There is no CPU speed ranking. R10 is the highest observed passing point score; its interval overlaps R07's.
Its m8 router is slower than isolated R08, and some unchanged loads vary across
runs. Preserve those qualifications and all controls. No held-out/nearby-shape GPU
validation or final leaderboard rank is claimed. Source docstrings preserve their
creation-time context; current validation is authoritative in the index/remote records.

Current admissions and allowance are in `../experiments/remote-budget.json`; at most
two active jobs and ten new checks, $0. The historical R06 denial was resolved by
specific new human approval and one accepted retry; see `../DENIED_R06.md`. No
scored/public/final entry, paid rental, new credential or material agreement is authorized.
