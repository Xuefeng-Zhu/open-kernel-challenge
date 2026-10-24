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
| `router_fused_view.py` | R05 passed, highest observed aggregate 1.0509508×; larger-router regression versus R02, overlapping score intervals |
| `router_transpose_view.py` | Local-only two-product layout control; no GPU validation |
| `pooler_triton_split8.py` | R06 passes correctness but regresses low-M pooler; do not compose |
| `router_hybrid_library.py` | Local-only composition of R03 small-M and R02 large-M paths |
| `router_hybrid_native.py` | R07 passes, current best 1.0991970×; native small-M + wide large-M composition |
| `router_triton_split32.py` | R08 passes 1.0617021× with R01 large-M control; m1/8 router 7.30/6.87µs |
| `qkv_triton_split8.py` | CPU-only; deprioritized after related R06 pooler regression |
| `pooler_simt_row.py` | R09 passes correctness; whole workload 1.0076197× (parity), target m1 pooler 18.30µs vs R01 21.66µs |
| `router_pooler_hybrid.py` | R10 final measured-component composition admitted; result pending, no allowance remains |

Every local candidate passed 39 full-size CPU contract cases and focused boundary,
probe/negative tests recorded separately. There is no CPU speed ranking. R07 is now the highest verified observed score. Its interval lies above the old R05
interval, but it remains one trial and nearby/held-out shapes are not GPU-verified.
R08 independently validates split32 on low-M router; the final R10 composition has been admitted and awaits its own GPU result. Preserve source hashes and all per-shape/failed evidence.

Current admissions and allowance are in `../experiments/remote-budget.json`; at most
two active jobs and ten new checks, $0. The historical R06 denial was resolved by
specific new human approval and one accepted retry; see `../DENIED_R06.md`. No
scored/public/final entry, paid rental, new credential or material agreement is authorized.
