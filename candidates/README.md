# Isolated candidates and actual validation

The original `../submission.py` remains unchanged: baseline H200 check passed at
0.9076064× SGLang. Current source/result commits, SHA256s, configurations and CPU
reports are in `../experiments/index.json`; all actual H200 results are in
`../REMOTE_RESULTS.md`. The latest combined Mac regression suite passes 40/40 tests.
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
| `pooler_triton_split8.py` | Local-only BF16 pooler; planned R06 denied before execution |
| `router_hybrid_library.py` | Local-only composition of R03 small-M and R02 large-M paths |
| `router_triton_split32.py` | Local-only split-count alternative motivated by R04 profile |

Every local candidate passed 39 full-size CPU contract cases and focused boundary,
probe/negative tests recorded separately. There is no CPU speed ranking. R05 has
the highest observed overall point score but is not statistically resolved above
R02/R04; preserve their sources and per-shape evidence when choosing future variants.
A native R04 + wide R02 composition has not yet been implemented or tested.

Five new checks were admitted and all are terminal; five allowances remain at $0.
Further source uploads are paused by automatic approval review, see `../DENIED_R06.md`.
Do not retry denied R06 or upload another candidate as a workaround. No scored or
public/final entry, paid rental, new credential or material agreement is authorized.
