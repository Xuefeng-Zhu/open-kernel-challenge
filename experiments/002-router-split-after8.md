# 002 — SGLang-style router split above eight tokens

Hypothesis: replacing only the starter's m>8 FP32 router product with the official
SGLang BF16 hi+lo/FP32-accumulation path may remove the measured large-M router deficit.
The real baseline m=256 router was 45.53 µs against SGLang's 17.34 µs; this is motivation,
not a timing prediction for this candidate. Other projections and m<=8 retain baseline math.

Candidate: `candidates/router_split_after8.py`; SHA256 `6b1b66e2875ad4a755e98e41235243c5f0b5131c32f5f6b08e49d2f4f7d5c352`.
Control: proven baseline `fde8a042944c742c20b76f27fcc030d767fd0058` / job
`e5a5b2d20da54187`. Configuration: `ROUTER_FP32_MAX_TOKENS=8`, two products,
contiguous [K,N] hi/lo weights; no precision reduction beyond the permitted router split.
Only weight work is in prepare; fresh activation conversion/products remain timed.

```sh
.venv/bin/python check.py --submission candidates/router_split_after8.py --device cpu --suite all --output results/router-after8-cpu.json
ROUTER_CANDIDATE=candidates/router_split_after8.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_split_after8.py
```

Actual results: **39/39 full-size CPU oracle cases pass; 3/3 focused tests pass; static
checks pass**. Maximum router relative L2 in the full suite: **2.729e-6** (limit 1e-4).
Focused coverage includes m=1/2/7/8/9/16/32/128/256, outlier channels, fresh activations,
weight-column probes, zero rows, a BF16-only negative control, original-weight/input
preservation and other-projection slice semantics.

Environment: Apple M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1; full suite uses four
CPU threads. CPU explicitly models FP32 products of BF16-rounded operands. CUDA uses
`torch.mm(..., out_dtype=torch.float32)` as the official baseline does, but **the CUDA
path has not been compiled/executed and H200 correctness/performance are unverified**.
No GPU run ID exists for this experiment; no upload/check was made. Evidence:
`results/router-after8-cpu.json`. No CPU timing is used to rank GPU performance.

The commit containing this candidate and record is its local source/result snapshot;
the final `experiments/index.json` links the exact commit. This reproduces a known
SGLang technique rather than claiming a novel kernel or measured speedup.
