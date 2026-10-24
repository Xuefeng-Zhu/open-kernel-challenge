# 004 — fuse hi/lo products into one wide-N GEMM

Hypothesis: one [M,K] x [K,2N] BF16 GEMM with FP32 outputs may reduce router launch
cost and change useful GEMM tiling compared with the two products in experiment 002.
Prepare packs the permitted hi/lo BF16 components into output columns; the timed call
adds the FP32 output halves. No input duplication or activation work happens in prepare.
Other projections and the <=8-token FP32 branch are unchanged. This is one fusion
hypothesis; its packing is required for that fused product, not a collection of tuning changes.

Candidate: `candidates/router_fused_wide.py`; SHA256 `016dbc6828bbf85dd6479a119b0b6f4b06cdb13fdc486fa1df16a09139537ccd`.
Control: experiment 002 / `candidates/router_split_after8.py`. Configuration: threshold
8, one GEMM with N doubled from 256 to 512 on official router shapes, then FP32 add.
Original router weights are retained; only the explicitly allowed hi+lo representation
is additionally stored. There is no lower-precision approximation or cached output.

```sh
.venv/bin/python check.py --submission candidates/router_fused_wide.py --device cpu --suite all --output results/router-fused-wide-cpu.json
ROUTER_CANDIDATE=candidates/router_fused_wide.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_fused_wide.py
```

Actual result: **39/39 full-size CPU oracle cases and 3/3 focused tests pass; static
checks pass**. Maximum router relative L2: **2.729e-06**, below 1e-4. Tests include
dispatch edges, outliers, probes, zero rows, a BF16-only negative control, fresh input,
original weights and non-router output slices. Evidence: `results/router-fused-wide-cpu.json`.

Environment: Apple M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1, four CPU threads
for the full suite. CPU explicitly emulates the BF16 operands using FP32 multiplication.
**CUDA compilation/execution, H200 correctness/capture and performance are unverified.**
No GPU job ID, upload, check or paid resource exists for this experiment.

Risk: the wider GEMM may select an unfavorable tactic, and separate output-half sum
adds output storage/reads; reduced launch count does not prove lower latency. Validate
all shapes and aggregate loads in a separately approved target run. The exact local
source/result commit is linked by the final experiment index; no speedup is claimed.
