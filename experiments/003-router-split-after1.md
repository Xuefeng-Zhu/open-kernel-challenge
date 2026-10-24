# 003 — change only the router dispatch boundary

Hypothesis: the measured m=8 FP32 router cost (26.13 µs, against SGLang's 26.27 µs)
suggests testing the legally allowed hi+lo path earlier. This experiment changes
**only** the threshold in experiment 002 from 8 to 1, retaining the one-token FP32
path and all matrix layouts/operations. No performance inference is made from CPU.

Candidate: `candidates/router_split_after1.py`; SHA256 `5945aa3b214cece1363641a86b7cae2827cdd9d7af956fa42eab4af91994fdb8`.
Control: experiment 002 / `candidates/router_split_after8.py`. Proven GPU baseline
remains commit `fde8a042944c742c20b76f27fcc030d767fd0058`, job `e5a5b2d20da54187`.
Configuration: threshold 1, two BF16 hi+lo products, FP32 accumulation, contiguous
[K,N] weights. The rule explicitly permits router hi+lo at any m.

```sh
.venv/bin/python check.py --submission candidates/router_split_after1.py --device cpu --suite all --output results/router-after1-cpu.json
ROUTER_CANDIDATE=candidates/router_split_after1.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_split_after1.py
```

Actual result: **39/39 full-size CPU cases and 3/3 focused tests pass; static checks
pass**. Maximum router relative L2: **2.729e-06**, below 1e-4. Focused tests cover
m=1/2/7/8/9/16/32/128/256, outliers, probes, zero rows, fresh inputs, a BF16-only
negative control and unchanged other projections/weights.

Environment: macOS arm64/Apple M2, Python 3.12.13, PyTorch 2.14.1, four threads for
the full suite. CPU uses the explicitly labeled FP32 numerical model of BF16 operands.
**CUDA compilation/execution, H200 correctness, capture and performance remain
unverified.** No GPU job ID, upload, check, scored entry or paid compute for this
experiment. Evidence: `results/router-after1-cpu.json`; exact commit linked in the
final experiment index. Risk: two launches may cost more than FP32 at some small m;
only a separately approved H200 comparison can establish the useful dispatch boundary.
