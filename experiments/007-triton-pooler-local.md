# 007 — low-batch BF16 pooler split-K (local gate)

Hypothesis: R01 measures the low-batch pooler at about 21.2–21.7 µs for a 64 MiB
weight, versus the task's roughly 14 µs memory floor. A tiled split-K GEMM may improve
that memory-heavy shape. Theoretical traffic is motivation, not achieved bandwidth.
Only pooler m<=8 changes; other BF16 projections and the validated R01 router control
remain unchanged. No inter-call/layer fusion or activation caching is attempted.

Source: `candidates/pooler_triton_split8.py`; SHA256
`76f202f5fe9efb23ce0e10c6739871fab74512464a0e9f1229cda53c0c625406`.
Configuration: 8 splits, N tile 64, K tile 64, padded M tile 16, 4 warps, 2 stages;
FP32 partials/reduction followed by BF16 output. Original BF16 weights stay BF16.
Official [4096,8192] weights yield 512 partial CTAs and disjoint 1024-column K segments.

```sh
.venv/bin/python check.py --submission candidates/pooler_triton_split8.py --device cpu --suite all --output results/router-pooler-split8-cpu.json
.venv/bin/python -m unittest discover -s tests -p test_pooler_candidate.py -v
.venv/bin/python static_check.py --submission candidates/pooler_triton_split8.py
.venv/bin/python -m py_compile candidates/pooler_triton_split8.py
```

Actual local result: **39/39 full-size CPU oracle cases, 3 focused tests and static/
Python syntax checks pass**. Focused coverage includes irregular N/K tails and m=7/8/9,
full-width probes at columns 0/63/64/511/512/4095/4096/8191 (bit-exact BF16 output),
weight preservation and unchanged FP32-router correctness. Evidence:
`results/router-pooler-split8-cpu.json`.

Environment: M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1, full suite four threads.
**This is only a CPU model of partitioned FP32 accumulation and BF16 rounding.**
Triton/CUDA definitions are skipped on this Mac. No CUDA compilation, graph capture,
device masking or H200 performance validation is claimed; no remote job belongs to
this local gate. Risks include padded compute, register pressure, reduction traffic,
warp/tile choice and actual cold-L2 memory throughput. GPU admission requires capacity
inside the authorized budget and review of already admitted results, not an automatic run.
