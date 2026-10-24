# 033 — unchanged-kernel resources and ordinary CUDA timeline

Missing evidence: prior ephemeral apps did not retain compiled resources/code or
an ordinary timeline. Existing timings cannot distinguish spills/instruction path/
shared resources from launch overhead. No new algorithm is introduced or benchmarked.
Inspect exact immutable fullcomposition c34c497b, poolergrouped573f2f9e and
poolernarrow (source031), not a tuned/modified candidate.

Questions: (1) driver-reported n_regs/n_spills and shared allocation; (2) static PTX
mma.sync versus WGMMA, cp.async versus TMA, local-memory ops; (3) ordinary CUDA
kernel activity within graph for pooler128/256,QKV16/128,unpooler128/256. Static
operand model BN32 quadruples logical X requests while retaining logical W count.
No claim those are actual HBM bytes, achieved occupancy or causal bottlenecks.

Resource ceiling uses architecture limits with explicit allocation-granularity caveat.
No Nsight/hardware counter access. If ordinary PyTorch CUPTI activity is unavailable,
record once/no profiler retry or permissions bypass. Two fresh oracle checks per
existing graph;6graphwarmups. No score, sample deletion, launch-cost subtraction or
benchmark selection from profiler overhead. Preserve full PTX/TTGIR/hash where
available and sanitized activity names/durations. Compiler API follows official
v3.8.0 source and softmax tutorial warmup/_init_handles pattern.

Sources verified2026-10-24:
- https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/compiler/compiler.py
- https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- https://www.nvidia.com/en-us/data-center/h200/
- https://docs.pytorch.org/docs/2.14/profiler.html

One bounded H200, existingimage/auth,20allowlistedsource/licensefiles+manifest;
CPU2/RAM8GiB request+limit,max1/min0/buffer0,scaledown2,retries0,function180s,
startup60s,internal150s/wrapper165s. Reserve$0.45 inside same cumulative$2 stage,
net cap$0. Currentconservative$0.964254 +reserve<$2. No newcredentials/terms/
payment/cap/push. No official check; secondcycle2/10used unchanged.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile projection_resource_probe.py modal_preflight.py modal_gpu_worker.py modal_diagnostics.py finalize_modal_diagnostic.py
.venv/bin/python modal_preflight.py --candidate candidates/projection_pooler_grouped.py --resource-probe --build local-notes/modal-m33-bundle
.venv/bin/python modal_diagnostics.py --experiment M33-resource-probe --bundle local-notes/modal-m33-bundle
.venv/bin/python finalize_modal_diagnostic.py M33-resource-probe
```

Local syntax pass; CPU parser/traffic interpretation tests are not GPU measurements.
GPU diagnostic pending at preparation commit. At most two concrete hypotheses will
be proposed from results, with benefits/risks before any later algorithm benchmark.
