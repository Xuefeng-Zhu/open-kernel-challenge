# 006 — low-batch fused hi+lo split-K router (local gate)

Hypothesis: the real baseline's m=1 and m=8 FP32 router takes 10.13 and 26.13 µs.
A fused pair of BF16 hi/lo tensor-core products plus a split reduction may reduce
that cost without changing the permitted precision. This experiment changes only
router m<=8; m>8 retains experiment 002's control and other projections remain baseline.

Source: `candidates/router_triton_split16.py`; SHA256
`64aeef36775c050908dd5ddcbd9125a10083ea0d6a2bdbaad704b348debb7725`.
Configuration: 16 splits, N tile 32, K tile 64, padded M tile 16; 4 warps/2 stages;
FP32 partials and reduction. The official K=8192 partitions into 16 disjoint 512-column
segments. N=256 produces 8x16=128 partial CTAs for the H200's 132 SMs. This is a
structural scheduling hypothesis, not evidence of achieved occupancy or bandwidth.
Both components are accumulated in FP32, no BF16/FP8 re-quantization of original
projection weights is introduced, and every activation-dependent operation is timed.

```sh
.venv/bin/python check.py --submission candidates/router_triton_split16.py --device cpu --suite all --output results/router-triton-split16-cpu.json
ROUTER_CANDIDATE=candidates/router_triton_split16.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_triton_split16.py
.venv/bin/python -m py_compile candidates/router_triton_split16.py
```

Actual local results: **39/39 full-size oracle cases and 4 focused tests pass**;
Python syntax/static checks pass. Router maximum relative L2: **2.729e-06**.
Focused probes include columns 0/63/64/511/512/4095/4096/8191, in addition to m-boundaries,
outliers, zero rows and negative BF16-only controls. Original weights/inputs remain unchanged.

Environment: M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1, full suite 4 threads.
**No Triton/CUDA compilation or kernel execution occurred on this Mac.** CUDA-only
imports/definitions are skipped on CPU. The CPU path explicitly models split boundaries
and FP32 partial sums using PyTorch; it is not a Triton interpreter or GPU proxy.
The generic hardware driver/compiler, graph capture, device masking and performance
remain unverified until an admitted H200 job. No GPU job belongs to this local gate.

Risks: padded-M work, register pressure, split/reduction traffic and reduction-order
rounding; masks must cover each K exactly once. The two larger-M PyTorch checks already
admitted under the new budget are separate controls, not validation of this kernel.
No extra request was made while those controls remain queued.
