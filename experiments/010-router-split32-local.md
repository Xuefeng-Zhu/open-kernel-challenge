# 010 — one-variable split parallelism alternative (local gate only)

R04 verified split-K 16 on H200. Its diagnostic profile attributes about 79% of GPU
time to the partial kernel; the known grid is 128 partial CTAs for 132 SMs. This
hypothesis changes **only SPLITS=16 to 32**, doubling available independent partial
CTAs and halving K per CTA, while increasing partial/reduction traffic. This is not
an occupancy or bandwidth measurement; warm diagnostic times do not replace the
cold-L2 judge times. No broad tile/warp/precision sweep is bundled into this change.

Source: `candidates/router_triton_split32.py`; SHA256
`3f269bf18100c8dbe111dfe6f4380ab323f684a23855e9a0fee5379bc451adf9`. Control: R04 job `e3bc1481851f4a23`.
All dtype/weight rules, N tile 32, K tile 64, M tile 16, warps/stages and larger-M
router control remain unchanged.

```sh
.venv/bin/python check.py --submission candidates/router_triton_split32.py --device cpu --suite all --output results/router-triton-split32-cpu.json
ROUTER_CANDIDATE=candidates/router_triton_split32.py .venv/bin/python -m unittest discover -s tests -p test_router_candidates.py -v
.venv/bin/python static_check.py --submission candidates/router_triton_split32.py
```

Actual result: **39/39 CPU oracle cases, 4 focused tests and static checks pass**.
Environment: M2/macOS arm64, Python 3.12.13, PyTorch 2.14.1, full suite four threads.
The CPU model exercises partitioned FP32 sums and probes but does not compile or run
Triton. **CUDA correctness/capture/performance are unverified** for this split count.

No GPU upload or request was made for this candidate. Further uploads are paused:
automatic approval review denied the distinct planned R06 pooler upload before its
command executed, and this local experiment is not a workaround or replacement POST.
The pending R05 monitor continues; five admitted checks remain the budget usage.
