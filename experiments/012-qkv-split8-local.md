# 012 — isolate low-batch QKV BF16 split-K

R01 measured qkv_gate m1/8 at14.20/13.82µs; R02 at14.00/14.53µs, versus the task's
7.9µs theoretical traffic/compute floor. These are actual H200 timings; the floor
is motivation, not an attainable measured promise. This candidate applies the same
BF16 split8 implementation prepared for pooler to QKV ONLY at m<=8, retaining R01
router control and every other path. It does not bundle pooler/router improvements.
QKV's distinct 2304-output width yields288 partial CTAs with the unchanged BN64/S8.
The experiment tests generalization to a different memory footprint, not extra tuning.

Source: `candidates/qkv_triton_split8.py`; SHA256 `cf10ebcda0782330b1e9b0cd633ffa1f299810787bbfd9191af57ea9b2438be2`.
Configuration S8,BN64,BK64,BM16,warps4,stages2; original BF16 weights, FP32 partials,
FP32 reduction then BF16 output. Original weights retained; no activation hoisting.

```sh
.venv/bin/python check.py --submission candidates/qkv_triton_split8.py --device cpu --suite all --output results/router-qkv-split8-cpu.json
.venv/bin/python -m unittest discover -s tests -p test_pooler_candidate.py -v
.venv/bin/python static_check.py --submission candidates/qkv_triton_split8.py
.venv/bin/python -m py_compile candidates/qkv_triton_split8.py tests/test_pooler_candidate.py
```

Actual result: **39/39 CPU cases**, **3 focused QKV tests** and static/syntax checks
pass. The shared BF16 projection suite also reran all3 pooler tests successfully
(6/6 combined). It covers irregular N/K tails, m1/7/8/9, original-weight preservation,
full-width probe columns including split boundaries with exact BF16 output, and
unchanged router control. Environment macOS M2 arm64, Python3.12.13,PyTorch2.14.1,
noCUDA, full suite four threads. CPU is only an explicit FP32 split-sum numerical model.
Triton compilation, masking/capture and H200 performance are unverified for this file.

No remote request has been made. Wait for R06's actual GPU compiler/correctness result
before any admission of this related BF16 kernel; honor at most2 active jobs and the
remaining user budget. Do not knowingly upload a shared compiler/numerical failure.
