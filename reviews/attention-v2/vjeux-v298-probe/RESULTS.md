**Official follow-up:** one non-ranked check `9292ad030b2a4b3d` passed at **2.618804×**, CI [2.579522, 2.658086], with all16 tests and12 shapes. See [official result and comparison](official-check/RESULTS.md). Champion selection remains unchanged.

# v298 local comparison

Pristine Apache-2.0 upstream v298 passed the bounded same-H200 comparison. Selected champion remains unchanged (`3da22a16…`); the native batch made no official check or ranked submission.

All 16 contract cases passed for both implementations. Across all 12 scored shapes and two fresh fixtures: 176 original-oracle checks, including fixed-address QKV/page-table mutation, and 576 timed graph replays passed. Every shape/fixture won all six reversed pairs.

Relative to champion, the six-load geometric means were **1.844421×** and **1.851968×**. These use weights 46 sliding / 16 full, raw whole-call CUDA graph times, cold L2 outside events, and no fixed-cost subtraction. They are not official scores.

| Shape | Champion raw µs, seeds 135901 / 135902 | v298 raw µs | Paired speedup |
|---|---:|---:|---:|
| B1 L1 sliding | 16.112 / 16.624 | 9.536 / 9.856 | 1.676× / 1.662× |
| B1 L1 full | 35.328 / 34.672 | 17.760 / 17.872 | 1.988× / 1.939× |
| B1 L8 sliding | 22.608 / 22.512 | 11.440 / 11.536 | 1.989× / 1.942× |
| B1 L8 full | 38.240 / 38.224 | 19.936 / 19.888 | 1.906× / 1.942× |
| B16 L1 sliding | 20.464 / 20.576 | 14.048 / 14.112 | 1.461× / 1.464× |
| B16 L1 full | 203.344 / 209.280 | 99.152 / 100.032 | 2.050× / 2.094× |
| B16 L8 sliding | 28.448 / 29.104 | 16.480 / 16.608 | 1.743× / 1.732× |
| B16 L8 full | 210.912 / 212.144 | 110.176 / 110.944 | 1.918× / 1.911× |
| B32 L1 sliding | 27.072 / 27.456 | 17.120 / 16.640 | 1.554× / 1.730× |
| B32 L1 full | 300.032 / 300.096 | 179.408 / 180.416 | 1.677× / 1.663× |
| B32 L8 sliding | 35.664 / 35.392 | 20.896 / 20.160 | 1.584× / 1.732× |
| B32 L8 full | 397.504 / 397.488 | 193.264 / 193.840 | 2.050× / 2.048× |

Runtime: NVIDIA H200 / 132 SMs, Torch 2.14.1+cu130, Triton 3.8.0; exact native binary identity is retained in native-result.json. The mirror is not certified as the official judge image. Worker wall time 218.69s. App `ap-T7nUZmPwXPU6yYWfNhleVX` was explicitly stopped and independently observed stopped/tasks0. $2 free-credit reserve; actual charge was not queried; paid limit remains $0.

Static source difference: native FP8 QK, shape-derived global cross-request partitioning for batched full, GPU live lengths/page mapping, producer/consumer staging, paired KV-head processing and fused merge/output. L1 PV uses FP16 and default L8 PV uses FP8, with FP32 state; this is a precision/dataflow difference, not a measured causal attribution. No NCU counters were collected. Global per-device scratch was tested serially only; concurrent graph safety and live length/scale mutation remain unproven. No automatic promotion.

Upstream source/license/provenance are preserved separately; see ATTRIBUTION.md. The existing selected champion and supplied user v4 files were not edited.

The worker/controller/test files are frozen snapshots from the isolated `vjeux-attention-v298-trial` run directory, not a relocatable runner. The exact source/accounting manifest is retained; the service credentials/configuration were never uploaded. Use `candidate.py` as the pristine single-file candidate; this record does not execute any official command.
