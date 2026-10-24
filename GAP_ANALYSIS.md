# Gap diagnosis from actual H200 evidence

Our R07-router-native-wide: job `26ea99ee0869482a`, 1.0991970× SGLang.
Leader snapshot 2026-10-24 16:06 UTC: 1.5607955616× (accountforgoofystuff).
At identical normalization, matching it requires 29.57% further latency reduction. This compares independent runs, not paired inputs/GPUs.

Official scoring is the geometric mean of six SGLang/candidate load ratios. Each load sums its five shape latencies times weight62. Shares below average each family’s fraction within those six loads; they sum to100% and reflect local log-score sensitivity to proportional family changes, not a replacement score.

| Family | Measured mean per-load share | Measured latency range µs | TASK floor range µs | Optimistic additional speedup if only this whole family reached its floors |
|---|---|---|---|---|
| pooler | 32.67% | 20.61–29.09 | 14.0–17.4 | 1.1325× |
| qkv_gate | 21.63% | 13.68–19.78 | 7.9–9.8 | 1.1082× |
| unpooler | 21.42% | 6.84–24.17 | 1.7–17.4 | 1.0824× |
| router | 13.96% | 9.14–10.92 | 1.8–2.7 | 1.1250× |
| o | 10.32% | 6.35–9.53 | 3.5–4.5 | 1.0519× |

**Floor columns are theoretical scenarios, not measured/predicted performance or candidate scores.** They assume the published optimistic traffic/compute floor is attainable for every shape of one family while all other times stay fixed. Launch costs, utilization, reductions, precision requirements and memory access can prevent that. Family bounds cannot be added together.

Even the impossible zero-time router scenario gives only 1.1625× further speedup, or 1.2778× SGLang—below the leader snapshot. The gap therefore cannot be closed by router-only work on these measured loads.

R07 changes only router: native split16 at m<=8 and wide hi+lo GEMM above8. All BF16 families remain the original starter. R06 pooler split8 regressed despite passing correctness; retain that failure. R08 isolates router split count; R09 tests a distinct single-row SIMT pooler. QKV split8 stays CPU-only after the related R06 regression. Further BF16 GEMM/GEMV layout, tile and shape-dispatch work must start from profiles and preserve BF16 weights/activations; library presence alone does not justify a port or precision reduction. No unsupported cross-layer fusion: this contract supplies one layer/call, so caching activations is forbidden.

Public-source limits: the official board returns participant aggregate score, attempts and a job ID, not code/per-shape records. The documented GET /jobs/{id} route is own-jobs-only. An unauthenticated read of leader job5987bf672867464e returned HTTP401; no token retry or access bypass was attempted. No verified public competitor source/per-shape details were found in the inspected official sources, so their implementation remains unknown. This is not a claim about unrelated public repositories elsewhere.

Sources: [official task](https://openkernelchallenge.dev/problems/muse_bf16_proj), [official API/mode guide](https://openkernelchallenge.dev/llms.txt), [problem board](https://openkernelchallenge.dev/leaderboard?problem=muse_bf16_proj), `results/remote-R07-router-native-wide.json` and CSV. Reproduce offline with `.venv/bin/python analyze_gap.py R07-router-native-wide` (uses the committed sanitized public leader snapshot; no network/GPU call is made).
