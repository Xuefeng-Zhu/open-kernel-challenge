# Current gap update — official P2-02, 2026-10-24

Source full composition c34c497b, job48b1357016134e52,1.1486499523xSGLang,
95%CI[1.1314202031,1.1658797016]. Versus the dated16:06leader snapshot1.56079556,
26.41%further latency reduction would be required at identical normalization.
This is a cross-run/reference comparison, not current rank or a competitor profile.

| Family | Measured mean per-load share | Measured latency range µs | TASK floor range µs | Optimistic additional speedup if only this whole family reached its floors |
|---|---|---|---|---|
| pooler | 32.14% | 18.51–30.26 | 14.0–17.4 | 1.1153× |
| qkv_gate | 22.42% | 12.05–20.13 | 7.9–9.8 | 1.1122× |
| unpooler | 21.01% | 4.12–24.42 | 1.7–17.4 | 1.0755× |
| router | 13.49% | 7.37–11.44 | 1.8–2.7 | 1.1175× |
| o | 10.95% | 6.86–9.40 | 3.5–4.5 | 1.0567× |

**Floor columns are theoretical scenarios, not measured/predicted performance or candidate scores.** They assume the published optimistic traffic/compute floor is attainable for every shape of one family while all other times stay fixed. Launch costs, utilization, reductions, precision requirements and memory access can prevent that. Family bounds cannot be added together.

Even the impossible zero-time router scenario gives only 1.1560× further speedup, or 1.3278× SGLang—below the leader snapshot. The gap therefore cannot be closed by router-only work on these measured loads.


Follow-up Modal screens test these opportunities, not competitor implementations.
QKV library layout regresses m256; unpooler library layout aggregate parity;
native pooler BM32/BN128 is much slower despite correctness. No new score inferred
from those Modal gross measurements. Next BN32 test asks whether higher parallelism
recovers native pooler regression; its result is recorded in MODAL_FOLLOWUP_RESULTS.md.
No cross-layer input fusion/cache or lower BF16 precision is allowed.

Reproduce current shares/floor scenarios offline:
`.venv/bin/python analyze_gap.py P2-02-full-composition`.
The historical analysis below preserves its earlier R10 measurement context.

---

# Gap diagnosis from actual H200 evidence

Subsequent official result: full composition job48b1357016134e52 passes at
1.1486499523x SGLang,95%CI[1.1314202031,1.1658797016]. See
OFFICIAL_COMPOSITION_RESULT.md for current budgets/profile/next hypothesis.
The dated analysis below retains its original measurement context.

Our R10-final-composition: job `02e099c3724e4a05`, 1.1207320× SGLang.
Leader snapshot 2026-10-24 16:06 UTC: 1.5607955616× (accountforgoofystuff).
At identical normalization, matching it requires 28.19% further latency reduction. This compares independent runs, not paired inputs/GPUs.

Official scoring is the geometric mean of six SGLang/candidate load ratios. Each load sums its five shape latencies times weight62. Shares below average each family’s fraction within those six loads; they sum to100% and reflect local log-score sensitivity to proportional family changes, not a replacement score.

| Family | Measured mean per-load share | Measured latency range µs | TASK floor range µs | Optimistic additional speedup if only this whole family reached its floors |
|---|---|---|---|---|
| pooler | 32.31% | 18.13–28.79 | 14.0–17.4 | 1.1223× |
| qkv_gate | 22.42% | 13.68–19.55 | 7.9–9.8 | 1.1149× |
| unpooler | 21.56% | 6.83–24.14 | 1.7–17.4 | 1.0828× |
| router | 13.01% | 7.18–11.01 | 1.8–2.7 | 1.1122× |
| o | 10.70% | 6.77–9.22 | 3.5–4.5 | 1.0550× |

**Floor columns are theoretical scenarios, not measured/predicted performance or candidate scores.** They assume the published optimistic traffic/compute floor is attainable for every shape of one family while all other times stay fixed. Launch costs, utilization, reductions, precision requirements and memory access can prevent that. Family bounds cannot be added together.

Even the impossible zero-time router scenario gives only 1.1496× further speedup, or 1.2884× SGLang—below the leader snapshot. The gap therefore cannot be closed by router-only work on these measured loads.

The final verified composition uses split32 native router at m<=8, wide hi+lo router above8, and single-row SIMT pooler. Other BF16 paths remain original. R06 pooler split8 regressed despite passing correctness; retain that failure. R08 validated router split32 and R09 validated the one-row SIMT pooler component; R10 checks their composition. QKV split8 stays CPU-only after the related R06 regression. Further BF16 GEMM/GEMV layout, tile and shape-dispatch work must start from profiles and preserve BF16 weights/activations. No unsupported cross-layer fusion: this contract supplies one layer/call, so caching activations is forbidden. All10 new checks are terminal; no further quota is authorized. A future GPU experiment needs a new explicit budget.

Public-source limits: the official board returns participant aggregate score, attempts and a job ID, not code/per-shape records. The documented GET /jobs/{id} route is own-jobs-only. An unauthenticated read of leader job5987bf672867464e returned HTTP401; no token retry or access bypass was attempted. No verified public competitor source/per-shape details were found in the inspected official sources, so their implementation remains unknown. This is not a claim about unrelated public repositories elsewhere.

Sources: [official task](https://openkernelchallenge.dev/problems/muse_bf16_proj), [official API/mode guide](https://openkernelchallenge.dev/llms.txt), [problem board](https://openkernelchallenge.dev/leaderboard?problem=muse_bf16_proj), `results/remote-R10-final-composition.json` and CSV. Reproduce offline with `.venv/bin/python analyze_gap.py R10-final-composition` (uses the committed sanitized public leader snapshot; no network/GPU call is made).
