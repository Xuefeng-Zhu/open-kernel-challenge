# Ownership and decode-merge cycle: completed

**Follow-up:** A13 subsequently passed 42 bounded H200 replay checks on this exact selected source. Actual overlap was observed for full decode B32 only; see [A13 result](A13-RESULT.md). Statements below about zero native stress describe the earlier cycle snapshot.

Updated 2026-10-25T05:58:09.177342+00:00. All four admitted jobs are terminal; no active or uncertain request. Selected attention source is A2-11, 1.1876787561× SGLang, 95% interval [1.1698635747, 1.2109342502]. Projection P039 remains 1.1732389990×; its code and GPU result were unchanged in this cycle.

## Experiment cards

| Experiment | Problem and isolated change | Result and decision | Job |
|---|---|---|---|
| A2-09 | Independent calls shared a completion counter. Give each invocation its own timed zeroed counter; math and dispatch unchanged. | All 16+12 pass, 1.14665988×. Ownership repaired, but initialization costs time. Preserve as correctness baseline. | `3fb683ba982d448c` |
| A2-10 | Avoid counter initialization on decode by moving the identical serial final merge into a separate launch. Eight-token fallback unchanged. | All 16+12 pass, 1.14054863×. Rejected for performance; several target paths slowed. | `08235841c4044048` |
| A2-11 | Serial merge is expensive. Give each query head a CTA and reduce split partials in parallel FP32, with neutral padding; producer and partition unchanged. | All 16+12 pass, 1.18767876×. Retained: every predeclared target met, no >5% unchanged-control regression. | `cfa8a8670e394f80` |
| A2-12 | Sliding decode B1 repeats Q normalization/RoPE across splits. Prepare the identical BF16 Q once in a timed per-call buffer. Other 11 computations unchanged. | All 16+12 pass, 1.17342999×. Rejected: target normalized latency increased 8.676%, missing the 15% reduction goal. | `b24bbd05419c499a` |

Each experiment has its own source and result commit, exact source SHA, CPU preflight, official receipt, 12-shape CSV, assessment and decision. Read the corresponding A2-NN-EXPERIMENT.md and A2-NN-RESULT.md. Original source commits are mapped to sanitized private export commits in analysis/PUSH_EXPORT_MAP.json; source bytes are unchanged.

## What improved and what did not

A2-11 sliding decode B1/B16/B32 normalized latency cuts versus A2-10 were 25.868% / 16.571% / 5.712%; full decode B32 improved 1.922%. Aggregate +4.132% versus A2-10 has nonoverlapping reported intervals. Compared with the old fastest A2-08, the point gain is +2.761%, with slightly overlapping intervals; this latter aggregate difference is not statistically resolved by these runs.

A2-12 producer activity dropped from 8.1 to 6.8 µs, but its extra preparation activity cost 2.5 µs (merge 1.7→1.6 µs). The measured whole-call target went 11.821→12.916 µs, 8.676% slower after same-run-reference normalization. Profile activities support the direction; adding activity times is not the judge measurement. The predeclared 15% target-only model predicted 1.20499534×; the actual complete run was 1.17342999×. The aggregate -1.200% has overlapping intervals, but the target hypothesis is unsupported.

## Evidence and reproducibility

All four runs used H200 and the same evaluator image digest 9a7f287b80d772c6f636bbcb57422c619fe326e2aabc53803b85007bca93f68e, code hash 1d519ebf2014fef91b41ffea9bcd7e9c5066784ef5a2a5e2cf06a8cec554d188. Full machine/driver/evaluator metadata is in each receipt. Exact package/compiler versions were not separately emitted by these checks; do not substitute historical local/Modal versions.

The official harness times cold-L2 CUDA graph replays, alternating candidate/reference on fresh inputs, subtracts fixed graph overhead, and reports median paired ratios. Raw medians, adjusted times and paired ratios remain separate. Each of six loads is C=46×sliding+16×full (likewise S); the score is the geometric mean of the six S/C values. Comparisons between our variants are separate runs normalized against their own SGLang measurements, not directly paired variant-versus-variant measurements.

Actual CPU/static commands and passed counts (Mac, Python 3.12.13; no CUDA/Triton execution):

```sh
.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_local_tickets.py -v  # 5
.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_decode_merge.py -v  # 5; 120 float64 cases
.venv/bin/python -B reviews/attention-v2/test_decode_parallel_merge.py  # 8
.venv/bin/python -B reviews/attention-v2/test_decode_qhoist.py  # 7
.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_check_cycle_limits.py -v  # 12
.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_native_ownership_probe.py -v  # 5
.venv/bin/python -B reviews/attention-v2/native_ownership_probe.py --list-cases
```

These were already executed during source preflight; timestamps and scope are in each CPU report. The projection policy change passed 12 admission tests and its full 143-test CPU suite. No new projection GPU check was dispatched during this ownership cycle.

Official commands that actually admitted each job were `bin/okc check reviews/attention-v2/<candidate>.py --no-wait --json`, using respectively mixed_fa3_local_tickets.py, mixed_fa3_decode_merge.py, mixed_fa3_decode_parallel_merge.py, mixed_fa3_decode_qhoist.py. Exact argv/source commit/hash/job are in attention-cycle2-budget.json. Do not rerun these commands merely to reproduce this report. Read existing results with `bin/okc status <job> --json`; offline reconstruction is `.venv/bin/python reviews/attention-v2/assess_cycle2.py A2-12` (the manual retain/reject decision follows the computed metrics).

## Correctness limits and reasoned stopping point

A2-11 has no device-global completion counter. Decode fallback uses separate merge; eight-token fallback retains invocation-local zeroed counters. Official 16 prerequisites plus 12 scored shapes verify outputs and new cache within the exact official tolerances, including nearby shapes/page/scales represented by those fixtures. They do not establish arbitrary concurrent stream execution or dedicated current-source A→B→A replay/untouched-cache coverage.

The prepared native ownership probe has five passing CPU tests, **zero native stress replays**. Its default plan is five L1 cases × two independent graph captures × A/B/A = 30 checked replays with changing actual lengths, stable addresses, page64/non-unit scales, cache writes and untouched rows. L8 local-ticket coverage still needs a separate native stress case. Enqueuing two streams does not prove physical GPU overlap. Same graph/output/cache overlapping use and deliberately shared capture pools remain outside scope.

The next priority is this exact-source native validation, followed by exact-source resource profiling if performance work resumes. It needs a permitted H200 environment with the official dependencies; this Mac cannot execute CUDA, new Modal launches/credentials are forbidden in this phase, and ordinary OKC checks do not accept an arbitrary stress harness. The probe was not launched or denied. Historical Modal report retrieval A09 remains failed with BlobCreate 401; no retry, credential renewal or new app was attempted.

No equally supported next performance mechanism is established. Full B32 producer activity is 99.3% of its profile, which locates the cost but does not distinguish spills, occupancy, memory or instruction throughput. Current-source register/spill/shared-memory/counter measurements are absent. Extending full prologue fusion to three larger paths would yield only about 0.84% aggregate even if each hypothetically saved 5 µs; that is scenario arithmetic, not a measured opportunity. Stop this bounded cycle instead of launching parameter sweeps or repeating the failed query-hoist mechanism. This is an evidence-based stopping point, not an exhausted quota or a failed official prerequisite.

## Resource and quota status

At 2026-10-25 05:55:57 UTC: attention checks 14/20 used, 6 left; projection 3/20 used, 17 left; scored submissions 0/10 used for both. Reset is 2026-10-26 00:00 UTC. User count ceilings have been removed; platform limits still apply. No active jobs appeared in the latest 35 returned jobs and the coordinator slot is terminal. See analysis/CURRENT_PLATFORM_STATE.json.

No paid compute, ranked entry, new credentials, Modal launch or serving deployment. Historical conservative Modal accounting remains $1.708917914/$2 and prior apps were stopped; no new spend or resource was created here. Reviewed source/results are automatically pushed only to the existing private repository, with original attention history preserved locally and sanitized per-experiment export mapping.
