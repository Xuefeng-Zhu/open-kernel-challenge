# Evidence-based experiment backlog

This is a plan, **not an experiment result or permission for a run**. The only approved
remote check is baseline job `e5a5b2d20da54187`; do not submit another. It finished with correctness passing and speedup **0.9076×**. Its source is
commit `fde8a042944c742c20b76f27fcc030d767fd0058`, SHA256
`78bd5697391a335d5ba771035624bde12e36ecfa5f3058f3db08d00fbe08ae3e`.
The baseline remains unchanged and preserved; additional runs need fresh approval.

## What the contract establishes

- Five projections at each of six loads: `m=1/16/32` decode and `m=8/128/256` verify.
  Every shape has weight 62. Aggregate each load's weighted latency first, then take
  the geometric mean of the six SGLang/candidate load-time ratios. A single fast shape
  is insufficient to establish a better problem score.
- Four projections have BF16 weights/outputs. The router has FP32 weights/output and
  a tighter relative L2 limit, **1e-4**, versus **1e-2** for the other projections.
  Each row must meet five times its limit; weight-column probes must meet **1e-3**.
- SGLang's published baseline uses its BF16 GEMM dispatcher (cuBLAS on H200) for the
  four BF16 projections. Router: FP32 linear at `m<=8`, two BF16 hi/lo products with
  FP32 accumulation at `m>8`. Our starter instead uses FP32 router products at every m.
- Weight traffic alone is 36 MiB (`qkv_gate`), 16 MiB (`o`), 64 MiB (`pooler`),
  8/64 MiB (`unpooler`, at/beyond `m=8`) and 8 MiB (FP32 router). These exact byte
  counts suggest inspecting memory bandwidth and low-batch utilization; they are
  **not measured bottlenecks**. The task's theoretical floors are not achieved times.
- Untimed `prepare` may transform only weights, without reducing their precision
  except the explicitly allowed router BF16 hi+lo representation. Timed functions
  must remain graph-capturable and recompute every activation-dependent result.

Sources: preserved [TASK.md](../TASK.md), [baseline](../official/baseline.py),
[oracle](../official/reference.py), [API snapshot](../official/muse_bf16_proj.json),
[technical rules](https://openkernelchallenge.dev/rules).

## First measured evidence (job e5a5b2d20da54187)

The first profile is now available. The unchanged starter passes 9/9 prerequisites and
30/30 scored-shape checks, but its speedup is **0.9076×**, 95% interval
**[0.8940, 0.9212]**. Router m=16/32/128/256 takes **25.90/19.48/28.57/45.53 µs**
versus SGLang's **14.69/14.94/16.26/17.34 µs**. The BF16 projections are mostly near
parity. This makes hypothesis 1 the clearest first controlled change; it does not
establish that a custom GEMV, split-K or layout change will help. Full evidence:
[experiment 001](001-offboard-baseline.md), [JSON](../results/h200-check.json),
[all 30 shape timings](../results/h200-per-shape.csv).

## Ordered hypotheses — use the real profile

| Priority | One-variable experiment | Hypothesis / evidence needed | Correctness risk and required coverage |
|---|---|---|---|
| 0 | Establish the unchanged baseline | Finish the accepted job; inspect all prerequisite tests, all 30 shape times, SGLang times, load totals and confidence interval. Identify where weighted time is actually spent. | Passing CPU tests does not prove CUDA correctness, capture or H200 timing. A failed test is a result to record, not a reason to launch a duplicate. |
| 1 | Router only: use SGLang's hi+lo path for `m>8`; keep everything else fixed | Test whether the starter's FP32 router is a measurable disadvantage at 16/32/128/256 tokens. This initially reproduces SGLang's existing technique; do not call it novel. Compare router shape times and total load effect. | FP32 accumulation of both terms; BF16-only weights are invalid. Check m=8/9 boundary, router outliers, weight-column probes and fresh weights. CPU cannot validate CUDA `out_dtype` support. |
| 2 | One BF16 projection at low batch: specialized GEMV/small-M kernel | Only pursue a projection shown expensive in the profile. Test one layer and one static-M range at a time against the same source baseline. Inspect device bandwidth, grid utilization and graph latency if profiler output provides them. | Read all K and all requested output rows; avoid input-dependent shortcuts. Cover m=1/7/8/9 and non-power-of-two sizes, probes, outliers, output dtype and per-row bounds. Do not repeatedly stream a weight for every token without measuring the cost. |
| 3 | Change only that kernel's split-K factor | Test whether more independent blocks improve H200 utilization enough to offset partial-buffer traffic and reduction work. Sweep a small declared set, one configuration per recorded experiment. | Races or incomplete reductions; cancellation/rounding changes. Start with FP32 partials. Each candidate needs the full official oracle and nearby shapes, including m=512/1000 outlier tests. |
| 4 | Change only a weight layout in `prepare` | A lossless contiguous transpose/repack may help an observed access pattern. Hold kernel math, launch parameters and precision fixed to isolate layout effect. | Preserve every BF16/FP32 value and slice semantics. No activation work or output cache in preparation. Check full versus first-1024-row unpooler at m=8/9; account for memory and 300-second prepare budget. |
| 5 | Router only: fuse the two hi/lo products into one custom kernel | If router launch/reduction cost remains material after priority 1, test whether one kernel retains required accuracy while reducing overhead. Compare against the validated two-product router, not against an unrelated changed kernel. | Accumulate both components in FP32; no FP32-to-BF16-only approximation. Validate tight router tolerance, probes, outliers, fresh seeds and m=1/8/9/16/128/256. |
| 6 | One larger-M BF16 projection: change one tile dimension or warp count | Only if a measured large-M shape is a remaining hotspot. Keep all other kernel/configuration choices fixed; inspect whether any gain survives the six-load aggregate. | Masks on tails, correct strides and BF16 output rounding. Test prerequisite m=33/130/512/1000 and nearby dimensions permitted by the contract; avoid tuning only to published seeds/shapes. |

These priorities are conditional. If the baseline profile points elsewhere, record
that evidence before changing the order. A layout experiment or router experiment
must not silently include a split-K, tile-size and precision change at the same time.

## Efficient iteration and evidence chain

1. **Plan without a GPU call:** one hypothesis, one change, explicit configuration,
   expected affected shapes, correctness risks and a stopping criterion. Create a new
   experiment record. Keep the previous passing baseline recoverable.
2. **Cheap local filter:** run static/source-hash checks and the focused tests. Run
   the complete CPU oracle where the implementation can execute on CPU. A CUDA-only
   implementation must be labeled unvalidated locally, not given a silent CPU fallback
   that is presented as GPU evidence. Record failures as well as successes.
3. **Before any additional GPU call:** get separate approval, reread live account
   quota, confirm the target/image and keep the spending cap. The service's allowance
   is not the user's authorization. Checks are off-board but upload the whole source.
4. **Target validation:** use approved CUDA/H200 correctness and CUDA-graph capture;
   test nearby shapes and fresh seeds. Use graphs, cold L2 and paired same-GPU inputs
   for comparisons. Eager or warm-cache numbers are diagnostic and cannot substitute
   for judge timing. A rented GPU is not authorized here.
5. **One accepted job, one result:** save the response job ID immediately and monitor
   it. Resolve uncertain outcomes with read-only jobs/status inspection before any
   retry. Never resubmit simply because a queue is slow.
6. **Evaluate the whole score:** retain every shape's candidate/SGLang latency, ratio,
   load and weight; calculate load totals before the six-load geometric mean. Preserve
   the official confidence interval and exact result fields. Gains inside its noise
   interval (at least ±1.5%) are inconclusive. Do not cherry-pick faster shapes.
7. **One experiment commit:** record the actual outcome and commit its source/config
   and sanitized evidence separately. Do not squash distinct experiments or fabricate
   old commits. Do not change a candidate while its result is pending. No new kernel
   candidate is implemented by this backlog.

For this project, the exact local commands are:

```sh
.venv/bin/python static_check.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python check.py --device cpu --suite all --output results/cpu-correctness.json
```

The current static allowlist is intentionally minimal; review it explicitly if a later
authorized candidate introduces Triton/CUDA imports. Do not weaken it just to obtain a pass.

Every GPU experiment's evidence must link:

```text
experiment ID -> source commit + submission SHA256 + exact configuration
              -> accepted job ID + mode + target environment + quota before/after
              -> correctness result + all per-shape timings + aggregate + uncertainty
              -> result commit containing the record and sanitized JSON
```

The commit containing a completed experiment record is its result commit (retrieve
with `git log -1 --format=%H -- experiments/<record>.md`). Record that exact commit in
the final report or a subsequent history index, rather than placing a self-referential
commit hash inside its own file. Do not invent missing runtime, timing or interval data.

## Quotas are different from approvals

Current authenticated API: selected kernel **20 checks/day, 10 scored runs/day**;
after our accepted check, **1 check used, 19 left; 0 scored used, 10 left**. Other
kernel tasks have different check caps. Serving is currently closed and has **1 scored
submission/day, 3 checks/day**. Daily reset: 00:00 UTC; per-minute limits also apply.
No separate final-upload quota is documented; final placement uses organizer review
and reruns. These facts do not authorize another remote request.

[Kernel API](https://openkernelchallenge.dev/api/v1/problems),
[serving API](https://openkernelchallenge.dev/api/v1/serving),
[current guide](https://openkernelchallenge.dev/llms.txt).
