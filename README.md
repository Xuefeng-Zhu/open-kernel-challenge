# Open Kernel Challenge experiments

Current status lives in [CONTINUE.md](CONTINUE.md) and the two per-problem ledgers.
The records below the historical heading describe earlier stages and do not set
current permissions, budgets or selected candidates.

| Track | Selected checked code | Official score | Current experiment |
|---|---|---:|---|
| Attention | [A2-11 parallel decode merge](reviews/attention-v2/mixed_fa3_decode_parallel_merge.py) | 1.18767876x | A2-11 retained; A2-12 rejected after passing correctness |
| Projection | P039, on [projection-pipeline-preflight branch](https://github.com/Xuefeng-Zhu/open-kernel-challenge/tree/projection-pipeline-preflight) | 1.17323900x | Selected P039 preserved; serving interfaces audited |

Attention receipts,12shape rows and six weighted loads are in
[its experiment README](reviews/attention-v2/README.md). Only official H200 checks
establish native correctness/performance; CPU models remain explicitly separate.
The [ownership/merge cycle report](reviews/attention-v2/OWNERSHIP_MERGE_CYCLE.md) records the completed counter repair, merge improvement and rejected query hoist.
The [fusion series report](reviews/attention-v2/FUSION_SERIES_RESULTS.md) records
A2-05 through A2-08: all four pass16prerequisites+12timedshapes, with source hashes,
raw timings and CPU limitations. The historical A2-08 score CI is[1.13843065,1.17310367]. The
A2-08 0.45% aggregate increment overlaps its parent CI; its12.54%target latency
cut supports retention. Series point change versus its starting candidate is
+7.307%, not a claim that every small step has a resolved aggregate improvement.
Projection P040's slightly higher point score was not promoted because it missed
its predeclared target. A2-09 `3fb683ba982d448c` passed at 1.14665988x with initialization cost.
A2-10 `08235841c4044048` passed but regressed; A2-11
`cfa8a8670e394f80` passed all16+12 at1.18767876x and is retained.
It improves2.761% over A2-08, with slightly overlapping confidence intervals. See the
[best-version manifest](BEST_VALIDATED_MANIFEST.json) and
[correctness/serving audit](CORRECTNESS_AND_SERVING_AUDIT.md).

The [A13 H200 diagnostic](reviews/attention-v2/A13-RESULT.md) passed 42 replay/cache checks on the selected attention source. Actual concurrent-graph overlap was observed for full decode B32 only; broader concurrency remains unverified. The one approved Modal app is stopped.

Standing reviewed automatic pushes go only to this private repository. Original
attention research history stays local; `attention-fa3-sanitized` preserves source
bytes and maps source commits via analysis/PUSH_EXPORT_MAP.json. Credentials and
local-notes are excluded. No formal/ranked entries, paid compute, new credentials
or Modal launches. User count ceilings for free attention/projection checks have been removed;
platform daily limits still apply. [Current authorization](CURRENT_AUTHORIZATION.md)
supersedes historical count statements below.

## Historical projection baseline and first cycle

# Open Kernel Challenge — dense projections

Live remote iteration: [results/budget](REMOTE_RESULTS.md) and [best completed candidate](BEST.md).
These current records supersede the historical first-baseline and local-only summaries below.

The ten-check cycle is complete: **1.120732× SGLang**,95% interval
**[1.103921,1.137543]**, from [router_pooler_hybrid.py](candidates/router_pooler_hybrid.py),
job `02e099c3724e4a05`. All ten new H200 checks pass9+30 correctness gates. The
57-test Mac suite is separate CPU/static evidence. Budget10/10used,$0,zero scored
entries; no further check is authorized. See BEST.md for overlap/regression limits.

Correctness-first work on `muse_bf16_proj`, with the official task contract, SGLang
baseline and reproducible correctness/diagnostic harness. The original starter remains preserved; subsequent verified GPU improvements are listed above. CPU checks and the first off-board H200 check pass correctness. The measured starter
is slower than SGLang: **0.9076×**, 95% interval **[0.8940, 0.9212]**.

## Reproduce locally

Python 3.12 is used. The Mac environment is pinned in `requirements-macos.lock`.

```sh
python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-macos.lock
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python static_check.py
.venv/bin/python check.py --device cpu --suite all --output results/cpu-correctness.json
```

The full-size suite covers the nine official prerequisite shapes plus all 30 scored
shapes using the unmodified official FP64 reference, per-row limits and weight-column
probe checks. The baseline passes **39/39 CPU cases**; each isolated candidate has its own full-size CPU report. The latest combined contract/regression suite passes **57/57 tests**. The local
Apple M2 host has no CUDA; CPU correctness does not establish H200 correctness or speed.
The nine prerequisite shapes use disclosed local seeds, not hidden judge seeds.

`submission.py` retains the starter algorithm: `x @ W[:n].T`, BF16 for the dense layers
and FP32 for the router. `prepare` retains original weights. The unpooler uses its first
1024 output rows for `m<=8` and all 8192 above that boundary.

## GPU validation

Completed off-board job **`e5a5b2d20da54187`**: **9/9 prerequisite tests and 30/30
scored-shape correctness checks passed** on NVIDIA H200. The judge reports speedup
`0.9076063758`, 95% interval `[0.8939922801, 0.9212204714]`, and normalized `score_us`
`5322.5716885` (not a directly measured single-load latency). No optimization was present in this starter trial.
The main measured gap is the FP32 router for m>8; at m=256 it takes 45.53 µs versus
SGLang's 17.34 µs. See [experiment 001](experiments/001-offboard-baseline.md),
[full sanitized records](results/h200-check.json) and [per-shape CSV](results/h200-per-shape.csv).


The original one-check authorization is complete. The user subsequently approved up to
**10 NEW off-board experiments**, $0 maximum; the prior baseline is excluded.
The admission ledger is [remote-budget.json](experiments/remote-budget.json). Current admitted jobs and remaining allowance are recorded there. The historical
R06 automatic-review block was resolved by specific new human approval and a single
accepted retry; see [the audit history](DENIED_R06.md). Budgeted checks use at most
two immutable independent jobs active; no scored submission, paid resource or new agreement is authorized. Human CLI login and rules acceptance
were verified before sending anything.

```sh
# Historical baseline command; use the admission ledger for the new budget.
okc check submission.py --no-wait --json
okc status <accepted-job-id> --wait --json
```

On an independently approved Linux H200 SXM host in the official SGLang image:

```sh
python check.py --device cuda --suite all --output results/h200-correctness.json
python benchmark.py --repeats 12 --output results/h200-diagnostic.json
```

Do not install the Mac lock into the GPU image. The official environment is
`lmsysorg/sglang:dev-muse-spark-open`; the inspected image digest was
`sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`.
Refresh official metadata before provisioning; no rental is authorized here.

The local diagnostic captures candidate and SGLang CUDA graphs, alternates order,
uses fresh paired inputs, reads 256 MiB before each replay, and checks both outputs.
It reports gross CUDA-event times without guessing replay-overhead subtraction or
claiming an official score/confidence interval. Its GPU branch remains untested until
run on appropriate hardware; the hosted judge is authoritative.

## Official contract and scoring

[Selected task](https://openkernelchallenge.dev/problems/muse_bf16_proj),
[API/agent guide](https://openkernelchallenge.dev/llms.txt),
[technical rules](https://openkernelchallenge.dev/rules),
[official terms](https://openkernelchallenge.dev/terms.pdf).
Selected source files and SHA256 provenance are in `official/`; see [TASK.md](TASK.md)
for dimensions, tolerances, nine prerequisite cases and 30 scored shapes.

Kernel timing is on one NVIDIA H200 SXM using alternating cold-L2 CUDA graph replays.
Per-load time sums each shape's latency times its weight; the problem speedup is the
geometric mean of SGLang/candidate load-time ratios over six loads. Published SGLang
reference values in the snapshot are not measurements made by this project.

At inspection, this task permits 20 off-board checks and 10 scored runs per day,
resetting at 00:00 UTC. A check uploads the source to the contest service, stays off the
leaderboard, and has no confidentiality guarantee. Terms say no purchase or payment
is necessary to participate; no separate hosted-check fee is published. Stop if a
charge appears. No scored submission is authorized.

The **once-per-day scored submission limit belongs to the serving track**, whose
separate check cap is 3/day. Account-specific usage and exact read timestamps are in the admission ledger; the
original baseline also consumes service quota, separately from the ten-new-run budget. Other kernel
tasks can have different check caps. API requests are also limited to 6 submissions/minute
per token and 10/minute per account; these are distinct from the per-day caps. There is
no separately documented final-upload allowance: final placement uses organizer review
and reruns of scored entries. None of these service limits authorizes an extra run here.

2026 kernel deadline: **November 4, 11:59 p.m. Pacific**. Serving has a discrepancy:
the site/API says November 19, while the terms PDF says November 18, both 11:59 p.m.
Pacific. Plan for the earlier date pending organizer clarification. Prizes are on the
later serving track; kernel standings determine serving-queue priority.

Eligibility: individual age 18+, one account, legal/location/export-control requirements
and sponsor/family/household exclusions. The human must review and accept agreements
personally. Submitted contributions require appropriate rights and Apache-2.0 licensing;
terms include sponsor use/publicity rights and arbitration provisions.

## Experiment history and attribution

See [EXPERIMENTS.md](EXPERIMENTS.md) for one-commit-per-experiment rules and
[experiments/](experiments/) for hypotheses, exact commands, environments, outcomes,
source hashes and remote run IDs. Failed experiments must also be recorded.
The [experiment backlog](experiments/BACKLOG.md) explains how to choose one-variable
changes from actual profiles while preserving correctness and quota discipline.
[Isolated candidates](candidates/README.md) have separate experiment commits and CPU
gates. Completed H200 evidence and unadmitted CPU-only variants are distinguished there.
The latest combined local regression suite passes 57/57 tests. Current remote states,
remaining budget and any admission block are recorded in the ledger.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` for challenge,
SGLang and GPU MODE attribution and license scope. Only selected task material and
sanitized evidence are tracked. Credentials, CLI binaries, private account records,
caches, unrelated downloaded tasks and machine-specific notes are excluded.

Current performance diagnosis: [GAP_ANALYSIS.md](GAP_ANALYSIS.md), reproduced from
actual shape timings and explicitly theoretical floor bounds, without new measurements.
Offline candidate comparisons use `compare_results.py`; they retain cross-run caveats.
