# Open Kernel Challenge — dense projections

Latest official passing composition: **1.1486499523x SGLang**, 95% interval
**[1.1314202031,1.1658797016]**, job48b1357016134e52, off-board only.
[Source](candidates/projection_full_composition.py), [result and limitations](OFFICIAL_COMPOSITION_RESULT.md),
[best](BEST.md), [current budgets/status](STATUS.md).9/9prerequisites and30/30scored
shape checks pass, no flags. Prior candidate intervals overlap, so differences
between point scores are not declared statistically resolved.

Second projection cycle2/10used,8remaining; all accepted jobs terminal. Fresh
platform13/20checks used,7left; no scored entry. Four additional Modal screens
passed CUDA correctness but regressed or showed parity; none admitted officially.
All8apps stopped/tasks0, latest$0.87000000credits/$0paid, conservative stageguard
$0.964253594<$2, netcap$0. See [follow-up outcomes](MODAL_FOLLOWUP_RESULTS.md).
104localtests pass; CPU/static and actual H200 evidence remain distinct.
Historical first-cycle records remain in REMOTE_RESULTS.md and their immutable sources.

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
probe checks. The baseline passes **39/39 CPU cases**; each isolated candidate has its own full-size CPU report. The latest combined contract/regression suite passes **95/95 tests**. The local
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
