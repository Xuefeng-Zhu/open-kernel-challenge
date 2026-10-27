# Projection parallel research: pipeline admission

Preserved parent: projection-bf16-local at0eabf0e48640be3c401c437471af724fa06154d9.
Best source candidates/projection_full_composition.py, source commit
b9b0fd9fb0cf326b5836167ffc29d1a0d3386fa7, SHA256
c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7.
Official job48b1357016134e52 passed1.1486499523461586×,
CI[1.1314202030609661,1.165879701631351]. No candidate source changed.

## Experiment card

Problem: medium/large BF16 projection, initially pooler m256. Pooler is32.14%
of mean load time, but the proposed single target is31.64% of only verify32.
Hypothesis: explicit TMA-fed asynchronous BF16 pipeline plus greater M-tile reuse
may reduce copy cost versus the rejected kernels. Actual memory bandwidth,
stalls and achieved occupancy are unknown; this is a hypothesis, not a measured
bottleneck. Prototype geometry would be BM64/BN64/BK64, pooler m256 only, original
BF16 operands and FP32 accumulation, unchanged other paths.

Intended controlled change: compare TMA against conventional-copy code at the
same new geometry/scheduling. Keep library fallback for unsupported/tail layouts.
Do not attribute a combined geometry/instruction change to TMA alone. No new
kernel is implemented or claimed ready for GPU admission in this research phase.

Actual outcome: offline score reconstruction matches the existing official score
exactly. A10% latency reduction at this single shape changes aggregate score only
+0.5373%; +1.5% aggregate needs27.0105% latency reduction, from30.2629us to22.0887us.
Even reaching the published optimistic17.4us floor yields only+2.4362% aggregate
when every other timing is frozen. These are scenarios, not predictions or GPU
results. Model is projection_admission.py; output and exact receipt hash are in
results/projection-pipeline-admission.json.

Decision: do not spend an official check on an unimplemented/unvalidated pipeline
or another tile sweep. No sufficiently supported, locally admissible candidate
emerged from this bounded review. This does not disprove TMA. Continue only after
an isolated descriptor/barrier-compatible prototype and conventional-copy control
exist, with a credible benefit beyond aggregate noise. No extra approval gate is
invented; this is an engineering admission decision within the existing budget.

## Prior negative evidence reviewed

- MODAL_FOLLOWUP_RESULTS.md: QKV transpose regresses m256; unpooler transpose is
  aggregate parity. Taking its tiny m128-only gain is about0.11% aggregate.
- Pooler BM32/BN128 and BM32/BN32 regress badly. Raising CTA count is not enough.
- HOPPER_DIAGNOSTICS.md: pooler m256 is already one library GEMM, so removing a
  reduction launch is not a mechanism for this target. Logical request models
  do not establish actual HBM traffic. Register/shared ceilings are not achieved
  occupancy. Inspected failed poolers already use cp.async but not TMA/WGMMA.
- QKV_M128_SINGLEPASS_RESULTS.md: BM64/BN32 full-K emits WGMMA, zero reported spills,
  yet55.10us versus21.89us control,0/15 paired wins. WGMMA alone is not justification.

## Reproducibility and boundaries

Read TASK.md/AGENTS.md first. No .agents skill directory is present in the original
checkout. This worktree is isolated from the active attention-fa3-local branch.

Run using the existing primary-workspace Python:

```sh
<primary-workspace>/.venv/bin/python projection_admission.py
```

The script only parses frozen CSV measurements, checks30shapes/sixloads and
reconstructs the known score. No CUDA, third-party software execution, package
installation, credential read or benchmark launch. No new CPU kernel correctness
pass is claimed because no new kernel exists.

User projection budget remains8checks; UTC reset gave20daily checks but does not
reset authorization. Attention E5 uses its own budget and exclusively owns the
GPU slot until terminal. Primary coordinator owns scheduling and integration.
Do not launch from this worktree. No push, scored entry or paid compute.

IMPORTANT: this worktree's old Modal ledger is historical. Current cumulative
guard is$1.708917914/$2 across both tracks, authoritative in the primary workspace
attention ledger and ignored shared bridge. Never reset it to this branch's old
$1.146 value. No Modal app was launched; prior BlobCreate401 remains unresolved.
No new or renewed credentials or payment/cap change occurred.
