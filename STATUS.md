# Final checkpoint — 2026-10-24T09:15:14.728079+00:00

All five admitted NEW checks are terminal and passing. **5/10 used, 5 remain,
0 pending, 0 uncertain.** The original baseline is separate. $0 spend, zero scored entries.

| Experiment | Job | Official speedup vs SGLang |
|---|---|---|
| R01-router-control | `59abe5825690427c` | 1.0012943× [0.9862749, 1.0163137] |
| R02-router-wide | `89fe00c5034a4198` | 1.0476946× [1.0319792, 1.0634100] |
| R03-router-dispatch | `718e0730e1bd4c30` | 1.0267804× [1.0113787, 1.0421821] |
| R04-router-triton16 | `e3bc1481851f4a23` | 1.0459769× [1.0302873, 1.0616666] |
| R05-wide-layout | `1631810d19af4af4` | 1.0509508× [1.0351865, 1.0667151] |

Each run passed 9/9 prerequisites and 30/30 scored-shape checks. R05 is the highest
observed point score: 1.0509508×, with overlapping intervals versus R02/R04. Its
router m=128/256 regresses R02 by about 16%/12%; R02 remains the better-supported
large-M component and R04 the measured low-M native component. Their native/wide
composition has not been tested. Original `submission.py` is unchanged.

The final local regression suite passes **40/40** on Apple M2, Python 3.12.13,
PyTorch 2.14.1, no CUDA. Each isolated candidate has a 39-case CPU report. CPU models
do not establish native CUDA compilation, graph correctness or H200 speed.

Account quota: **6/20 checks used, 14 left; 0/10 scored used** (includes baseline).
This quota is distinct from the user authorization.

**Remaining blocker:** automatic approval review denied the planned R06 pooler upload
before process creation. No POST, job, uncertainty or budget use resulted. No retry or
workaround was attempted. New sends are guarded in the ledger/wrapper; existing jobs
were followed to completion. See `DENIED_R06.md` for exact action, target and reason.

All source/result commits, per-shape measurements, configurations and sanitized evidence
are indexed in `experiments/index.json`; current budget is `experiments/remote-budget.json`.
