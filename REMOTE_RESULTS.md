# Remote experiment results and budget

**6/10 new requests admitted/reserved; 4 remain.** 5 terminal, 1 pending, 0 uncertain.
Prior baseline job `e5a5b2d20da54187` is excluded. Only off-board checks and $0 spend are authorized; no scored/final submission, new credentials or material agreement.

| Experiment | Job | State | Correctness | Speedup vs SGLang (95% interval) | vs original, normalized |
|---|---|---|---|---|---|
| R01-router-control | `59abe5825690427c` | done | 9/9 + 30/30 | 1.0013× [0.9863, 1.0163] | 1.1032× |
| R02-router-wide | `89fe00c5034a4198` | done | 9/9 + 30/30 | 1.0477× [1.0320, 1.0634] | 1.1543× |
| R03-router-dispatch | `718e0730e1bd4c30` | done | 9/9 + 30/30 | 1.0268× [1.0114, 1.0422] | 1.1313× |
| R04-router-triton16 | `e3bc1481851f4a23` | done | 9/9 + 30/30 | 1.0460× [1.0303, 1.0617] | 1.1525× |
| R05-wide-layout | `1631810d19af4af4` | done | 9/9 + 30/30 | 1.0510× [1.0352, 1.0667] | 1.1579× |
| R06-pooler-triton8 | `b70d87fc51864276` | queued | pending | — | — |

The original starter scored 0.9076063758×. Comparisons to it are reference-normalized across separate runs, not paired same-input measurements. Official within-run SGLang pairings and their intervals are authoritative. Missing results/timings are not inferred from CPU models.

Ledger: `experiments/remote-budget.json`. Exact source/result commits and per-shape evidence: `experiments/index.json`. Uncertain requests stay reserved until reconciled; never blindly retry a POST.

Latest account quota (distinct from this authorization): `{"scored": {"per_day": 10, "used": 0, "left": 10}, "checks": {"per_day": 20, "used": 6, "left": 14}}`.
