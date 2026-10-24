# Remote experiment results and budget

**10/10 new requests admitted/reserved; 0 remain.** 9 terminal, 1 pending, 0 uncertain.
Prior baseline job `e5a5b2d20da54187` is excluded. Only off-board checks and $0 spend are authorized; no scored/final submission, new credentials or material agreement.

| Experiment | Job | State | Correctness | Speedup vs SGLang (95% interval) | vs original, normalized |
|---|---|---|---|---|---|
| R01-router-control | `59abe5825690427c` | done | 9/9 + 30/30 | 1.0013× [0.9863, 1.0163] | 1.1032× |
| R02-router-wide | `89fe00c5034a4198` | done | 9/9 + 30/30 | 1.0477× [1.0320, 1.0634] | 1.1543× |
| R03-router-dispatch | `718e0730e1bd4c30` | done | 9/9 + 30/30 | 1.0268× [1.0114, 1.0422] | 1.1313× |
| R04-router-triton16 | `e3bc1481851f4a23` | done | 9/9 + 30/30 | 1.0460× [1.0303, 1.0617] | 1.1525× |
| R05-wide-layout | `1631810d19af4af4` | done | 9/9 + 30/30 | 1.0510× [1.0352, 1.0667] | 1.1579× |
| R06-pooler-triton8 | `b70d87fc51864276` | done | 9/9 + 30/30 | 0.9930× [0.9781, 1.0079] | 1.0941× |
| R07-router-native-wide | `26ea99ee0869482a` | done | 9/9 + 30/30 | 1.0992× [1.0827, 1.1157] | 1.2111× |
| R08-router-split32 | `ab74f25caa2e46a8` | done | 9/9 + 30/30 | 1.0617× [1.0458, 1.0776] | 1.1698× |
| R09-pooler-simt | `97aa5690ff134369` | done | 9/9 + 30/30 | 1.0076× [0.9925, 1.0227] | 1.1102× |
| R10-final-composition | `02e099c3724e4a05` | running | pending | — | — |

The original starter scored 0.9076063758×. Comparisons to it are reference-normalized across separate runs, not paired same-input measurements. Official within-run SGLang pairings and their intervals are authoritative. Missing results/timings are not inferred from CPU models.

Ledger: `experiments/remote-budget.json`. Exact source/result commits and per-shape evidence: `experiments/index.json`. Uncertain requests stay reserved until reconciled; never blindly retry a POST.

Latest account quota (distinct from this authorization), read 2026-10-24T16:35:09.571881+00:00: `{"scored": {"per_day": 10, "used": 0, "left": 10}, "checks": {"per_day": 20, "used": 11, "left": 9}}`.
