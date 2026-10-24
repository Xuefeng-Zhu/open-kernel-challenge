# Current experiment checkpoint

9/10 new checks admitted/reserved; 8 terminal, 1 pending, 0 uncertain, 1 allowances remain.
$0; no scored/public/final entry. Original baseline is separately preserved.

| Experiment | Job | Status | Official speedup vs SGLang |
|---|---|---|---|
| R01-router-control | `59abe5825690427c` | done | 1.0012943x |
| R02-router-wide | `89fe00c5034a4198` | done | 1.0476946x |
| R03-router-dispatch | `718e0730e1bd4c30` | done | 1.0267804x |
| R04-router-triton16 | `e3bc1481851f4a23` | done | 1.0459769x |
| R05-wide-layout | `1631810d19af4af4` | done | 1.0509508x |
| R06-pooler-triton8 | `b70d87fc51864276` | done | 0.9930291x |
| R07-router-native-wide | `26ea99ee0869482a` | done | 1.0991970x |
| R08-router-split32 | `ab74f25caa2e46a8` | done | 1.0617021x |
| R09-pooler-simt | `97aa5690ff134369` | queued | unmeasured |

Exact source/result hashes, confidence intervals, correctness, per-shape times and experiment qualifications: `experiments/index.json`, `REMOTE_RESULTS.md`, `BEST.md`.
The ledger records each read timestamp and quota; queued jobs are not completed results.

The historical R06 denial was resolved by specific new human approval; its single retry was accepted. Do not repeat it; preserve `DENIED_R06.md` history/resolution.
