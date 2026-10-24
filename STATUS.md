# Current experiment checkpoint

Current continuation: see MODAL_STAGE2_RESULTS.md. Unpooler019 and independent
QKV020 both completed/passed and stopped; all native correctness checks pass.
Latest cumulative Modal billing$0.21credits/net$0, conservative bound$0.3286below
$2. No new official check or push. Report before expanding experiments.

## Current local continuation

Second projection cycle: P2-01 eeec05d897ef4946 passed, official1.1290207051x, 1/10used/9remain, dispatch paused. Modal first smoke completed and stopped, app ap-ukvOyTeF1ztdOLggl5xJee, all native checks pass; see MODAL_RESULTS.md. One Modal initial allowance consumed, no new runs or pushes. The table below is historical first-cycle evidence.

## Preserved first-cycle evidence

10/10 new checks admitted/reserved; 10 terminal, 0 pending, 0 uncertain, 0 allowances remain.
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
| R09-pooler-simt | `97aa5690ff134369` | done | 1.0076197x |
| R10-final-composition | `02e099c3724e4a05` | done | 1.1207320x |

Exact source/result hashes, confidence intervals, correctness, per-shape times and experiment qualifications: `experiments/index.json`, `REMOTE_RESULTS.md`, `BEST.md`.
The ledger records each read timestamp and quota; queued jobs are not completed results.

Most recent observed job state: 2026-10-24T16:37:47.378717+00:00.
Account quota read at 2026-10-24T16:40:26.967388+00:00: `{"scored": {"per_day": 10, "used": 0, "left": 10}, "checks": {"per_day": 20, "used": 11, "left": 9}}`.

The historical R06 denial was resolved by specific new human approval; its single retry was accepted. Do not repeat it; preserve `DENIED_R06.md` history/resolution.
