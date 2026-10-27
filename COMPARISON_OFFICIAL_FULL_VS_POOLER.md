# P2-02-full-composition vs P2-01-pooler-layout — saved H200 records only

P2-01-pooler-layout: job eeec05d897ef4946, 1.1290207x SGLang, interval [1.1120853945463307, 1.145956015700026]
P2-02-full-composition: job 48b1357016134e52, 1.1486500x SGLang, interval [1.1314202030609661, 1.165879701631351]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 14.3010 | 12.0471 | 1.15348 |
| 8 | 14.0731 | 14.5510 | 0.97484 |
| 16 | 14.4810 | 14.5134 | 0.98552 |
| 32 | 14.4570 | 14.3957 | 1.00204 |
| 128 | 16.9906 | 16.7119 | 1.00998 |
| 256 | 19.5361 | 20.1286 | 0.99590 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.089213 | 1.212103 |
| decode 16 | 1.083234 | 1.086826 |
| decode 32 | 1.087329 | 1.096358 |
| verify 1 | 1.347514 | 1.350693 |
| verify 16 | 1.106820 | 1.098073 |
| verify 32 | 1.082443 | 1.072227 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
