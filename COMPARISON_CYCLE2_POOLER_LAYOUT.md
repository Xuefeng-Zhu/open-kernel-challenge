# P2-01-pooler-layout vs R10-final-composition — saved H200 records only

R10-final-composition: job 02e099c3724e4a05, 1.1207320x SGLang, interval [1.1039210682982272, 1.137543029769239]
P2-01-pooler-layout: job eeec05d897ef4946, 1.1290207x SGLang, interval [1.1120853945463307, 1.145956015700026]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 18.1286 | 18.4982 | 0.97852 |
| 8 | 20.8359 | 20.2619 | 1.02754 |
| 16 | 20.7819 | 20.7076 | 1.02765 |
| 32 | 21.6892 | 20.6395 | 1.04072 |
| 128 | 24.7756 | 23.0184 | 1.07911 |
| 256 | 28.7850 | 27.9043 | 1.03558 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.108670 | 1.089213 |
| decode 16 | 1.080303 | 1.083234 |
| decode 32 | 1.070531 | 1.087329 |
| verify 1 | 1.339258 | 1.347514 |
| verify 16 | 1.078765 | 1.106820 |
| verify 32 | 1.069726 | 1.082443 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
