# R10-final-composition vs R08-router-split32 — saved H200 records only

R08-router-split32: job ab74f25caa2e46a8, 1.0617021x SGLang, interval [1.0457765729614412, 1.0776276360973225]
R10-final-composition: job 02e099c3724e4a05, 1.1207320x SGLang, interval [1.1039210682982272, 1.137543029769239]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 7.3016 | 7.1789 | 0.99985 |
| 8 | 6.8717 | 7.4584 | 0.93956 |
| 16 | 14.6561 | 9.1754 | 1.59733 |
| 32 | 15.0783 | 9.4573 | 1.57755 |
| 128 | 16.6252 | 10.2562 | 1.59288 |
| 256 | 17.4248 | 11.0143 | 1.58201 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.043892 | 1.108670 |
| decode 16 | 0.997077 | 1.080303 |
| decode 32 | 1.005218 | 1.070531 |
| verify 1 | 1.352192 | 1.339258 |
| verify 16 | 1.012933 | 1.078765 |
| verify 32 | 0.999431 | 1.069726 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
