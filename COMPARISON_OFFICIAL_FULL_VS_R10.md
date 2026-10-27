# P2-02-full-composition vs R10-final-composition — saved H200 records only

R10-final-composition: job 02e099c3724e4a05, 1.1207320x SGLang, interval [1.1039210682982272, 1.137543029769239]
P2-02-full-composition: job 48b1357016134e52, 1.1486500x SGLang, interval [1.1314202030609661, 1.165879701631351]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 6.8327 | 4.1234 | 1.69937 |
| 8 | 7.1360 | 6.9891 | 0.97523 |
| 16 | 18.7354 | 18.8572 | 0.98511 |
| 32 | 18.6157 | 18.9696 | 0.98981 |
| 128 | 19.6400 | 19.8536 | 0.98682 |
| 256 | 24.1363 | 24.4160 | 0.98985 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.108670 | 1.212103 |
| decode 16 | 1.080303 | 1.086826 |
| decode 32 | 1.070531 | 1.096358 |
| verify 1 | 1.339258 | 1.350693 |
| verify 16 | 1.078765 | 1.098073 |
| verify 32 | 1.069726 | 1.072227 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
