# R08-router-split32 vs R04-router-triton16 — saved H200 records only

R04-router-triton16: job e3bc1481851f4a23, 1.0459769x SGLang, interval [1.0302872752554082, 1.0616665831312073]
R08-router-split32: job ab74f25caa2e46a8, 1.0617021x SGLang, interval [1.0457765729614412, 1.0776276360973225]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 9.1967 | 7.3016 | 1.27726 |
| 8 | 9.2721 | 6.8717 | 1.34351 |
| 16 | 14.4969 | 14.6561 | 0.99453 |
| 32 | 14.6392 | 15.0783 | 0.99715 |
| 128 | 16.4662 | 16.6252 | 1.00791 |
| 256 | 17.6067 | 17.4248 | 0.99484 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.013796 | 1.043892 |
| decode 16 | 1.003851 | 0.997077 |
| decode 32 | 1.010039 | 1.005218 |
| verify 1 | 1.289262 | 1.352192 |
| verify 16 | 0.995057 | 1.012933 |
| verify 32 | 0.993083 | 0.999431 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
