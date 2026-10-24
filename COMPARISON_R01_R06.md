# R06-pooler-triton8 vs R01-router-control — saved H200 records only

R01-router-control: job 59abe5825690427c, 1.0012943x SGLang, interval [0.9862748867495631, 1.0163137157876208]
R06-pooler-triton8: job b70d87fc51864276, 0.9930291x SGLang, interval [0.9781337122250308, 1.0079245867090418]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 21.6569 | 22.7957 | 0.95501 |
| 8 | 21.1521 | 23.0604 | 0.92696 |
| 16 | 21.5553 | 21.3583 | 1.00697 |
| 32 | 21.5518 | 22.0162 | 0.99551 |
| 128 | 25.0790 | 25.1497 | 0.97111 |
| 256 | 30.6833 | 30.9291 | 0.99051 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 0.986818 | 0.987536 |
| decode 16 | 0.999472 | 1.003426 |
| decode 32 | 1.011325 | 0.995623 |
| verify 1 | 1.006305 | 0.972771 |
| verify 16 | 1.004900 | 0.993358 |
| verify 32 | 0.999124 | 1.005826 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
