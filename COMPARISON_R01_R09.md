# R09-pooler-simt vs R01-router-control — saved H200 records only

R01-router-control: job 59abe5825690427c, 1.0012943x SGLang, interval [0.9862748867495631, 1.0163137157876208]
R09-pooler-simt: job 97aa5690ff134369, 1.0076197x SGLang, interval [0.9925054300746219, 1.0227340218535443]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 21.6569 | 18.3032 | 1.19824 |
| 8 | 21.1521 | 21.4543 | 0.99412 |
| 16 | 21.5553 | 21.2393 | 1.01337 |
| 32 | 21.5518 | 21.8979 | 1.00016 |
| 128 | 25.0790 | 25.3753 | 0.99400 |
| 256 | 30.6833 | 30.7063 | 1.01270 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 0.986818 | 1.066917 |
| decode 16 | 0.999472 | 1.003430 |
| decode 32 | 1.011325 | 1.004229 |
| verify 1 | 1.006305 | 0.991044 |
| verify 16 | 1.004900 | 0.983286 |
| verify 32 | 0.999124 | 0.998980 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
