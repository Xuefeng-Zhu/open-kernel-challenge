# R04-router-triton16 vs R03-router-dispatch — saved H200 records only

R03-router-dispatch: job 718e0730e1bd4c30, 1.0267804x SGLang, interval [1.011378678442847, 1.0421820899690248]
R04-router-triton16: job e3bc1481851f4a23, 1.0459769x SGLang, interval [1.0302872752554082, 1.0616665831312073]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 10.1601 | 9.1967 | 1.11171 |
| 8 | 14.5391 | 9.2721 | 1.54993 |
| 16 | 14.6802 | 14.4969 | 1.01597 |
| 32 | 15.3670 | 14.6392 | 1.01564 |
| 128 | 16.5844 | 16.4662 | 0.99163 |
| 256 | 17.5998 | 17.6067 | 0.99509 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 0.998140 | 1.013796 |
| decode 16 | 0.998580 | 1.003851 |
| decode 32 | 0.994664 | 1.010039 |
| verify 1 | 1.179015 | 1.289262 |
| verify 16 | 1.002382 | 0.995057 |
| verify 32 | 1.000143 | 0.993083 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
