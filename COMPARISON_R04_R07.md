# R07-router-native-wide vs R04-router-triton16 — saved H200 records only

R04-router-triton16: job e3bc1481851f4a23, 1.0459769x SGLang, interval [1.0302872752554082, 1.0616665831312073]
R07-router-native-wide: job 26ea99ee0869482a, 1.0991970x SGLang, interval [1.0827090458817419, 1.1156849559085968]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 9.1967 | 9.2852 | 0.99047 |
| 8 | 9.2721 | 9.1417 | 1.01364 |
| 16 | 14.4969 | 9.2756 | 1.56119 |
| 32 | 14.6392 | 9.6759 | 1.57682 |
| 128 | 16.4662 | 10.3692 | 1.58487 |
| 256 | 17.6067 | 10.9153 | 1.59691 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.013796 | 1.019832 |
| decode 16 | 1.003851 | 1.073936 |
| decode 32 | 1.010039 | 1.083296 |
| verify 1 | 1.289262 | 1.298175 |
| verify 16 | 0.995057 | 1.070246 |
| verify 32 | 0.993083 | 1.069997 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
