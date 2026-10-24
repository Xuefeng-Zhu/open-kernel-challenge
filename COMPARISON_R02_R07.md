# R07-router-native-wide vs R02-router-wide — saved H200 records only

R02-router-wide: job 89fe00c5034a4198, 1.0476946x SGLang, interval [1.031979203716431, 1.063410042408302]
R07-router-native-wide: job 26ea99ee0869482a, 1.0991970x SGLang, interval [1.0827090458817419, 1.1156849559085968]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 10.5051 | 9.2852 | 1.10043 |
| 8 | 26.0400 | 9.1417 | 2.84497 |
| 16 | 9.6031 | 9.2756 | 1.01644 |
| 32 | 9.8180 | 9.6759 | 1.02319 |
| 128 | 10.0582 | 10.3692 | 0.97963 |
| 256 | 11.3033 | 10.9153 | 1.01873 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.007337 | 1.019832 |
| decode 16 | 1.081297 | 1.073936 |
| decode 32 | 1.066095 | 1.083296 |
| verify 1 | 0.995165 | 1.298175 |
| verify 16 | 1.071094 | 1.070246 |
| verify 32 | 1.068488 | 1.069997 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
